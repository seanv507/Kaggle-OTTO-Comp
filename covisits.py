"""Covisitation matrices from Chris Deotte's part of the 3rd place OTTO solution.

Each `make_covisits_<VER>` converts train/covisit_matrices/{cpu,gpu}-<VER>.ipynb from
cdeotte/Kaggle-OTTO-Comp (the notebooks run by train/covisit_matrices/script.ipynb).
They all return `aid, aid_y, value`: the top `top` aid_y per aid, sorted by aid,
value descending, aid_y.

Conventions, shared by every matrix:
- Events must be sorted by session, then ts (as data/*.parquet and the reference's
  train_data files are). Row order stands in for the reference's sort order.
- After a session self-join the x event's columns are unsuffixed and the y event's
  columns end in `_y` (the reference's `_x` / `_y`).
- Duplicate pairs: the reference sorts, merges and then calls
  `drop_duplicates(['session','aid_x','aid_y'])`, which keeps the first row. On the GPU
  notebooks cudf's merge order is unspecified, so which copy survives was arbitrary. Here
  it follows pandas' merge order (left rows in order, then right rows in order), which
  is what the sort before the merge implies: after an ascending sort the earliest copy
  is kept, after a descending sort the latest. Where no pair filter comes before the
  dedup, both sides are deduplicated before the join, which gives the same result with
  a much smaller join.
- Times: the reference uses epoch seconds; here `ts` is Datetime(ms). Cutoffs are
  compared with Python datetimes, gaps use `dt.total_seconds`. The GPU engine rejects
  datetime +/- literal durations and duration / duration.
"""

import datetime
import math
from typing import TYPE_CHECKING, Callable, TypeVar

import polars as pl
from loguru import logger

from event_types import CARTS, CLICKS, ORDERS

if TYPE_CHECKING:
    from polars._typing import EngineType

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)
Events = pl.LazyFrame | pl.DataFrame

# output order of covisitation matrices: aid, value descending, aid_y (tie-break)
TOP_ORDER = ["aid", "value", "aid_y"]
TOP_DESCENDING = [False, True, False]

start_train = datetime.datetime.fromisoformat("2022-07-31 22:00:00")  # 1659304800
end_train = datetime.datetime.fromisoformat("2022-08-28 21:59:59")  # 1661716799
end_test = datetime.datetime.fromisoformat("2022-09-04 21:59:51")  # 1662328791
EPOCH = datetime.datetime(1970, 1, 1)
START_TRAIN_S = int((start_train - EPOCH).total_seconds())
END_TEST_S = int((end_test - EPOCH).total_seconds())


def in_last_days(days: int) -> pl.Expr:
    """ts > end_test - days (the reference's `df.ts > 1662328791 - 60*60*24*days`)."""
    return pl.col("ts") > end_test - datetime.timedelta(days=days)


carts_orders_filter = pl.col("type").is_in([CARTS, ORDERS])

# --- expressions on a session self-join ---
aid_diff_filter = pl.col("aid") != pl.col("aid_y")
# note: avoid .abs() on Duration columns - unsupported by the cudf GPU engine
ts_diff = pl.col("ts") - pl.col("ts_y")
within_14_days_filter = ts_diff.is_between(-pl.duration(days=14), pl.duration(days=14))
# hours from x to y, positive when y is later
# (not / pl.duration(hours=1): the GPU engine can't divide durations)
dt_hours = (pl.col("ts_y") - pl.col("ts")).dt.total_seconds(fractional=True) / 3600
hour_diff = dt_hours.abs()


def half_life(hours: pl.Expr, half_life_hours: float = 1) -> pl.Expr:
    return 0.5 ** (hours / half_life_hours)


# --- row order helpers ---
ROW = "_k"  # row index; after a self-join also ROW + "_y"


def with_row(events: pl.LazyFrame) -> pl.LazyFrame:
    return events.with_row_index(ROW)


def keep_one_per(events: pl.LazyFrame, keys: list[str], last: bool) -> pl.LazyFrame:
    """Keep the first (or last) row of each `keys` group, by the ROW column."""
    k = pl.col(ROW)
    return events.filter(k == (k.max() if last else k.min()).over(keys))


def dedup_pairs(
    pairs: pl.LazyFrame, last: bool, keys: tuple[str, ...] = ("session", "aid", "aid_y")
) -> pl.LazyFrame:
    """Keep one row per `keys` after a self-join: the one pandas' merge order puts first.

    Pandas orders merge output by left row, then right row. `last=False` keeps the
    smallest (x row, y row), i.e. the reference sorted ascending; `last=True` the
    largest, i.e. the reference sorted descending.
    """
    c = pl.col(ROW).cast(pl.Int64) * 2**32 + pl.col(ROW + "_y").cast(pl.Int64)
    return pairs.filter(c == (c.max() if last else c.min()).over(list(keys)))


def session_self_join(events: Frame, other: Frame | None = None) -> Frame:
    return events.join(events if other is None else other, on="session", suffix="_y")


def sum_value(pairs: pl.LazyFrame) -> pl.LazyFrame:
    return pairs.group_by(["aid", "aid_y"]).agg(pl.col("value").sum())


def iterate_session_chunks(events: pl.LazyFrame, session_chunk: int):
    """Yield (chunk_start, lazy chunk) for contiguous session ranges.

    `events` should be sorted by session so each range filter is cheap
    (and a parquet scan can skip row groups using statistics).
    """
    lo, hi = (
        events.select(pl.col("session").min().alias("lo"), pl.col("session").max().alias("hi"))
        .collect()
        .row(0)
    )
    first = (lo // session_chunk) * session_chunk
    n_chunks = (hi - first) // session_chunk + 1
    logger.info(f"sessions {lo:,}..{hi:,} in {n_chunks} chunks of {session_chunk:,}")
    for i, start in enumerate(range(first, hi + 1, session_chunk)):
        logger.info(f"chunk {i + 1}/{n_chunks} start {start:,}")
        yield start, events.filter(
            pl.col("session").is_between(start, start + session_chunk, closed="left")
        )


def top_covisit(df: Frame, top: int = 40) -> Frame:
    """For each aid keep the `top` aid_y with the highest `value` (ties: lowest aid_y).

    Output is sorted by aid, value descending, aid_y. Sort + group_by.head is ~10x faster
    on CPU than a rank(...).over("aid") filter (1.4s vs 13s on the 117M-row 90 matrix);
    the window version also ran out of GPU memory on a 4GB card.
    """
    return (
        df.sort(TOP_ORDER, descending=TOP_DESCENDING)
        .group_by("aid", maintain_order=True)
        .head(top)
    )


def build_covisits(
    events: Events,
    pairs: Callable[[pl.LazyFrame], pl.LazyFrame],
    session_chunk: int | None,
    engine: "EngineType",
    top: int,
    agg_engine: "EngineType" = "streaming",
    reduce: Callable[[pl.LazyFrame], pl.LazyFrame] = sum_value,
) -> pl.DataFrame:
    """Run `pairs` per session chunk, combine the chunks with `reduce`, and keep the
    `top` aid_y per aid.

    events: LazyFrame (e.g. pl.scan_parquet) sorted by session, then ts.
    pairs: session events -> partial results (aid, aid_y, value, ...), per chunk.
    session_chunk: sessions per chunk; None runs one query over all sessions.
    engine: engine for the per-chunk pair queries - "gpu", "streaming" or "in-memory".
        "gpu" falls back to CPU (with a warning) if the GPU engine is unavailable;
        pass pl.GPUEngine(raise_on_fail=True) to error instead.
    top: keep the top aid_y for each aid by value (0 returns all pairs).
    agg_engine: engine for combining the chunks and taking the top. Defaults to CPU
        streaming: the chunk results are already in host RAM, and on the full 90 matrix
        (117M pairs) this step ran out of memory on a 4GB GPU, while streaming takes ~1.5s.
    reduce: concatenated chunk results -> aid, aid_y, value. Default sums `value`.
    Returns aid, aid_y, value sorted by aid, value descending, aid_y.
    """
    events = events.lazy()
    if session_chunk is None:
        chunks = [pairs(events).collect(engine=engine)]
    else:
        chunks = [
            pairs(chunk).collect(engine=engine)
            for _, chunk in iterate_session_chunks(events, session_chunk)
        ]
    logger.info("aggregating chunks")
    result = reduce(pl.concat(chunks).lazy())
    if top:
        result = top_covisit(result, top)
    else:
        result = result.sort(TOP_ORDER, descending=TOP_DESCENDING)
    return result.collect(engine=agg_engine)


def train_test_events(train_events: Events, test_events: Events) -> pl.LazyFrame:
    """Concatenate train and test events with a boolean `_test` column.

    Train sessions must all come before test sessions, as in the data.
    """
    return pl.concat(
        [
            train_events.lazy().with_columns(_test=pl.lit(False)),
            test_events.lazy().with_columns(_test=pl.lit(True)),
        ],
        how="vertical_relaxed",
    )


# ---------------------------------------------------------------------------
# 90, 95, 99: CPU notebooks (pandas)
# ---------------------------------------------------------------------------


def covisits_90_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """Per (aid, aid_y): number of sessions where both were carted/ordered within 14 days."""
    events = events.filter(carts_orders_filter)
    return (
        session_self_join(events)
        .filter(within_14_days_filter & aid_diff_filter)
        .unique(subset=["session", "aid", "aid_y"])  # 1 pair per session
        .group_by(["aid", "aid_y"])
        .agg(pl.len().alias("value"))
    )


def make_covisits_90(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Carts/orders covisitation counts (reference: train/covisit_matrices/cpu-90.ipynb).

    See build_covisits for the arguments.

    Timings on full train.parquet (RTX 2050 4GB, 2026-11-04, before `top` was added):
    gpu + no chunking 21.4s (fastest); streaming best at ~500k sessions/chunk 26.3s.
    With top=40: gpu + no chunking 31.2s.
    """
    return build_covisits(events, covisits_90_pairs, session_chunk, engine, top, agg_engine)


def last_n_first_per_aid(events: pl.LazyFrame, n: int = 30) -> pl.LazyFrame:
    """Keep each session's last `n` events, then each aid's first event among them.

    Relies on row order (sessions sorted, events in time order within a session).
    Self-joining the result reproduces the reference's pandas
    `groupby('session').tail(30)` -> merge -> `drop_duplicates(['session','aid_x','aid_y'])`
    (keep first): the first (aid_x, aid_y) row is aid_x's first event paired with aid_y's
    first event. Deduplicating before the join is also deterministic, unlike
    `.unique()` after it.

    Written with a row index and min/max windows because the GPU engine doesn't
    support int_range/cum_count in windows. `k + n > max` avoids UInt32 underflow.
    """
    k = pl.col("_k")
    return (
        events.with_row_index("_k")
        .filter(k + n > k.max().over("session"))
        .filter(k == k.min().over("session", "aid"))
        .drop("_k")
    )


# time decay with half-life 1 hour; carts/orders as the *second* event count 4x clicks
weight_95 = half_life(hour_diff) * pl.when(pl.col("type_y") == CLICKS).then(1).otherwise(4)


def covisits_95_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    """Per (aid, aid_y): time-decayed, type-weighted covisits within each session's last 30 events."""
    events = last_n_first_per_aid(events, 30)
    return (
        session_self_join(events)
        .filter(aid_diff_filter)
        .group_by(["aid", "aid_y"])
        .agg(weight_95.sum().alias("value"))
    )


def make_covisits_95(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Clicks covisitation, time decay + type weight (reference: train/covisit_matrices/cpu-95.ipynb).

    Differs from the reference in one way: the reference splits each data file into
    120 row chunks before taking tail(30), so a session spanning two chunks is
    treated as two sessions. Here sessions are never split.

    See build_covisits for the arguments.
    """
    return build_covisits(events, covisits_95_pairs, session_chunk, engine, top, agg_engine)


def covisits_99_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = keep_one_per(
        with_row(events.filter(in_last_days(28) & carts_orders_filter)), ["session", "aid"], last=False
    )
    return (
        session_self_join(e)
        .filter(aid_diff_filter)
        .group_by(["aid", "aid_y"])
        .agg(half_life(hour_diff).sum().alias("value"))
    )


def make_covisits_99(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Buy2buy (reference: cpu-99.ipynb, saved as top_40_buy2buy_v99).

    Carts/orders in the last 28 days; pairs aid != aid_y, 1 per session (each aid's
    earliest event); weight 0.5 ** hours apart. Data: train + test events.
    Like 95, the reference splits each file into 120 row chunks, which can split a
    session; here sessions are never split. See build_covisits for the arguments.
    """
    return build_covisits(events, covisits_99_pairs, session_chunk, engine, top, agg_engine)


# ---------------------------------------------------------------------------
# GPU notebooks: time-weighted pairs
# ---------------------------------------------------------------------------


def covisits_93_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    k = pl.col(ROW)
    e = with_row(events).filter(k + 30 > k.max().over("session"))  # last 30 events
    time_weight = 1 + 3 * (pl.col("ts").dt.epoch("s") - START_TRAIN_S) / (END_TEST_S - START_TRAIN_S)
    return (
        session_self_join(e)
        .filter(aid_diff_filter & (hour_diff < 24))
        .pipe(dedup_pairs, last=True)
        .group_by(["aid", "aid_y"])
        .agg((time_weight * half_life(hour_diff)).sum().alias("value"))
    )


def make_covisits_93(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 80,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Clicks, time-weighted (reference: gpu-93.ipynb, saved as top_80_aids_v93).

    Each session's last 30 events; pairs aid != aid_y less than 24h apart, 1 per
    session (latest copy); weight (1 + 3 * x's position in train..test time) *
    0.5 ** hours apart. Data: train + test events. See build_covisits for the arguments.
    """
    return build_covisits(events, covisits_93_pairs, session_chunk, engine, top, agg_engine)


def covisits_115_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = keep_one_per(with_row(events.filter(in_last_days(21))), ["session", "aid"], last=True)
    return (
        session_self_join(e)
        .group_by(["aid", "aid_y"])
        .agg(half_life(hour_diff).sum().alias("value"))
    )


def make_covisits_115(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """All events in the last 21 days (reference: gpu-115.ipynb, top_40_aids_v115).

    All pairs in a session, 1 per session (each aid's latest event), weight 0.5 ** hours
    apart. As in the reference there is no aid != aid_y filter, so each aid pairs with
    itself (weight 1 per session). Data: train + test events.
    See build_covisits for the arguments.
    """
    return build_covisits(events, covisits_115_pairs, session_chunk, engine, top, agg_engine)


def covisits_116_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = with_row(events.filter(in_last_days(28)))
    type_weight = pl.when(pl.col("type_y") == CLICKS).then(0.0).otherwise(1.0)
    return (
        session_self_join(e)
        .filter(dt_hours > 0)
        .pipe(dedup_pairs, last=True)
        .group_by(["aid", "aid_y"])
        .agg((type_weight * half_life(dt_hours)).sum().alias("value"))
    )


def make_covisits_116(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Events in the last 28 days, y after x (reference: gpu-116.ipynb, top_40_aids_v116).

    Pairs with y strictly later, 1 per session (latest copy); weight 0 if y is a click,
    else 0.5 ** hours apart. Click pairs stay in the matrix with value 0, as in the
    reference. Data: train + test events. See build_covisits for the arguments.
    """
    return build_covisits(events, covisits_116_pairs, session_chunk, engine, top, agg_engine)


def covisits_155_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = with_row(events)
    return (
        session_self_join(e, e.filter(carts_orders_filter))
        .filter(dt_hours > 0)
        .pipe(dedup_pairs, last=True)
        .group_by(["aid", "aid_y"])
        .agg(half_life(dt_hours, 6).sum().alias("value"))
    )


def make_covisits_155(
    test_events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Test week, any event -> later cart/order (reference: gpu-155.ipynb, top_40_aids_v155_d0).

    Pairs x -> y with y a cart/order strictly later, 1 per session (latest copy);
    weight 0.5 ** (hours / 6). No aid != aid_y filter. Data: test events only.
    See build_covisits for the arguments.
    """
    return build_covisits(test_events, covisits_155_pairs, session_chunk, engine, top, agg_engine)


def covisits_157_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = keep_one_per(with_row(events), ["session", "aid"], last=True)
    return (
        session_self_join(e)
        .group_by(["aid", "aid_y"])
        .agg(half_life(hour_diff, 6).sum().alias("value"))
    )


def make_covisits_157(
    test_events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 80,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Test week, all pairs (reference: gpu-157.ipynb, top_80_aids_v157_d0).

    All pairs in a session, 1 per session (latest copy), weight 0.5 ** (hours / 6).
    No aid != aid_y filter, so each aid pairs with itself. Data: test events only.
    See build_covisits for the arguments.
    """
    return build_covisits(test_events, covisits_157_pairs, session_chunk, engine, top, agg_engine)


def _to_carts_orders_pairs(x: pl.LazyFrame, events: pl.LazyFrame) -> pl.LazyFrame:
    """x events (latest copy per aid) paired with the session's carts/orders (latest copy)."""
    x = keep_one_per(x, ["session", "aid"], last=True)
    y = keep_one_per(events.filter(carts_orders_filter), ["session", "aid"], last=True)
    return (
        session_self_join(x, y)
        .group_by(["aid", "aid_y"])
        .agg(half_life(hour_diff).sum().alias("value"))
    )


# reference: sessions of at most 6 events in train files, 3 in test files
_short_session = pl.len().over("session") <= pl.when(pl.col("_test")).then(3).otherwise(6)


def covisits_165_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = with_row(events.filter(_short_session))
    return _to_carts_orders_pairs(e, e)


def covisits_166_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = with_row(events.filter(~_short_session))
    return _to_carts_orders_pairs(e, e)


def make_covisits_165(
    train_events: Events,
    test_events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Short sessions -> carts/orders (reference: gpu-165.ipynb, top_40_aids_v165).

    Sessions of <= 6 events (train) or <= 3 (test). Pairs any event -> a cart/order
    in the session (no time order, no aid != aid_y filter), 1 per session (latest
    copies); weight 0.5 ** hours apart. See build_covisits for the other arguments.
    """
    return build_covisits(
        train_test_events(train_events, test_events),
        covisits_165_pairs, session_chunk, engine, top, agg_engine,
    )


def make_covisits_166(
    train_events: Events,
    test_events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Long sessions -> carts/orders (reference: gpu-166.ipynb, top_40_aids_v166).

    As 165, for sessions of > 6 events (train) or > 3 (test).
    """
    return build_covisits(
        train_test_events(train_events, test_events),
        covisits_166_pairs, session_chunk, engine, top, agg_engine,
    )


# hour of day in UTC+2 (the reference adds 2h to ts before taking ts % 1 day)
_local_hour = (pl.col("ts").dt.hour() + 2) % 24


def covisits_167_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = with_row(events)
    return _to_carts_orders_pairs(e.filter(_local_hour < 14), e)


def covisits_168_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    e = with_row(events)
    return _to_carts_orders_pairs(e.filter(_local_hour >= 14), e)


def make_covisits_167(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Morning events -> carts/orders (reference: gpu-167.ipynb, top_40_aids_v167).

    x events before 14:00 UTC+2, paired with any cart/order in the session (no time
    order, no aid != aid_y filter), 1 per session (latest copies); weight 0.5 ** hours
    apart. Data: train + test events. See build_covisits for the arguments.
    """
    return build_covisits(events, covisits_167_pairs, session_chunk, engine, top, agg_engine)


def make_covisits_168(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Afternoon/evening events -> carts/orders (reference: gpu-168.ipynb, top_40_aids_v168).

    As 167, for x events from 14:00 UTC+2.
    """
    return build_covisits(events, covisits_168_pairs, session_chunk, engine, top, agg_engine)


# ---------------------------------------------------------------------------
# GPU notebooks: neighbouring events (217-235)
# ---------------------------------------------------------------------------


def _neighbour_pairs(events: pl.LazyFrame, lo: int, hi: int, directed: bool) -> pl.LazyFrame:
    """Pairs where aid_y's first event in the session is lo..hi events from an x event.

    Reference: sort by (session, ts) ascending, k = position, merge with
    drop_duplicates(['session','aid']) (each aid's first event), filter on k_y - k_x
    (or its absolute value) and aid_x != aid_y.
    """
    e = with_row(events)
    y = keep_one_per(e, ["session", "aid"], last=False)
    gap = pl.col(ROW + "_y").cast(pl.Int64) - pl.col(ROW).cast(pl.Int64)
    gap = gap if directed else gap.abs()
    return session_self_join(e, y).filter(gap.is_between(lo, hi) & aid_diff_filter)


def _count_sessions(pairs: pl.LazyFrame) -> pl.LazyFrame:
    return (
        pairs.unique(subset=["session", "aid", "aid_y"])
        .group_by(["aid", "aid_y"])
        .agg(pl.len().alias("value"))
    )


def _sum_half_life_first(pairs: pl.LazyFrame) -> pl.LazyFrame:
    return (
        pairs.pipe(dedup_pairs, last=False)
        .group_by(["aid", "aid_y"])
        .agg(half_life(hour_diff).sum().alias("value"))
    )


def covisits_217_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    return _count_sessions(_neighbour_pairs(events, 1, 1, directed=False))


def covisits_220_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    return _count_sessions(_neighbour_pairs(events, 1, 2, directed=False))


def covisits_226_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    return _count_sessions(_neighbour_pairs(events.filter(carts_orders_filter), 1, 3, directed=True))


def covisits_232_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    return _sum_half_life_first(_neighbour_pairs(events, 1, 3, directed=False))


def covisits_235_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    return _sum_half_life_first(_neighbour_pairs(events, 1, 6, directed=False))


def make_covisits_217(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Adjacent events (reference: gpu-217.ipynb, top_40_aids_v217).

    Number of sessions where an event of aid is next to aid_y's first event.
    Data: train + test events. See _neighbour_pairs and build_covisits.
    """
    return build_covisits(events, covisits_217_pairs, session_chunk, engine, top, agg_engine)


def make_covisits_220(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 80,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Events up to 2 apart (reference: gpu-220.ipynb, top_80_aids_v220).

    As 217, with aid_y's first event 1 or 2 events before or after.
    """
    return build_covisits(events, covisits_220_pairs, session_chunk, engine, top, agg_engine)


def make_covisits_226(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 80,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Carts/orders, next 1-3 carts/orders (reference: gpu-226.ipynb, top_80_aids_v226).

    As 217 on carts/orders only (positions counted among them), with aid_y's first
    cart/order 1 to 3 positions *after* the x event.
    """
    return build_covisits(events, covisits_226_pairs, session_chunk, engine, top, agg_engine)


def make_covisits_232(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 80,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Events up to 3 apart, time-weighted (reference: gpu-232.ipynb, top_80_aids_v232).

    Pairs as 217 with aid_y's first event 1 to 3 events away, 1 per session (the
    earliest x event); weight 0.5 ** hours apart. Data: train + test events.
    """
    return build_covisits(events, covisits_232_pairs, session_chunk, engine, top, agg_engine)


def make_covisits_235(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 80,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Events up to 6 apart, time-weighted (reference: gpu-235.ipynb, top_80_aids_v235).

    As 232, with aid_y's first event 1 to 6 events away.
    """
    return build_covisits(events, covisits_235_pairs, session_chunk, engine, top, agg_engine)


# ---------------------------------------------------------------------------
# GPU notebook 239: closest later cart/order, per reference file
# ---------------------------------------------------------------------------


def _with_file_groups(events: Events, parts: int, offset: int) -> pl.LazyFrame:
    """Add `_group`: which of `parts` equal session ranges each event's session is in.

    Mirrors data/make_train_valid.ipynb, which cut the unique sessions into `parts`
    files of ceil(n_sessions / parts) sessions. Exact when session ids are contiguous.
    """
    events = events.lazy()
    lo, hi = events.select(pl.col("session").min(), pl.col("session").max().alias("hi")).collect().row(0)
    size = math.ceil((hi - lo + 1) / parts)
    return events.with_columns(_group=(pl.col("session") - lo) // size + offset)


def covisits_239_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    return (
        session_self_join(events, events.filter(carts_orders_filter))
        .filter((dt_hours > 0) & (dt_hours < 2))
        .group_by(["_group", "aid", "aid_y"])
        .agg(half_life(dt_hours).max().alias("value"))  # the closest y
    )


def _reduce_239(pairs: pl.LazyFrame) -> pl.LazyFrame:
    # a file's sessions may span chunks: combine within a file first, then sum files
    return sum_value(pairs.group_by(["_group", "aid", "aid_y"]).agg(pl.col("value").max()))


def make_covisits_239(
    train_events: Events,
    test_events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 20,
    agg_engine: "EngineType" = "streaming",
    train_parts: int = 100,
    test_parts: int = 20,
) -> pl.DataFrame:
    """Any event -> cart/order within the next 2 hours (reference: gpu-239.ipynb, top_20_aids_v239).

    The reference keeps one pair per (aid_x, aid_y) *per data file*, the one with y
    closest after x, weights it 0.5 ** hours apart and sums over files. Its files cut
    train sessions into 100 parts and test sessions into 20 (train_parts, test_parts).
    No aid != aid_y filter. See build_covisits for the other arguments.
    """
    events = pl.concat(
        [_with_file_groups(train_events, train_parts, 0), _with_file_groups(test_events, test_parts, train_parts)],
        how="vertical_relaxed",
    )
    return build_covisits(
        events, covisits_239_pairs, session_chunk, engine, top, agg_engine, reduce=_reduce_239
    )


# ---------------------------------------------------------------------------
# GPU notebooks 700, 701: first event of recent sessions
# ---------------------------------------------------------------------------


def _first_event_of_recent_sessions(events: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Sessions starting in the last 21 days: (their first event, all their events)."""
    k = pl.col(ROW)
    e = with_row(events).filter(pl.col("ts").min().over("session") > end_test - datetime.timedelta(days=21))
    return e.filter(k == k.min().over("session")), e


def covisits_700_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    first, e = _first_event_of_recent_sessions(events)
    return (
        session_self_join(first, e.filter(carts_orders_filter))
        .filter(aid_diff_filter)
        .unique(subset=["session", "aid", "aid_y", "type_y"])
        .group_by(["aid", "aid_y"])
        .agg(pl.len().alias("value"))
    )


def covisits_701_pairs(events: pl.LazyFrame) -> pl.LazyFrame:
    first, e = _first_event_of_recent_sessions(events)
    y = keep_one_per(e, ["session", "aid", "type"], last=False)
    return (
        session_self_join(first, y)
        .filter(aid_diff_filter)
        .group_by(["aid", "aid_y"])
        .agg(half_life(hour_diff).sum().alias("value"))
    )


def make_covisits_700(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """First event -> carts/orders (reference: gpu-700.ipynb, top_40_aids_v700).

    Sessions starting in the last 21 days. Counts, per (first aid, aid_y), the sessions
    where aid_y was carted plus those where it was ordered. Data: train + test events.
    """
    return build_covisits(events, covisits_700_pairs, session_chunk, engine, top, agg_engine)


def make_covisits_701(
    events: Events,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """First event -> any event, time-weighted (reference: gpu-701.ipynb, top_40_aids_v701).

    Sessions starting in the last 21 days. Pairs the first event with each (aid, type)'s
    earliest event in the session; weight 0.5 ** hours apart, summed over sessions
    and types. Data: train + test events.
    """
    return build_covisits(events, covisits_701_pairs, session_chunk, engine, top, agg_engine)
