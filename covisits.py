import datetime
from pathlib import Path
import polars as pl
from loguru import logger
from typing import TYPE_CHECKING, Callable, TypeVar

from event_types import CARTS, CLICKS, ORDERS

if TYPE_CHECKING:
    from polars._typing import EngineType


aid_diff_filter = pl.col("aid")!=pl.col("aid_y")
# note: avoid .abs() on Duration columns - unsupported by the cudf GPU engine
ts_diff = pl.col("ts") - pl.col("ts_y")
within_14_days_filter = ts_diff.is_between(-pl.duration(days=14), pl.duration(days=14))
start_train = datetime.datetime.fromisoformat('2022-07-31 22:00:00') 
end_train = datetime.datetime.fromisoformat('2022-08-28 21:59:59')
end_test = datetime.datetime.fromisoformat('2022-09-04 21:59:51') 
last_21_days_filter = (pl.col('ts')> end_test - pl.duration(days=21))
last_28_days_filter = (pl.col('ts')> end_test -pl.duration(days=28))
            # 1659304800   2022-07-31 22:00:00
            # 1661716799 "2022-08-28 21:59:59"	
            # 1662328791 '2022-09-04 21:59:51'


carts_orders_filter = pl.col("type").is_in([CARTS, ORDERS])
within_1_day_filter = ts_diff.is_between(-pl.duration(days=1), pl.duration(days=1))
# not ts_diff / pl.duration(hours=1): the GPU engine can't divide durations
hour_diff = (ts_diff.dt.total_seconds(fractional=True) / 3600).abs()
weight_93_ts = 1 + 3*(pl.col('ts') - start_train)/(end_test-start_train)

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)

# output order of covisitation matrices: aid, value descending, aid_y (tie-break)
TOP_ORDER = ["aid", "value", "aid_y"]
TOP_DESCENDING = [False, True, False]


def session_self_join(events: Frame) -> Frame:
    return events.join(events, on="session", suffix="_y")

def filter(events: pl.DataFrame, filter):
    pre = events.height
    events = events.filter(filter)
    logger.info(f'selecting {events.height:,} from {pre:,} with {filter}')
    return events


def iterate_sessions(events: pl.DataFrame, session_chunk: int):
    max_session = events.get_column('session').max()
    n_sessions =  events.get_column('session').n_unique()
    logger.info(f'Unique sessions {n_sessions} max {max_session} chunk{session_chunk}')
    for i_gp, (gp, df) in enumerate(events.group_by(pl.col('session')//session_chunk)):
        logger.info(f'Chunk {i_gp} {gp} of {max_session//session_chunk}')
        yield i_gp, gp, df


def update_df_agg(df, df_agg):
    logger.info('joining results')
    if df_agg is None:
        df_agg = df
    else:
        df_agg = pl.concat((df_agg,df)).group_by(["aid", "aid_y"]).agg(pl.col('value').sum())
    return df_agg

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


def build_covisits(
    events: pl.LazyFrame | pl.DataFrame,
    pairs: Callable[[pl.LazyFrame], pl.LazyFrame],
    session_chunk: int | None,
    engine: "EngineType",
    top: int,
    agg_engine: "EngineType" = "streaming",
) -> pl.DataFrame:
    """Run `pairs` (session events -> aid, aid_y, value) per session chunk, sum the chunks,
    and keep the `top` aid_y per aid.

    events: LazyFrame (e.g. pl.scan_parquet) sorted by session.
    session_chunk: sessions per chunk; None runs one query over all sessions.
    engine: engine for the per-chunk pair queries - "gpu", "streaming" or "in-memory".
        "gpu" falls back to CPU (with a warning) if the GPU engine is unavailable;
        pass pl.GPUEngine(raise_on_fail=True) to error instead.
    top: keep the top aid_y for each aid by value (0 returns all pairs).
    agg_engine: engine for summing the chunks and taking the top. Defaults to CPU
        streaming: the chunk results are already in host RAM, and on the full 90 matrix
        (117M pairs) this step ran out of memory on a 4GB GPU, while streaming takes ~1.5s.
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
    result = (
        pl.concat(chunks)
        .lazy()
        .group_by(["aid", "aid_y"])
        .agg(pl.col("value").sum())
    )
    if top:
        result = top_covisit(result, top)
    else:
        result = result.sort(TOP_ORDER, descending=TOP_DESCENDING)
    return result.collect(engine=agg_engine)


def make_covisits_90(
    events: pl.LazyFrame | pl.DataFrame,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
) -> pl.DataFrame:
    """Carts/orders covisitation counts (reference: train/covisit_matrices/cpu-90.ipynb).

    See build_covisits for the arguments.

    Timings on full train.parquet (RTX 2050 4GB, 2026-11-04, before `top` was added):
    gpu + no chunking 21.4s (fastest); streaming best at ~500k sessions/chunk 26.3s.
    """
    return build_covisits(events, covisits_90_pairs, session_chunk, engine, top)


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
weight_95 = (1 / 2) ** hour_diff * pl.when(pl.col("type_y") == CLICKS).then(1).otherwise(4)


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
    events: pl.LazyFrame | pl.DataFrame,
    session_chunk: int | None = None,
    engine: "EngineType" = "gpu",
    top: int = 40,
) -> pl.DataFrame:
    """Clicks covisitation, time decay + type weight (reference: train/covisit_matrices/cpu-95.ipynb).

    Differs from the reference in one way: the reference splits each data file into
    120 row chunks before taking tail(30), so a session spanning two chunks is
    treated as two sessions. Here sessions are never split.

    See build_covisits for the arguments.
    """
    return build_covisits(events, covisits_95_pairs, session_chunk, engine, top)


def aggregate_covisits(ver: int):
    Path(f'data/out/covisits_{ver}/agg/').mkdir(parents=True, exist_ok=True)
    for i in range(19):
        print(i)
        (
            pl.scan_parquet(f'data/out/covisits_{ver}/unagg/aid_seg={i}/')
            .group_by(["aid", "aid_y"])
            .agg(pl.col('value').sum())
            .sort(['aid','value'],descending = [False, True])
            .sink_parquet(f'data/out/covisits_{ver}/agg/aid_seg={i:02d}.parquet')
        )


# with open(f'../../data/covisit_matrices/top_40_orders_carts_v{VER}.pkl', 'wb') as f:

# pandas CPU times: user 2h 56min 20s, sys: 42.6 s, total: 2h 57min 3s
# Wall time: 3h 27s

# using polars about 10 minutes

def make_covisits_99():
    session_chunk = 1_000_000
    events =  (
        filter(pl.read_parquet('data/train.parquet'),
            #1662328791
            last_28_days_filter & 
            carts_orders_filter)
    )
    df_agg = None
    
    for i_gp,gp, df in iterate_sessions(events, session_chunk):
        df = (
            filter(session_self_join(df), aid_diff_filter)
            .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
            .group_by(["aid", "aid_y"]).agg(
                ((1/2)**(hour_diff)).alias('value').sum()
            )
        )
        df_agg = update_df_agg(df, df_agg)
    df_agg = df_agg.sort(['aid','value'],descending=[False, True])
    return df_agg


# Chunks:   0%|          | 0/120 [00:00<?, ?it/s]

# CPU times: user 9min 25s, sys: 8.75 s, total: 9min 33s
# Wall time: 9min 29s


#with open(f'../../data/covisit_matrices/top_40_buy2buy_v{VER}.pkl', 'wb') as f:


def make_covisits_93():
    events =  pl.read_parquet('data/train.parquet').group_by("session").tail(30)
    dfs = {}
    session_chunk = 500_000
    for i_gp,gp, df in iterate_sessions(events, session_chunk):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(aid_diff_filter & within_1_day_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            (weight_93_ts*(1/2)**(hour_diff)).alias('value').sum()
        )
    )
    df_agg = (
        pl.concat(dfs.values())
        .with_columns(aid_seg=pl.col('aid')//1_00_000)
    )
    return df_agg
# takes around 5 minutes on cpu (including save and regroup)
#df_agg_p = df_agg.partition_by('aid_seg',maintain_order=False, include_key=True, as_dict=True)
# memory error

    # df_agg.write_parquet(pl.PartitionBy('data/out/covisits_93/')
    
    #     .group_by(["aid", "aid_y"])
    #     .agg(pl.col('value').sum())
    #     .sort(['aid','value'], descending=[False, True])
    # )
    # return df_agg
# run on gpu
# CPU times: user 55.5 s, sys: 41.1 s, total: 1min 36s
# Wall time: 1min 40s
# output top 80


            # df['wgt'] = 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800)
            # df['wgt'] = df['wgt'] * ((1/2)**( (df.ts_x - df.ts_y).abs() /60/60))
            # df = df[['aid_x','aid_y','wgt']]


def make_covisits_115():
    events =  pl.read_parquet('data/train.parquet')
    events = filter(events, last_21_days_filter)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(aid_diff_filter & within_1_day_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
    df_agg = (
        pl.concat(dfs.values())
        .with_columns(aid_seg=pl.col('aid')//1_00_000)
    )
    return df_agg
# CPU times: user 2min 31s, sys: 1min 29s, total: 4min 1s
# Wall time: 4min 11s

# with open(f'../../data/covisit_matrices/top_40_aids_v{VER}_{PART}.pkl'
y_after_x_filter = (pl.col('ts_y') > pl.col('ts'))

def make_covisits_116(events: pl.DataFrame):
    events =  pl.read_parquet('data/train.parquet')
    events = filter(events, last_28_days_filter & carts_orders_filter)
    # type weight 0 for clicks, otherwise 1
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(y_after_x_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
    df_agg = (
        pl.concat(dfs.values())
        .with_columns(aid_seg=pl.col('aid')//1_00_000)
    )
    return df_agg
#with open(f'../../data/covisit_matrices/top_40_aids_v{VER}_{PART}.pkl', 'wb') as f:
       # pickle.dump(df.to_dict(), f)
    
def make_covisits_155(test_events: pl.DataFrame):
    events = filter(test_events, carts_orders_filter)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .filter(y_after_x_filter)
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff/6)).alias('value').sum()
        )
    )
    
def make_covisits_157(test_events: pl.DataFrame):
    events=test_events
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(df,on="session",suffix="_y")
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff/6)).alias('value').sum()
        )
    )
        # top 80

session_length = pl.col('aid').count().over('session').alias('x')
def make_covisits_165(train_events:pl.DataFrame, test_events: pl.DataFrame):
    train_events = train_events.filter(session_length <=6)
    test_events = test_events.filter(session_length <=3)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(filter(df, carts_orders_filter),on="session",suffix="_y")
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
# top 40

def make_covisits_166(train_events:pl.DataFrame, test_events: pl.DataFrame):
    train_events = train_events.filter(session_length > 6)
    test_events = test_events.filter(session_length > 3)
    dfs = {}
    for gp, df in events.group_by(pl.col('session')//100_000):
        dfs[gp] = (
        df
        .join(filter(df, carts_orders_filter),on="session",suffix="_y")
        .unique(subset = ["session","aid", "aid_y"]) # 1 pair per session (?random time then if duplicate)
        .group_by(["aid", "aid_y"]).agg(
            ((1/2)**(hour_diff)).alias('value').sum()
        )
    )
# top 40


#     "cell_type": "code",
# -   "execution_count": 3,
# +   "execution_count": null,
#     "id": "d487dbd4",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "aid_diff_filter = pl.col(\"aid\")!=pl.col(\"aid_y\")\n",
# -    "cpu_95_tail_30 =  (\n",
# -    "    pl.read_parquet(\"data/train.parquet\")\n",
# -    "    .set_sorted('session')\n",
# -    "    .group_by(\"session\").tail(30)\n",
# -    "    \n",
# -    ")\n",
# -    "dfs = {}\n",
# -    "for gp, df in cpu_95_tail_30.group_by(pl.col('session')//10_000):\n",
# -    "    dfs[gp] = (\n",
# -    "#covisits_95_lzy = (\n",
# -    "    df\n",
# -    "    .join(df,on=\"session\",suffix=\"_y\")\n",
# -    "    .filter(aid_diff_filter)\n",
# -    "    .unique(subset = [\"session\",\"aid\", \"aid_y\"]) # 1 pair per session (?random time then if duplicate)\n",
# -    "    .group_by([\"aid\", \"aid_y\"]).agg(\n",
# -    "        ((1/2)**((pl.col('ts')-pl.col('ts_y'))/pl.duration(hours=1))*pl.when(pl.col('type')==CLICKS).then(1).otherwise(4)).alias('weight').sum()\n",
# -    "    )\n",
# -    ")"
# -   ]
# +   "source": []
#    },
#    {
#     "cell_type": "code",
# @@ -1998,13 +2775,11 @@
#    },
#    {
#     "cell_type": "code",
# -   "execution_count": 4,
# +   "execution_count": null,
#     "id": "267c55ad",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "covisits_95 = pl.concat(dfs.values())"
# -   ]
# +   "source": []
#    },
#    {
#     "cell_type": "code",
# @@ -2259,30 +3034,7 @@
#     "id": "b8ecb411",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "id_diff_filter = pl.col(\"aid\")!=pl.col(\"aid_y\")\n",
# -    "cpu_99_28_days =  (\n",
# -    "    pl.read_parquet(\"data/train.parquet\")\n",
# -    "    .set_sorted('session')\n",
# -    "    .filter(\n",
# -    "        #1662328791\n",
# -    "        (pl.col('ts')> datetime.datetime.fromisoformat('2022-09-04 21:59:51') -pl.duration(days=28)) & \n",
# -    "        pl.col('type').is_in(['carts','orders'])\n",
# :
#     "outputs": [],
# -   "source": [
# -    "covisits_95 = pl.concat(dfs.values())"
# -   ]
# +   "source": []
#    },
#    {
#     "cell_type": "code",
# @@ -2259,30 +3034,7 @@
#     "id": "b8ecb411",
#     "metadata": {},
#     "outputs": [],
# -   "source": [
# -    "id_diff_filter = pl.col(\"aid\")!=pl.col(\"aid_y\")\n",
# -    "cpu_99_28_days =  (\n",
# -    "    pl.read_parquet(\"data/train.parquet\")\n",
# -    "    .set_sorted('session')\n",
# -    "    .filter(\n",
# -    "        #1662328791\n",
# -    "        (pl.col('ts')> datetime.datetime.fromisoformat('2022-09-04 21:59:51') -pl.duration(days=28)) & \n",
# -    "   
