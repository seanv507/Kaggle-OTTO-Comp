import datetime

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import candidates
import candidates_reference as ref
from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {  # same dtypes as data/*.parquet
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}


def _gpu_available():
    try:
        pl.LazyFrame({"a": [1]}).collect(engine=pl.GPUEngine(raise_on_fail=True))
        return True
    except Exception:
        return False


GPU = pytest.param(
    "gpu", marks=pytest.mark.skipif(not _gpu_available(), reason="GPU engine unavailable")
)
ENGINES = ["streaming", "in-memory", GPU]


def _engine(name):
    # for gpu, fail rather than silently fall back to CPU
    return pl.GPUEngine(raise_on_fail=True) if name == "gpu" else name


T0 = datetime.datetime(2022, 8, 22)
M = datetime.timedelta(minutes=1)
H = datetime.timedelta(hours=1)


def _hand_events():
    rows = [
        # session 0: sub-sessions d=0 (rows 0-2), d=1 (rows 3-6, a >2h gap before row 3)
        (0, 10, T0, CLICKS),
        (0, 11, T0 + M, CARTS),
        (0, 10, T0 + 2 * M, CLICKS),
        (0, 12, T0 + 3 * H, CLICKS),  # gap 2h58m > 2h -> d=1
        (0, 13, T0 + 3 * H + M, ORDERS),
        (0, 14, T0 + 3 * H + 40 * M, CLICKS),  # tie at the last ts of d=1 ...
        (0, 15, T0 + 3 * H + 40 * M, CLICKS),  # ... b goes to the earlier row (aid 14)
        # session 1: exactly 2h gap stays in the same sub-session
        (1, 20, T0, CLICKS),
        (1, 21, T0 + 2 * H, CLICKS),
    ]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


@pytest.mark.parametrize("engine", ENGINES)
def test_add_subsessions_hand_checked(engine):
    result = candidates.add_subsessions(_hand_events()).collect(engine=_engine(engine))
    expected = _hand_events().with_columns(
        pos=pl.Series([0, 1, 2, 3, 4, 5, 6, 0, 1], dtype=pl.UInt32),
        d=pl.Series([0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=pl.UInt16),
        a=pl.Series([1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=pl.UInt8),
        b=pl.Series([0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=pl.UInt8),
    )
    assert_frame_equal(result, expected)


HAND_SETS = {
    # name -> {session: aids in reference list order}
    "all": {0: [15, 14, 13, 12, 10, 11], 1: [21, 20]},
    "first_of_d": {0: [12, 10], 1: [20]},
    "last_of_d": {0: [14, 10], 1: [21]},
    "last_of_d_oldest": {0: [10, 14], 1: [21]},
    "last_d": {0: [15, 14, 13, 12], 1: [21, 20]},
    "last_30min": {0: [15, 14], 1: [21]},  # 13 is 39 minutes before the last event
    "buys": {0: [13, 11]},
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", HAND_SETS)
def test_source_set_hand_checked(name, engine):
    events = candidates.add_subsessions(_hand_events())
    result = candidates.source_set(events, name).collect(engine=_engine(engine))
    expected = pl.DataFrame(
        [(s, aid, i) for s, aids in HAND_SETS[name].items() for i, aid in enumerate(aids)],
        schema={"session": pl.Int32, "aid": pl.Int32, "src_idx": pl.UInt32},
        orient="row",
    )
    assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Against pandas ports of the notebooks, on random data
# ---------------------------------------------------------------------------


def _random_sessions(n_sessions, seed):
    """Sessions of 1-40 events over few aids, whole-second timestamps with many ties
    (gap 0) and some gaps over 2 hours, so sub-sessions and tie-breaks get exercised."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 41, n_sessions)
    n = lengths.sum()
    gaps = rng.choice([0, 1, 60, 1800, 7200, 7201, 30000], n, p=[0.3, 0.2, 0.2, 0.1, 0.05, 0.05, 0.1])
    first = np.r_[0, np.cumsum(lengths)[:-1]]
    gaps[first] = 0
    within = np.cumsum(gaps) - np.repeat(np.cumsum(gaps)[first], lengths)
    starts = 1661119200 + rng.integers(0, 6 * 86400, n_sessions)
    ts_s = np.repeat(starts, lengths) + within
    return pl.DataFrame(
        {
            "session": np.repeat(np.arange(n_sessions), lengths),
            "aid": rng.integers(0, 15, n),
            "ts": (ts_s * 1000).astype("datetime64[ms]"),
            "type": rng.choice([CLICKS, CARTS, ORDERS], n, p=[0.7, 0.2, 0.1]),
        },
        schema=EVENT_SCHEMA,
    )


RANDOM = _random_sessions(300, seed=0)


def _reference_events():
    return ref.add_a_b(ref.add_d(ref.to_reference_frame(RANDOM)))


@pytest.mark.parametrize("engine", ENGINES)
def test_add_subsessions_matches_reference(engine):
    expected = pl.from_pandas(_reference_events()[["d", "a", "b"]])
    result = candidates.add_subsessions(RANDOM).collect(engine=_engine(engine))
    assert_frame_equal(result.select("d", "a", "b"), expected, check_dtypes=False)


def _sets_frame(per_session: dict[int, dict[str, list[int]]], name: str) -> pl.DataFrame:
    return pl.DataFrame(
        [(s, aid, i) for s, sets in per_session.items() for i, aid in enumerate(sets[name])],
        schema={"session": pl.Int32, "aid": pl.Int32, "src_idx": pl.UInt32},
        orient="row",
    )


def _reference_sets():
    df = _reference_events()
    out = {}
    for session, g in df.groupby("session"):
        sets = ref.source_sets_orders(g)
        sets |= ref.source_sets_aids(g)
        out[session] = sets
    for row in ref.valid_lists(df):
        sets620 = ref.source_sets_620(row)
        for name, aids in sets620.items():
            # model-620 builds the same sets from the pickled lists
            assert aids == out[row[0]][name], (row[0], name)
        u, u2, u3, u4, u5, u6, buys = ref._sets_709(row)
        # ... and so does model-709, which adds last_day (unique_aids6)
        assert (u, u2, u3, u4, u5, buys) == tuple(
            out[row[0]][n] for n in ["all", "first_of_d", "last_of_d", "last_d", "last_30min", "buys"]
        ), row[0]
        out[row[0]]["last_day"] = u6
    return out


REFERENCE_SETS = _reference_sets()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", candidates.SOURCE_SETS)
def test_source_set_matches_reference(name, engine):
    events = candidates.add_subsessions(RANDOM)
    result = candidates.source_set(events, name).collect(engine=_engine(engine))
    assert_frame_equal(result, _sets_frame(REFERENCE_SETS, name))


# ---------------------------------------------------------------------------
# model-564 suggest_aids / suggest_orders against the reference
# ---------------------------------------------------------------------------


def _random_matrices(names, seed, n_aids=45, max_len=60):
    """Covisitation matrices as save_covisits.py writes them (aid, aid_y, value sorted by
    aid, value descending, aid_y), with lists of 0 to max_len neighbours (so short
    lists shift the reference's i) and some aids missing."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in names:
        rows = []
        for aid in range(n_aids):
            if rng.random() < 0.15:
                continue  # aid not in the matrix
            length = int(rng.integers(0, max_len + 1))
            ys = rng.choice(np.arange(n_aids + 15), size=min(length, n_aids + 15), replace=False)
            rows += [(aid, int(y), float(length - j)) for j, y in enumerate(ys)]
        out[name] = pl.DataFrame(
            rows, schema={"aid": pl.Int32, "aid_y": pl.Int32, "value": pl.Float64}, orient="row"
        )
    return out


def _as_dicts(matrices):
    return {
        name: {aid: g["aid_y"].to_list() for (aid,), g in m.group_by("aid", maintain_order=True)}
        for name, m in matrices.items()
    }


def _long_sessions(n_sessions, seed):
    """Like _random_sessions but up to 60 events over 45 aids, so some sessions have the
    20+ unique aids of the `auto` history branch."""
    rng = np.random.default_rng(seed)
    base = _random_sessions(n_sessions, seed)
    return base.with_columns(aid=pl.Series(rng.integers(0, 45, base.height), dtype=pl.Int32))


MATRICES = _random_matrices(
    [*candidates.ALL_564, "top20new", "top20new2", "day", "day2", "less", "more", "less2", "more2"], seed=3
)
MATRIX_DICTS = _as_dicts(MATRICES)
EVENTS_564 = pl.concat([
    _long_sessions(150, seed=4),
    _random_sessions(150, seed=5).with_columns(pl.col("session") + 1000),
    # long sessions: 60 events over 45 aids
    pl.DataFrame({
        "session": np.repeat(np.arange(2000, 2040), 60),
        "aid": np.random.default_rng(6).integers(0, 45, 2400),
        "ts": (np.repeat(1661119200 + np.arange(40) * 3600, 60) * 1000 + np.tile(np.arange(60), 40) * 1000)
        .astype("datetime64[ms]"),
        "type": np.random.default_rng(7).choice([CLICKS, CARTS, ORDERS], 2400, p=[0.7, 0.2, 0.1]),
    }, schema=EVENT_SCHEMA),
])
POPULAR = [3, 7] + list(range(500, 598))  # 100 like the notebooks; 3 and 7 can also be candidates


REF_INFER = {
    "564": ref.ALL_564,
    "612": [*ref.ALL_564, "top20new", "top20new2"],
    "614": ["top20new", "top20new2"],
}


def _reference_620(item_ct):
    rows = ref.valid_lists(ref.add_a_b(ref.add_d(ref.to_reference_frame(EVENTS_564))))
    out = []
    for row in rows:
        session, aids = ref.suggest_aids_620(row, MATRIX_DICTS, ITEM_CT2=item_ct, top_clicks=POPULAR)
        out += [(session, pos, a) for pos, a in enumerate(aids)]
    return pl.DataFrame(out, schema={"session": pl.Int32, "pos": pl.UInt32, "aid": pl.Int32}, orient="row")


def _clicks_620():
    """model-620 output (already checked against the reference) as 709's click_aids."""
    return candidates.suggest_564(
        EVENTS_564, MATRICES, "clicks", item_ct=50, popular=POPULAR, family="620"
    ).collect()


def _reference_709(target, item_ct):
    rows = ref.valid_lists(ref.add_a_b(ref.add_d(ref.to_reference_frame(EVENTS_564))))
    click_df = {s: g["aid"].to_list() for (s,), g in _clicks_620().group_by("session", maintain_order=True)}
    fn, pad_kw = (
        (ref.suggest_aids_709, "top_clicks") if target == "clicks" else (ref.suggest_orders_709, "top_orders")
    )
    out = []
    for row in rows:
        aids, scores = fn(row, MATRIX_DICTS, click_df, ITEM_CT=item_ct, **{pad_kw: POPULAR})
        out += [(row[0], pos, a, float(sc)) for pos, (a, sc) in enumerate(zip(aids, scores, strict=True))]
    return pl.DataFrame(
        out, schema={"session": pl.Int32, "pos": pl.UInt32, "aid": pl.Int32, "score": pl.Float64}, orient="row"
    )


def _reference_564(family, target, include, branch, item_ct):
    if family == "620":
        return _reference_620(item_ct)
    if family == "709":
        return _reference_709(target, item_ct)
    df = ref.add_d(ref.to_reference_frame(EVENTS_564))
    fn = ref.suggest_aids_564 if target == "clicks" else ref.suggest_orders_564
    pad_kw = "top_clicks" if target == "clicks" else "top_orders"
    rows = []
    for session, g in df.groupby("session"):
        aids, scores = fn(
            g, MATRIX_DICTS, INFER=REF_INFER[family] if include is None else include, branch=branch,
            ITEM_CT=item_ct, **{pad_kw: POPULAR},
        )
        rows += [(session, pos, a, float(s)) for pos, (a, s) in enumerate(zip(aids, scores, strict=True))]
    return pl.DataFrame(
        rows, schema={"session": pl.Int32, "pos": pl.UInt32, "aid": pl.Int32, "score": pl.Float64}, orient="row"
    )


# The sequential sum in total_scores (cum_sum per group, needed to break ties exactly as
# the reference's Counter does) is not supported by cudf-polars.
GPU_NO_GROUP_CUM_SUM = pytest.param(
    "gpu",
    marks=[
        pytest.mark.skipif(not _gpu_available(), reason="GPU engine unavailable"),
        pytest.mark.xfail(reason="cudf-polars: cum_sum in group_by", raises=pl.exceptions.ComputeError, strict=True),
    ],
)


@pytest.mark.parametrize("item_ct", [50, 12])
@pytest.mark.parametrize("engine", ["streaming", "in-memory", GPU_NO_GROUP_CUM_SUM])
@pytest.mark.parametrize("run", candidates.RUNS)
def test_suggest_matches_reference(run, engine, item_ct):
    family, target, include, branch = candidates.RUNS[run]
    expected = _reference_564(family, target, include, branch, item_ct)
    result = candidates.suggest_564(
        EVENTS_564, MATRICES, target, include=include, branch=branch, item_ct=item_ct, popular=POPULAR,
        family=family, clicks620=_clicks_620() if family == "709" else None,
    ).collect(engine=_engine(engine))
    if family == "620":  # model-620 wrote aids only
        result = result.drop("score")
    else:
        # rules produced some candidates (not only history aids and padding)
        assert expected.filter(pl.col("score") >= 0).height > 0
    assert_frame_equal(result, expected, check_dtypes=False)
