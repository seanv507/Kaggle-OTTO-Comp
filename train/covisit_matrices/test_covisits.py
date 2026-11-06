import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import numpy as np
import pandas as pd

import covisits
import covisits_reference as ref
from covisits import make_covisits_90, make_covisits_95, top_covisit
from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {  # same dtypes as data/train.parquet
    "session": pl.Int32,
    "aid": pl.Int32,
    "ts": pl.Datetime("ms"),
    "type": pl.UInt8,
}


def _events():
    t0 = datetime.datetime(2022, 8, 1)
    day = datetime.timedelta(days=1)
    rows = [
        # session, aid, ts, type
        (0, 1, t0, CARTS),
        (0, 2, t0 + day, ORDERS),
        (0, 2, t0 + 2 * day, CARTS),  # duplicate pair in session -> counted once
        (0, 9, t0, CLICKS),  # clicks ignored
        (1, 1, t0, ORDERS),
        (1, 2, t0 + 20 * day, CARTS),  # > 14 days apart -> ignored
        (2, 1, t0, CARTS),
        (2, 3, t0 + day, CARTS),
        (3, 1, t0, CARTS),
        (3, 2, t0, CARTS),
    ]
    return pl.LazyFrame(rows, schema=EVENT_SCHEMA, orient="row").sort("session")


EXPECTED_90 = pl.DataFrame(
    {
        "aid": [1, 1, 2, 3],
        "aid_y": [2, 3, 1, 1],
        "value": [2, 1, 2, 1],
    },
    schema_overrides={"value": pl.UInt32},
)


def _gpu_available():
    try:
        pl.LazyFrame({"a": [1]}).collect(engine=pl.GPUEngine(raise_on_fail=True))
        return True
    except Exception:
        return False


GPU = pytest.param(
    "gpu",
    marks=pytest.mark.skipif(not _gpu_available(), reason="GPU engine unavailable"),
)


ENGINES = ["streaming", "in-memory", GPU]


def _engine(name):
    # for gpu, fail rather than silently fall back to CPU
    return pl.GPUEngine(raise_on_fail=True) if name == "gpu" else name


@pytest.mark.parametrize("session_chunk", [None, 1, 2, 3])
@pytest.mark.parametrize("engine", ENGINES)
def test_make_covisits_90(session_chunk, engine):
    result = make_covisits_90(
        _events(), session_chunk=session_chunk, engine=_engine(engine)
    )
    assert_frame_equal(
        result.sort("aid", "aid_y"), EXPECTED_90, check_dtypes=False
    )


def _events_95():
    t0 = datetime.datetime(2022, 8, 1)
    hour = datetime.timedelta(hours=1)
    rows = [
        # short session (< 30 events): first events are A=1 click @0h, B=2 cart @1h
        (0, 1, t0, CLICKS),
        (0, 2, t0 + hour, CARTS),
        (0, 1, t0 + 2 * hour, CLICKS),  # repeat of A: only A's first event is used
        # 31 events: aid 7 is outside the last 30, so no (7, 8) pairs
        (1, 7, t0, CLICKS),
        *[(1, 8, t0, CLICKS)] * 30,
    ]
    return pl.LazyFrame(rows, schema=EVENT_SCHEMA, orient="row")


EXPECTED_95 = pl.DataFrame(
    {
        "aid": [1, 2],
        "aid_y": [2, 1],
        # A->B: B is a cart (weight 4), 1h apart: 4 * 0.5 ; B->A: A is a click: 1 * 0.5
        "value": [2.0, 0.5],
    }
)


@pytest.mark.parametrize("session_chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
def test_make_covisits_95_hand_checked(session_chunk, engine):
    result = make_covisits_95(
        _events_95(), session_chunk=session_chunk, engine=_engine(engine)
    )
    assert_frame_equal(result.sort("aid", "aid_y"), EXPECTED_95, check_dtypes=False)


def _random_events(n_sessions=300, seed=0):
    """Sessions of 1-60 events over few aids (so aids repeat), time-ordered."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 61, n_sessions)
    n = lengths.sum()
    gaps = rng.integers(0, 3 * 3600_000, n)  # up to 3 hours, in ms
    return pl.DataFrame(
        {
            "session": np.repeat(np.arange(n_sessions), lengths),
            "aid": rng.integers(0, 25, n),
            "ts": np.datetime64("2022-08-01", "ms") + np.cumsum(gaps).astype("timedelta64[ms]"),
            "type": rng.choice([CLICKS, CARTS, ORDERS], n, p=[0.8, 0.15, 0.05]),
        },
        schema=EVENT_SCHEMA,
    )


def _reference_95(events: pl.DataFrame) -> pl.DataFrame:
    """Direct pandas port of gen_pairs + weights from train/covisit_matrices/cpu-95.ipynb."""
    df = events.to_pandas()
    df = df.groupby("session", sort=False).tail(30)
    df = pd.merge(df, df, on="session")
    pairs = df.query("aid_x != aid_y")[
        ["session", "aid_x", "aid_y", "ts_x", "ts_y", "type_y"]
    ].drop_duplicates(["session", "aid_x", "aid_y"])
    hours = (pairs.ts_y - pairs.ts_x).abs().dt.total_seconds() / 3600
    w = np.where(pairs.type_y.isin([CARTS, ORDERS]), 4, 1) * 0.5**hours
    out = (
        pairs.assign(value=w)
        .groupby(["aid_x", "aid_y"], as_index=False)["value"]
        .sum()
        .rename(columns={"aid_x": "aid"})
    )
    return pl.from_pandas(out)


@pytest.mark.parametrize("session_chunk", [None, 50])
@pytest.mark.parametrize("engine", ENGINES)
def test_make_covisits_95_matches_reference(session_chunk, engine):
    events = _random_events()
    expected = _reference_95(events).sort("aid", "aid_y")
    result = make_covisits_95(
        events.lazy(), session_chunk=session_chunk, engine=_engine(engine), top=0
    ).sort("aid", "aid_y")
    assert_frame_equal(result, expected, check_dtypes=False, rel_tol=1e-9)


def test_top_covisit_keeps_top_k_per_aid():
    df = pl.DataFrame(
        {
            "aid": [1, 1, 1, 2, 2, 3],
            "aid_y": [10, 11, 12, 20, 21, 30],
            "value": [1.0, 3.0, 2.0, 5.0, 4.0, 1.0],
        }
    )
    expected = pl.DataFrame(
        {
            "aid": [1, 1, 2, 2, 3],
            "aid_y": [11, 12, 20, 21, 30],
            "value": [3.0, 2.0, 5.0, 4.0, 1.0],
        }
    )
    result = top_covisit(df, top=2)
    assert_frame_equal(
        result.sort("aid", "value", descending=[False, True]),
        expected,
    )


# ---------------------------------------------------------------------------
# Matrices from the GPU notebooks and cpu-99: hand-checked cases
# ---------------------------------------------------------------------------

C, A, O = CLICKS, CARTS, ORDERS
T0 = datetime.datetime(2022, 8, 30)  # inside the 21/28-day windows; 02:00 UTC+2
OLD = datetime.datetime(2022, 8, 1)  # before them


def h(hours):
    return T0 + datetime.timedelta(hours=hours)


def _frame(rows):
    """rows: (session, aid, ts, type), already sorted by session then ts."""
    return pl.LazyFrame(rows, schema=EVENT_SCHEMA, orient="row")


def _f93(seconds):
    """93's time weight for an x event `seconds` after start_train."""
    return 1 + 3 * seconds / (covisits.END_TEST_S - covisits.START_TRAIN_S)


S = covisits.start_train
HOUR = datetime.timedelta(hours=1)

# name -> (make function, list of event-row lists (1 frame, or train + test), extra kwargs,
#          expected {(aid, aid_y): value})
HAND_CASES = {
    "93": (
        covisits.make_covisits_93,
        [[(0, 1, S, C), (0, 2, S + HOUR, C), (0, 1, S + 2 * HOUR, C), (0, 3, S + 30 * HOUR, C)]],
        {},
        # (1,2): latest copy of 1 (at 2h) with 2, 1h apart; aid 3 is > 24h from both
        {(1, 2): _f93(7200) * 0.5, (2, 1): _f93(3600) * 0.5},
    ),
    "99": (
        covisits.make_covisits_99,
        [[(0, 4, OLD, A), (0, 1, h(0), A), (0, 2, h(1), O), (0, 3, h(2), C), (0, 1, h(3), A)]],
        {},
        # earliest copies: 1@0h, 2@1h; 4 too old, 3 a click
        {(1, 2): 0.5, (2, 1): 0.5},
    ),
    "115": (
        covisits.make_covisits_115,
        [[(0, 5, OLD, C), (0, 1, h(0), C), (0, 2, h(1), A), (0, 1, h(2), C)]],
        {},
        # latest copies 1@2h, 2@1h; self pairs kept; 5 too old
        {(1, 1): 1.0, (1, 2): 0.5, (2, 1): 0.5, (2, 2): 1.0},
    ),
    "116": (
        covisits.make_covisits_116,
        [[(0, 1, h(0), C), (0, 2, h(1), A), (0, 1, h(2), C), (0, 2, h(4), O)]],
        {},
        # y after x; latest x copy: (1@2h, 2@4h order) 2h; click y weight 0
        {(1, 1): 0.0, (1, 2): 0.25, (2, 1): 0.0, (2, 2): 0.125},
    ),
    "155": (
        covisits.make_covisits_155,
        [[(0, 1, h(0), C), (0, 2, h(6), A), (0, 1, h(12), C), (0, 2, h(18), O)]],
        {},
        # y a later cart/order, latest x copy, 6h half-life: (1@12, 2@18), (2@6, 2@18)
        {(1, 2): 0.5, (2, 2): 0.25},
    ),
    "157": (
        covisits.make_covisits_157,
        [[(0, 1, h(0), C), (0, 2, h(6), A), (0, 1, h(12), C)]],
        {},
        # latest copies 1@12h, 2@6h, 6h half-life, self pairs kept
        {(1, 1): 1.0, (1, 2): 0.5, (2, 1): 0.5, (2, 2): 1.0},
    ),
    "165": (
        covisits.make_covisits_165,
        [
            [(0, 1, h(0), C), (0, 2, h(1), A)] + [(1, 8, h(i), A) for i in range(7)],
            [(2, 3, h(0), C), (2, 4, h(1), A), (2, 3, h(2), C), (2, 4, h(3), A), (3, 5, h(0), C), (3, 6, h(2), O)],
        ],
        {},
        # train session 0 (2 events <= 6) and test session 3 (2 <= 3); 1 (7) and 2 (4) too long
        {(1, 2): 0.5, (2, 2): 1.0, (5, 6): 0.25, (6, 6): 1.0},
    ),
    "166": (
        covisits.make_covisits_166,
        [
            [(0, 1, h(0), C), (0, 2, h(1), A)] + [(1, 8, h(i), A) for i in range(7)],
            [(2, 3, h(0), C), (2, 4, h(1), A), (2, 3, h(2), C), (2, 4, h(3), A), (3, 5, h(0), C), (3, 6, h(2), O)],
        ],
        {},
        # train session 1 (7 > 6) and test session 2 (4 > 3), latest copies
        {(8, 8): 1.0, (3, 4): 0.5, (4, 4): 1.0},
    ),
    "167": (
        covisits.make_covisits_167,
        [[(0, 1, h(0), C), (0, 2, h(1), A), (0, 3, h(12), C), (0, 4, h(13), O)]],
        {},
        # x before 14:00 UTC+2 (h(0) is 02:00, h(12) is 14:00); y carts/orders
        {(1, 2): 0.5, (1, 4): 0.5**13, (2, 2): 1.0, (2, 4): 0.5**12},
    ),
    "168": (
        covisits.make_covisits_168,
        [[(0, 1, h(0), C), (0, 2, h(1), A), (0, 3, h(12), C), (0, 4, h(13), O)]],
        {},
        {(3, 2): 0.5**11, (3, 4): 0.5, (4, 2): 0.5**12, (4, 4): 1.0},
    ),
    "217": (
        covisits.make_covisits_217,
        [[(0, 1, h(0), C), (0, 2, h(1), C), (0, 1, h(2), C), (0, 3, h(3), C), (1, 1, h(0), C), (1, 2, h(1), C)]],
        {},
        # y is aid_y's first event: 3@3 is next to 1@2, but 1's first event is @0 -> no (3, 1)
        {(1, 2): 2, (2, 1): 2, (1, 3): 1},
    ),
    "220": (
        covisits.make_covisits_220,
        [[(0, 1, h(0), C), (0, 2, h(1), C), (0, 1, h(2), C), (0, 3, h(3), C), (1, 1, h(0), C), (1, 2, h(1), C)]],
        {},
        {(1, 2): 2, (2, 1): 2, (1, 3): 1, (2, 3): 1, (3, 2): 1},
    ),
    "226": (
        covisits.make_covisits_226,
        [[(0, 1, h(0), A), (0, 9, h(1), C), (0, 2, h(2), A), (0, 3, h(3), O), (0, 1, h(4), A), (0, 4, h(5), O), (0, 5, h(6), A)]],
        {},
        # carts/orders only, positions 1,2,3,1,4,5; aid_y's first one 1-3 positions later
        {p: 1 for p in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]},
    ),
    "232": (
        covisits.make_covisits_232,
        [[(0, 1, h(0), C)] + [(0, 2, h(i), C) for i in range(1, 6)] + [(0, 3, h(6), C)]],
        {},
        # (2, 3): x copies of 2 within 3 of 3@6 are @3,@4,@5; the earliest (@3) is kept: 3h
        {(1, 2): 0.5, (2, 1): 0.5, (2, 3): 0.5**3},
    ),
    "235": (
        covisits.make_covisits_235,
        [[(0, 1, h(0), C)] + [(0, 2, h(i), C) for i in range(1, 6)] + [(0, 3, h(6), C)]],
        {},
        {(1, 2): 0.5, (1, 3): 0.5**6, (2, 1): 0.5, (2, 3): 0.5**5, (3, 1): 0.5**6, (3, 2): 0.5**5},
    ),
    "239": (
        covisits.make_covisits_239,
        [
            [(0, 1, h(0), C), (0, 2, h(1), A), (0, 3, h(2), O), (1, 1, h(0), C), (1, 2, h(1.5), A)],
            [(2, 1, h(0), C), (2, 2, h(1), A)],
        ],
        {"train_parts": 1, "test_parts": 1},
        # train sessions 0, 1 are one file: (1, 2) keeps the closest (1h, not 1.5h); + test file 1h.
        # (1, 3) is 2h apart: not < 2h
        {(1, 2): 1.0, (2, 3): 0.5},
    ),
    "700": (
        covisits.make_covisits_700,
        [
            [(0, 1, h(0), C), (0, 2, h(1), A), (0, 2, h(2), O), (0, 2, h(3), A), (0, 1, h(4), O), (0, 3, h(5), C)]
            + [(1, 1, OLD, C), (1, 2, h(1), A)]
            + [(2, 1, h(0), C), (2, 2, h(1), A)]
        ],
        {},
        # session 0: 2 carted and ordered -> 2; session 1 started too early; session 2 -> 1
        {(1, 2): 3},
    ),
    "701": (
        covisits.make_covisits_701,
        [[(0, 1, h(0), C), (0, 2, h(1), C), (0, 2, h(2), A), (0, 2, h(3), C), (0, 1, h(4), C), (1, 1, OLD, C), (1, 2, h(1), A)]],
        {},
        # earliest click of 2 (1h) + earliest cart of 2 (2h); session 1 started too early
        {(1, 2): 0.5 + 0.25},
    ),
}


def _expected(pairs):
    return pl.DataFrame(
        [(a, b, float(v)) for (a, b), v in sorted(pairs.items())],
        schema={"aid": pl.Int32, "aid_y": pl.Int32, "value": pl.Float64},
        orient="row",
    )


@pytest.mark.parametrize("session_chunk", [None, 1])
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", HAND_CASES)
def test_hand_checked(name, engine, session_chunk):
    make, frames, kwargs, expected = HAND_CASES[name]
    result = make(
        *[_frame(rows) for rows in frames],
        session_chunk=session_chunk, engine=_engine(engine), top=0, **kwargs,
    )
    assert_frame_equal(
        result.sort("aid", "aid_y"), _expected(expected), check_dtypes=False, rel_tol=1e-9
    )


# ---------------------------------------------------------------------------
# Same matrices against pandas ports of the notebooks, on random data
# ---------------------------------------------------------------------------


def _random_sessions(n_sessions, first_session, seed):
    """Sessions starting anywhere from start_train to end_test - 1 day (so the 21/28-day
    windows cut some), 1-8 or 1-60 events over few aids, strictly increasing whole-second
    timestamps (so the reference's sorts have no ties)."""
    rng = np.random.default_rng(seed)
    lengths = np.where(rng.random(n_sessions) < 0.5, rng.integers(1, 9, n_sessions), rng.integers(1, 61, n_sessions))
    span = int((covisits.end_test - covisits.start_train).total_seconds()) - 86400
    starts = covisits.START_TRAIN_S + rng.integers(0, span, n_sessions)
    gaps = rng.integers(1, 3 * 3600, lengths.sum())  # seconds
    session = np.repeat(np.arange(n_sessions), lengths)
    first = np.r_[0, np.cumsum(lengths)[:-1]]
    gaps[first] = 0
    within = np.cumsum(gaps) - np.repeat(np.cumsum(gaps)[first], lengths)
    ts_s = np.repeat(starts, lengths) + within
    n = lengths.sum()
    return pl.DataFrame(
        {
            "session": session + first_session,
            "aid": rng.integers(0, 25, n),
            "ts": (ts_s * 1000).astype("datetime64[ms]"),
            "type": rng.choice([C, A, O], n, p=[0.7, 0.2, 0.1]),
        },
        schema=EVENT_SCHEMA,
    )


TRAIN = _random_sessions(300, 0, seed=1)
TEST = _random_sessions(100, 300, seed=2)
ALL = pl.concat([TRAIN, TEST])

# name -> (make function, reference port, takes train + test separately)
REFERENCE_CASES = {
    "93": (covisits.make_covisits_93, ref.ref_93, False),
    "99": (covisits.make_covisits_99, ref.ref_99, False),
    "115": (covisits.make_covisits_115, ref.ref_115, False),
    "116": (covisits.make_covisits_116, ref.ref_116, False),
    "155": (covisits.make_covisits_155, ref.ref_155, "test"),
    "157": (covisits.make_covisits_157, ref.ref_157, "test"),
    "165": (covisits.make_covisits_165, ref.ref_165, True),
    "166": (covisits.make_covisits_166, ref.ref_166, True),
    "167": (covisits.make_covisits_167, ref.ref_167, False),
    "168": (covisits.make_covisits_168, ref.ref_168, False),
    "217": (covisits.make_covisits_217, ref.ref_217, False),
    "220": (covisits.make_covisits_220, ref.ref_220, False),
    "226": (covisits.make_covisits_226, ref.ref_226, False),
    "232": (covisits.make_covisits_232, ref.ref_232, False),
    "235": (covisits.make_covisits_235, ref.ref_235, False),
    "239": (covisits.make_covisits_239, ref.ref_239, True),
    "700": (covisits.make_covisits_700, ref.ref_700, False),
    "701": (covisits.make_covisits_701, ref.ref_701, False),
}


@pytest.mark.parametrize("session_chunk", [None, 50])
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", REFERENCE_CASES)
def test_matches_reference(name, engine, session_chunk):
    make, port, split = REFERENCE_CASES[name]
    if split is True:
        frames = [TRAIN, TEST]
    elif split == "test":
        frames = [TEST]
    else:
        frames = [ALL]
    expected = port(*[ref.to_reference_frame(f) for f in frames]).sort("aid", "aid_y")
    assert expected.height > 0
    result = make(
        *[f.lazy() for f in frames], session_chunk=session_chunk, engine=_engine(engine), top=0
    ).sort("aid", "aid_y")
    assert_frame_equal(result, expected, check_dtypes=False, rel_tol=1e-9)
