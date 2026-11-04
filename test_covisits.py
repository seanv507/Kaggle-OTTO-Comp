import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import numpy as np
import pandas as pd

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
