import datetime

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import features
import features_reference as ref
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


START = datetime.datetime(2022, 7, 31, 22)  # valid_train starts here; valid_test 21 days later


def _random_events(n_sessions, days, start, seed, first_session=0, n_aids=40):
    """Sessions sorted by session, ts, with repeated aids, ts ties and long sessions."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(first_session, first_session + n_sessions):
        n = int(rng.integers(1, 30))
        t0 = rng.integers(0, days * 86400)
        # a mix of short gaps (ties included) and multi-hour/day gaps
        gaps = rng.choice([0, 30, 600, 3 * 3600, 2 * 86400], size=n, p=[0.1, 0.4, 0.3, 0.15, 0.05])
        ts = np.minimum(t0 + np.cumsum(gaps) - gaps[0], days * 86400 - 1)
        aids = rng.integers(0, n_aids, size=n)
        types = rng.choice([CLICKS, CARTS, ORDERS], size=n, p=[0.7, 0.2, 0.1])
        for a, t, ty in zip(aids, ts, types, strict=True):
            rows.append((s, int(a), start + datetime.timedelta(seconds=int(t)), int(ty)))
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row").sort("session", "ts", maintain_order=True)


TEST_START = START + datetime.timedelta(days=21)
DATA = {
    # dense: every aid in every window and day
    "dense": (_random_events(300, 21, START, seed=1),
              _random_events(200, 7, TEST_START, seed=2, first_session=1000)),
    # sparse: most aids seen a few times, so the item12 window misses aids (nulls), item13
    # days are missing (-1) and single-row stds / unbought aids (-1) occur
    "sparse": (_random_events(300, 21, START, seed=3, n_aids=2000),
               _random_events(200, 7, TEST_START, seed=4, first_session=1000, n_aids=2000)),
}


def _reference(name, events):
    fn, _ = ref.FEATURES[name]
    return pl.from_pandas(fn(ref.to_reference_frame(events)), nan_to_null=True)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("data", list(DATA))
@pytest.mark.parametrize("name", list(features.FEATURES))
def test_matches_reference(name, data, engine):
    fn, which = features.FEATURES[name]
    train, test = DATA[data]
    events = train if which == "train" else test
    got = fn(events).collect(engine=_engine(engine))
    key = got.columns[0]
    assert key in ("aid", "session")
    assert got.schema[key] == pl.Int32
    assert all(d in (pl.Float32, pl.Int32) for d in got.schema.values())
    want = _reference(name, events)
    assert want.columns == got.columns
    want = want.cast(got.schema)  # cudf keeps int32 under nulls (item12); pandas makes float64
    assert_frame_equal(got.sort(key), want.sort(key), rel_tol=1e-5, abs_tol=1e-4)


def test_hand_checked():
    """item4 / item10 / user7 on four events, checked by hand."""
    t0 = datetime.datetime(2022, 8, 22, 1)  # epoch s 1661130000: hour 3600, day 349200
    events = pl.DataFrame(
        [
            (0, 5, t0, CLICKS),
            (0, 7, t0 + datetime.timedelta(seconds=60), CARTS),
            (0, 5, t0 + datetime.timedelta(seconds=120), ORDERS),
            (1, 5, t0 + datetime.timedelta(seconds=600), ORDERS),
        ],
        schema=EVENT_SCHEMA, orient="row",
    )
    item4 = features.item_4(events).collect().sort("aid")
    assert item4.row(0, named=True) == {
        "aid": 5, "buy_ratio": pytest.approx(4 / 3), "count_item": 3, "count_user": 2,
        "hour_mean": pytest.approx(3600 + 720 / 3), "day_mean": pytest.approx(349200 + 720 / 3),
        "repeat": 1.5, "buy_ratio_std": pytest.approx(np.std([0, 2, 2], ddof=1)),
        "hour_std": pytest.approx(np.std([0, 120, 600], ddof=1)),
        "day_std": pytest.approx(np.std([0, 120, 600], ddof=1)),
    }
    assert item4.filter(aid=7)["buy_ratio_std"].item() == -1  # one row: std -> -1

    item10 = features.item_10(events).collect().sort("aid").select("aid", "prev3", "next3", "orders3",
                                                                    "carts3", "order_repeat3", "cart_repeat3")
    # aid 5: prev diffs {5-7=-2} (row 2), next diffs {5-7=-2} (row 0) -> 1 each
    # aid 7: prev {7-5=2}, next {7-5=2}
    assert item10.rows() == [(5, 1, 1, 2, 0, 1.0, -1.0), (7, 1, 1, 0, 1, -1.0, 1.0)]

    user7 = features.user_7(events).collect().sort("session")
    assert user7.rows() == [(0, 120, 0, 120), (1, 600, 600, 0)]
