import datetime

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import merge
import merge_reference as ref
from event_types import CARTS, CLICKS, ORDERS

EVENT_SCHEMA = {"session": pl.Int32, "aid": pl.Int32, "ts": pl.Datetime("ms"), "type": pl.UInt8}
N_AIDS = 120
START = datetime.datetime(2022, 8, 22)


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
    return pl.GPUEngine(raise_on_fail=True) if name == "gpu" else name


def _events(rng, n_sessions):
    rows = []
    for s in range(n_sessions):
        n = int(rng.integers(1, 15))
        t = np.cumsum(rng.choice([0, 60, 3600, 2 * 86400], size=n, p=[0.2, 0.5, 0.2, 0.1]))  # ties, >1 day gaps
        aids = rng.integers(0, N_AIDS, size=n)
        types = rng.choice([CLICKS, CARTS, ORDERS], size=n, p=[0.6, 0.25, 0.15])  # some sessions without buys
        rows += [(s, int(a), START + datetime.timedelta(seconds=int(x)), int(ty))
                 for a, x, ty in zip(aids, t, types, strict=True)]
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def _run(rng, sessions, length):
    """A candidate file: unique aids, a few history (-1) scores, padding (-2) that may
    repeat a real candidate, as save_candidates.py writes."""
    rows = []
    for s in sessions:
        n_real = int(rng.integers(length // 3, length + 1))
        aids = rng.choice(N_AIDS, size=n_real, replace=False)
        scores = np.where(rng.random(n_real) < 0.2, -1.0, rng.random(n_real).round(3) * 10)
        pad = rng.integers(0, N_AIDS, size=length - n_real)
        all_aids = np.concatenate([aids, pad])
        all_scores = np.concatenate([scores, np.full(len(pad), -2.0)])
        rows += [(s, p, int(a), float(w)) for p, (a, w) in enumerate(zip(all_aids, all_scores, strict=True))]
    return pl.DataFrame(rows, schema={"session": pl.Int32, "pos": pl.Int32, "aid": pl.Int32, "score": pl.Float64},
                        orient="row")


def _labels(rng, sessions):
    rows = []
    for s in sessions:
        for ty in [CLICKS, CARTS, ORDERS]:
            if rng.random() < 0.6:
                rows.append((s, ty, [int(a) for a in rng.choice(N_AIDS, size=int(rng.integers(1, 6)), replace=False)]))
    return pl.DataFrame(rows, schema={"session": pl.Int32, "type": pl.UInt8, "ground_truth": pl.List(pl.Int32)},
                        orient="row")


def _feature(rng, key, keys, name, cover=0.8):
    """Two columns (int32, float32 with nulls) for most keys. Names end in 12, as item41's
    do, so the 12 -> 13 rename of step F applies to every column (no name clashes); it
    is exercised on all files alike."""
    keys = np.sort(rng.choice(keys, size=int(len(keys) * cover), replace=False))
    f = rng.random(len(keys)).astype(np.float32)
    f[rng.random(len(keys)) < 0.1] = np.nan
    return pl.DataFrame({
        key: keys.astype(np.int32),
        f"{name}_i12": rng.integers(0, 100, len(keys)).astype(np.int32),
        f"{name}_f12": f,
    }).with_columns(pl.col(f"{name}_f12").fill_nan(None))


def _data(seed=0, n_sessions=60):
    rng = np.random.default_rng(seed)
    events = _events(rng, n_sessions)
    sessions = list(range(n_sessions))
    runs = {r for m in merge.TARGET for r in merge.runs_needed(m)}
    cands = {r: _run(rng, sessions, 20 if r.startswith("709") else 10) for r in sorted(runs)}
    feats = {}
    for name in ["item4", "item10", "item12", "item13", "item20", "item21", "item22", "item40", "item41"]:
        feats[name] = _feature(rng, "aid", np.arange(N_AIDS), name)
    for name in ["user4", "user10", "user20", "user21"]:
        feats[name] = _feature(rng, "session", np.array(sessions), name)
    # user7 covers every session (as the real one), so its int32 columns stay int32
    feats["user7"] = pl.DataFrame({"session": np.array(sessions, np.int32),
                                   "ts_max7": rng.integers(0, 9999, n_sessions).astype(np.int32)})
    embed = rng.normal(size=(N_AIDS, 8)).astype(np.float32)
    return events, cands, _labels(rng, sessions), feats, embed


def _reference(model, events, cands, labels, feats, embed):
    pd_cands = {k: v.rename({"session": "user", "aid": "item"}).to_pandas() for k, v in cands.items()}
    pd_events = events.with_columns(pl.col("ts").dt.epoch("s")).to_pandas()
    pd_feats = {k: v.to_pandas() for k, v in feats.items()}
    out = ref.make(model, pd_cands, labels.to_pandas(), pd_events, pd_feats, embed)
    return pl.from_pandas(out, nan_to_null=False)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("model", ["152", "172", "174"])
def test_matches_reference(model, engine):
    events, cands, labels, feats, embed = _data()
    got = merge.make(model, {k: v.lazy() for k, v in cands.items()}, labels.lazy(), events,
                     {k: v.lazy() for k, v in feats.items()}, embed, engine=_engine(engine))
    want = _reference(model, events, cands, labels, feats, embed)
    assert got.columns == want.columns
    # pandas holds missing floats as NaN, Polars as null
    want = want.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)).cast(got.schema)
    assert_frame_equal(got, want, rel_tol=1e-5, abs_tol=1e-5)
    use = merge.TARGET[model]
    assert got[use].sum() > 0
    if model != "152":  # only sessions with a positive
        assert got.group_by("user").agg(pl.col(use).sum())[use].min() > 0


@pytest.mark.parametrize("model", ["152", "172", "174"])
def test_without_labels(model):
    """The LB rows: the labelled rows of every session, without the target column."""
    events, cands, labels, feats, embed = _data()
    args = ({k: v.lazy() for k, v in cands.items()}, events, {k: v.lazy() for k, v in feats.items()}, embed)
    got = merge.make(model, args[0], None, *args[1:])
    use = merge.TARGET[model]
    assert use not in got.columns
    assert got["user"].n_unique() == events["session"].n_unique()
    # labelled rows exist only for sessions with a positive (except 152): compare those
    want = merge.make(model, args[0], labels.lazy(), *args[1:]).drop(use)
    assert_frame_equal(got.join(want.select("user").unique(), on="user", how="semi", maintain_order="left"), want)


def test_score_columns_hand_checked():
    """Session 0: 709 lists aids 5, 6, 5 (padding repeats 5); a 564 run scores 6 and pads 5."""
    sch = {"session": pl.Int32, "pos": pl.Int32, "aid": pl.Int32, "score": pl.Float64}
    empty = pl.DataFrame(schema=sch)
    cands = {r: empty.lazy() for r in merge.runs_needed("152")}
    cands["709_clicks"] = pl.DataFrame([(0, 0, 5, 3.0), (0, 1, 6, -1.0), (0, 2, 5, -2.0)], schema=sch, orient="row").lazy()
    cands["564_1"] = pl.DataFrame([(0, 0, 6, 2.5), (0, 1, 5, -2.0)], schema=sch, orient="row").lazy()
    labels = pl.DataFrame([(0, CLICKS, [6])], schema={"session": pl.Int32, "type": pl.UInt8,
                                                      "ground_truth": pl.List(pl.Int32)}, orient="row").lazy()
    users = merge.step_b(merge.step_a(cands, labels, "clicks", False), cands,
                         {k: pl.LazyFrame({"aid": [0], "session": [0]}).select(
                             pl.col("aid" if k.startswith("item") else "session").cast(pl.Int32))
                          for k in merge.FEATURES_ITEM_B + ["user4"]}, "clicks").collect()
    assert users.select("item", "pos", "wgt", "clicks", "score_1", "score_3").rows() == [
        (5, 0, 3.0, 0, -3.0, -3.0),  # 564_1 only padded 5 (-2, dropped): -3
        (6, 1, -1.0, 1, 2.5, -3.0),
        (5, 2, -2.0, 0, -3.0, -3.0),
    ]
