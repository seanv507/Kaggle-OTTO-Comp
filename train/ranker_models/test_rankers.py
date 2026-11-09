import json

import numpy as np
import polars as pl
import pytest

import rankers
from event_types import CARTS, CLICKS, ORDERS


def test_sample_negatives():
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    idx = rankers.sample_negatives(y, 0.4, seed=0)
    assert np.all(np.diff(idx) > 0)  # sorted, unique
    assert {1, 4} <= set(idx)  # every positive
    assert (y[idx] == 0).sum() == round(0.4 * 8)


def test_folds_split_by_user():
    users = np.repeat(np.arange(20), 5)
    fs = rankers.folds(users, 5)
    assert len(fs) == 5
    for tr, va in fs:
        assert not set(users[tr]) & set(users[va])
    assert sorted(np.concatenate([va for _, va in fs]).tolist()) == list(range(100))


def test_recall_at_20():
    pred = pl.DataFrame({
        "user": [1] * 22 + [2, 2],
        "item": list(range(22)) + [5, 6],
        "p": [1.0] * 20 + [0.5, 2.0] + [0.9, 0.1],  # user 1: item 21 ranks first, 20 is cut
    })
    labels = pl.DataFrame(
        [(1, CLICKS, [21, 20]), (2, CLICKS, [6]), (3, CLICKS, [1, 2]), (2, ORDERS, [5])],
        schema={"session": pl.Int32, "type": pl.UInt8, "ground_truth": pl.List(pl.Int32)}, orient="row",
    )
    # hits: user 1 -> 21 (20 is 21st), user 2 -> 6, user 3 has no predictions; total 2 + 1 + 2
    assert rankers.recall_at_20(pred, labels, "clicks") == pytest.approx(2 / 5)
    assert rankers.recall_at_20(pred, labels, "orders") == 1.0


def _rows(target, n_users=60, ct=10, seed=0):
    """Synthetic merged rows: user, item, pos, wgt, <target>, score_30, f; the target
    follows wgt and f."""
    rng = np.random.default_rng(seed)
    n = n_users * ct
    wgt = rng.normal(size=n).astype(np.float32)
    f = rng.normal(size=n).astype(np.float32)
    y = ((wgt + f + rng.normal(scale=0.5, size=n)) > 1.5).astype(np.int8)
    return pl.DataFrame({
        "user": np.repeat(np.arange(n_users, dtype=np.int32), ct),
        "item": rng.integers(0, 1000, n).astype(np.int32),
        "pos": np.tile(np.arange(ct, dtype=np.int32), n_users),
        "wgt": wgt,
        target: y,
        "score_30": rng.normal(size=n).astype(np.float32),
        "f": f,
    })


def _parts(rows, directory, n=3):
    """Write rows as n parts split at user boundaries, as save_merged.py does."""
    directory.mkdir()
    bounds = np.linspace(0, rows["user"].max() + 1, n + 1).astype(int)
    paths = []
    for k in range(n):
        path = directory / f"part-{k:03d}.parquet"
        rows.filter(pl.col("user").is_between(bounds[k], bounds[k + 1], closed="left")).write_parquet(path)
        paths.append(path)
    return paths


def test_load_training_rows_samples_per_part(tmp_path):
    rows = _rows("clicks")
    parts = _parts(rows, tmp_path / "rows")
    features, X, y, users, keep = rankers.load_training_rows(parts, "clicks", 0.4, seed=7)
    assert features == ["pos", "wgt", "score_30", "f"]
    assert len(keep) == 3 and len(X) == len(y) == len(users) == sum(len(k) for k in keep)
    for k, (path, idx) in enumerate(zip(parts, keep, strict=True)):
        part_y = pl.read_parquet(path)["clicks"].to_numpy()
        np.testing.assert_array_equal(idx, rankers.sample_negatives(part_y, 0.4, 7 + k))
    assert np.all(np.diff(users) >= 0)  # still sorted by user (qid needs it)
    np.testing.assert_array_equal(X[:, 1], pl.concat([pl.read_parquet(p)[i] for p, i in zip(parts, keep, strict=True)])["wgt"])


@pytest.mark.parametrize("target,neg_frac", [("clicks", 0.4), ("orders", None)])
def test_train_end_to_end(tmp_path, target, neg_frac):
    rows = _rows(target)
    parts = _parts(rows, tmp_path / "rows")
    oof = rankers.train(parts, target, tmp_path, n_folds=3, neg_frac=neg_frac, num_boost_round=20,
                        early_stopping_rounds=5, verbose_eval=False)
    assert oof.select("user", "item").equals(rows.select("user", "item"))  # every row, in order
    assert oof.columns == ["user", "item", "p", "sampled"] and oof.height == rows.height
    assert json.loads((tmp_path / "features.json").read_text()) == ["pos", "wgt", "score_30", "f"]
    assert sorted(p.name for p in tmp_path.glob("fold*.ubj")) == ["fold0.ubj", "fold1.ubj", "fold2.ubj"]
    users = np.concatenate([np.load(tmp_path / f"valid_users_f{k}.npy") for k in range(3)])
    assert sorted(users.tolist()) == list(range(60))  # each session held out exactly once
    assert oof["p"].null_count() == 0
    assert oof["sampled"].all() == (neg_frac is None)
    # the model learnt something: positives score higher out of fold
    p, y = oof["p"].to_numpy(), rows[target].to_numpy()
    assert p[y == 1].mean() > p[y == 0].mean()


def test_click_oof_swaps_and_uses_the_held_out_fold(tmp_path):
    clicks = _rows("clicks")
    rankers.train(_parts(clicks, tmp_path / "rows"), "clicks", tmp_path, n_folds=3, num_boost_round=10, early_stopping_rounds=5,
                  verbose_eval=False)
    rows = _rows("orders", n_users=70, seed=1)  # users 60-69 are in no clicks fold
    got = rankers.click_oof(rows, tmp_path)
    assert got.columns == rows.columns + ["click_oof"]
    assert (got.filter(pl.col("user") >= 60)["click_oof"] == 0).all()
    # user u in valid fold k is scored by fold k's model on the swapped columns
    swapped = rows.rename({"wgt": "score_30", "score_30": "wgt"})
    for k in range(3):
        users = np.load(tmp_path / f"valid_users_f{k}.npy")
        sel = swapped.filter(pl.col("user").is_in(users.tolist()))
        model = rankers.load_model(tmp_path / f"fold{k}.ubj")
        want = rankers.predict(model, sel.select(json.loads((tmp_path / "features.json").read_text())).to_numpy())
        np.testing.assert_allclose(got.filter(pl.col("user").is_in(users.tolist()))["click_oof"].to_numpy(), want,
                                   rtol=1e-6)


def test_click_oof_average_is_the_mean_of_the_folds(tmp_path):
    clicks = _rows("clicks")
    rankers.train(_parts(clicks, tmp_path / "rows"), "clicks", tmp_path, n_folds=3, num_boost_round=10, early_stopping_rounds=5,
                  verbose_eval=False)
    rows = _rows("orders", n_users=70, seed=1).drop("orders")  # LB rows: no target
    got = rankers.click_oof(rows, tmp_path, average=True)
    assert got.columns == rows.columns + ["click_oof"]
    X = rows.rename({"wgt": "score_30", "score_30": "wgt"}).select(json.loads((tmp_path / "features.json").read_text())).to_numpy()
    want = np.mean([rankers.predict(rankers.load_model(tmp_path / f"fold{k}.ubj"), X) for k in range(3)], axis=0)
    np.testing.assert_allclose(got["click_oof"].to_numpy(), want, rtol=1e-5)


def test_params():
    assert rankers.params("clicks")["objective"] == "binary:logistic"
    assert rankers.params("carts", "cuda")["objective"] == "rank:pairwise"
    assert rankers.params("orders", "cuda")["device"] == "cuda"
    assert rankers.TYPE == {"clicks": CLICKS, "carts": CARTS, "orders": ORDERS}
