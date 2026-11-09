import numpy as np
import polars as pl

import predict
from train.ranker_models import rankers


def test_top_items_hand_checked():
    rows = pl.DataFrame({
        "user": np.array([7, 7, 7, 3, 3], np.int32),
        "item": np.array([10, 11, 12, 20, 21], np.int32),
        "pos": np.array([0, 1, 2, 0, 1], np.int32),
    })
    # user 7: 12 best, then 10 and 11 tie (lower pos first); k=2 cuts 11
    got = predict.top_items(rows, [0.5, 0.5, 0.9, 0.1, 0.2], "carts", k=2)
    assert got.columns == ["session_type", "labels"]
    assert got.rows() == [("3_carts", "21 20"), ("7_carts", "12 10")]


def _rows(target, n_users, seed):
    rng = np.random.default_rng(seed)
    n = n_users * 30
    wgt = rng.normal(size=n).astype(np.float32)
    cols = {
        "user": np.repeat(np.arange(n_users, dtype=np.int32), 30),
        "item": rng.integers(0, 1000, n).astype(np.int32),
        "pos": np.tile(np.arange(30, dtype=np.int32), n_users),
        "wgt": wgt,
        "score_30": rng.normal(size=n).astype(np.float32),
    }
    if target:
        cols[target] = ((wgt + rng.normal(scale=0.5, size=n)) > 1.5).astype(np.int8)
    return pl.DataFrame(cols)


def _train(rows, target, directory):
    part = directory / "rows" / target / "part-000.parquet"
    part.parent.mkdir(parents=True)
    rows.write_parquet(part)
    rankers.train([part], target, directory / target, n_folds=3, num_boost_round=10, early_stopping_rounds=5,
                  verbose_eval=False)


def test_predict_target_orders_uses_click_oof_and_fold_means(tmp_path):
    models = tmp_path / "models"
    _train(_rows("clicks", 40, 0), "clicks", models)
    # the orders model is trained with click_oof, as train_rankers.py does
    _train(rankers.click_oof(_rows("orders", 40, 1), models / "clicks"), "orders", models)

    lb = _rows(None, 25, 2)  # LB rows: no target column
    (tmp_path / "lb" / "train_v172").mkdir(parents=True)
    lb.head(300).write_parquet(tmp_path / "lb" / "train_v172" / "part-000.parquet")
    lb.tail(lb.height - 300).write_parquet(tmp_path / "lb" / "train_v172" / "part-001.parquet")

    got = predict.predict_target(tmp_path / "lb", models, "orders", "cpu")
    with_oof = rankers.click_oof(lb, models / "clicks", average=True)
    want = predict.top_items(with_oof, rankers.predict_folds(with_oof, models / "orders"), "orders")
    assert got.equals(want)
    assert got.height == 25
    assert (got["labels"].str.split(" ").list.len() == predict.TOP).all()
