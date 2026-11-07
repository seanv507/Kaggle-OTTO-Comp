"""XGBoost rankers of Chris Deotte's XGB-186-CLICKS, XGB-406-CARTS and XGB-412-ORDERS
notebooks, and step H of Make-172/174 (`click_oof`).

Each target trains 5 GroupKFold (by session) folds on the merged rows of
train/make_parquets (train_v152 for clicks, train_v174 for carts, train_v172 for
orders), with early stopping on the held-out fold, and is scored by OTTO recall@20 of
its out-of-fold predictions. Features are the rows' `columns[2:]` minus the target, in
file order, as the notebooks.

- clicks: `binary:logistic` (scale_pos_weight 8) on all positives and 40% of the
  negatives; the session groups only feed the `map` eval metric.
- carts / orders: `rank:pairwise`, one group of 100 rows per session (every kept session
  has model 709's 100 candidates), all rows, plus `click_oof`.
- click_oof (step H): the clicks models applied to the carts / orders rows, each session
  by the fold model that did not train on it, with `wgt` and `score_30` swapped so they
  mean what they meant to the clicks model (709's clicks / orders scores).

Differences from the notebooks:
- XGBoost 3: `tree_method='hist'` with `device`, `QuantileDMatrix` (the valid matrix
  shares the train bins), and predictions use the best early-stopping iteration.
- clicks negatives are sampled from all rows at once (seeded), not per part A/B.
- clicks recall is reported twice: on the sampled rows (as the notebook, optimistic: each
  session's positives compete with 40% of its negatives) and on all rows.
The CatBoost notebooks (CAT-200/203) are not ported: the LB inference uses XGBoost only.
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import GroupKFold

from event_types import CARTS, CLICKS, ORDERS

ROWS = {"clicks": "152", "carts": "174", "orders": "172"}  # train_v<model> of make_parquets
VERSION = {"clicks": 186, "carts": 406, "orders": 412}
TYPE = {"clicks": CLICKS, "carts": CARTS, "orders": ORDERS}
CT = 100  # candidates per session of model 709
SWAP = ("wgt", "score_30")


def params(target: str, device: str = "cpu", seed: int = 42, lr: float = 0.1) -> dict:
    p = {
        "max_depth": 4,
        "learning_rate": lr,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "eval_metric": "map",
        "tree_method": "hist",
        "device": device,
        "seed": seed,
    }
    if target == "clicks":
        return p | {"objective": "binary:logistic", "scale_pos_weight": 8}
    return p | {"objective": "rank:pairwise"}


def feature_columns(columns: list[str], target: str) -> list[str]:
    return [c for c in columns[2:] if c != target]


def sample_negatives(y: np.ndarray, frac: float, seed: int = 42) -> np.ndarray:
    """Sorted row indices: every positive and round(frac * negatives) random negatives
    (pandas' sample(frac=...) count)."""
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(y == 0)
    picked = rng.choice(neg, size=int(round(frac * len(neg))), replace=False)
    return np.sort(np.concatenate([np.flatnonzero(y == 1), picked]))


def group_sizes(users: np.ndarray) -> np.ndarray:
    """Row counts of consecutive runs of the same user (rows are grouped by user)."""
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    return np.diff(np.r_[starts, len(users)])


def folds(users: np.ndarray, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """sklearn's GroupKFold by user, as the notebooks."""
    return list(GroupKFold(n_splits=n_folds).split(np.zeros(len(users)), groups=users))


def recall_at_20(pred: pl.DataFrame, labels: pl.DataFrame, target: str) -> float:
    """OTTO recall of each session's top 20 `item` by `p`, over every labelled session of
    `labels` (sessions without predictions score 0, as the notebooks' fill_up)."""
    top = (
        pred.sort(["user", "p"], descending=[False, True], maintain_order=True)
        .group_by("user", maintain_order=True).head(20)
        .select("user", "item")
    )
    gt = labels.filter(pl.col("type") == TYPE[target]).select(user=pl.col("session"), item=pl.col("ground_truth"))
    hits = gt.explode("item", empty_as_null=False).join(top.unique(), on=["user", "item"], how="semi").height
    total = gt.select(pl.col("item").list.len().clip(0, 20).sum()).item()
    return hits / total


def _matrix(X, y, users, ref=None):
    return xgb.QuantileDMatrix(X, label=y, group=group_sizes(users), max_bin=256, ref=ref)


def train(
    rows: pl.DataFrame,
    target: str,
    out: Path,
    device: str = "cpu",
    n_folds: int = 5,
    neg_frac: float | None = None,
    num_boost_round: int = 10_000,
    early_stopping_rounds: int = 200,
    verbose_eval: int | bool = 100,
    seed: int = 42,
) -> pl.DataFrame:
    """Train the fold models into `out` and return the out-of-fold predictions
    `user, item, p, sampled` for every row (`sampled`: the row was in the training sample).

    Writes fold<k>.ubj, valid_users_f<k>.npy, features.json and importance.csv."""
    out.mkdir(parents=True, exist_ok=True)
    features = feature_columns(rows.columns, target)
    (out / "features.json").write_text(json.dumps(features))
    users = rows["user"].to_numpy()
    y = rows[target].to_numpy()
    sampled = sample_negatives(y, neg_frac, seed) if neg_frac else np.arange(len(rows))
    X = rows.select(features).to_numpy().astype(np.float32, copy=False)
    oof = np.zeros(len(rows), dtype=np.float32)
    gains = []
    for fold, (tr, va) in enumerate(folds(users[sampled], n_folds)):
        t = time.perf_counter()
        tr, va = sampled[tr], sampled[va]
        valid_users = np.unique(users[va])
        np.save(out / f"valid_users_f{fold}.npy", valid_users)
        dtrain = _matrix(X[tr], y[tr], users[tr])
        dvalid = _matrix(X[va], y[va], users[va], ref=dtrain)
        model = xgb.train(
            params(target, device, seed), dtrain, num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval,
        )
        model.save_model(out / f"fold{fold}.ubj")
        gains.append(model.get_score(importance_type="gain"))
        # predict every row of the held-out sessions (sampled or not)
        rows_va = np.flatnonzero(np.isin(users, valid_users))
        oof[rows_va] = predict(model, X[rows_va])
        logger.info(f"{target} fold {fold}: best iteration {model.best_iteration}, "
                    f"valid map {model.best_score:.5f} ({time.perf_counter() - t:.0f}s)")
    imp = pl.DataFrame({"feature": features}).with_columns(
        pl.Series(f"gain_{k}", [g.get(f, 0.0) for f in features]) for k, g in enumerate(gains)
    ).with_columns(gain=pl.mean_horizontal(pl.exclude("feature"))).sort("gain", descending=True)
    imp.write_csv(out / "importance.csv")
    in_sample = np.zeros(len(rows), dtype=bool)
    in_sample[sampled] = True
    return rows.select("user", "item").with_columns(p=pl.Series(oof), sampled=pl.Series(in_sample))


def predict(model: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Predictions of the best early-stopping iteration."""
    return model.predict(xgb.DMatrix(X), iteration_range=(0, model.best_iteration + 1))


def load_model(path: Path) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(path)
    return model


def click_oof(rows: pl.DataFrame, clicks_models: Path, device: str = "cpu") -> pl.DataFrame:
    """Step H: `rows` (train_v172 / train_v174) with `click_oof` appended."""
    features = json.loads((clicks_models / "features.json").read_text())
    swapped = rows.with_columns(pl.col(SWAP[0]).alias(SWAP[1]), pl.col(SWAP[1]).alias(SWAP[0]))
    X = swapped.select(features).to_numpy().astype(np.float32, copy=False)
    users = rows["user"].to_numpy()
    out = np.zeros(len(rows), dtype=np.float32)  # sessions in no clicks fold keep 0, as the notebook
    for path in sorted(clicks_models.glob("fold*.ubj")):
        fold = path.stem.removeprefix("fold")
        mask = np.isin(users, np.load(clicks_models / f"valid_users_f{fold}.npy"))
        model = load_model(path)
        model.set_param({"device": device})
        out[mask] = predict(model, X[mask])
    return rows.with_columns(click_oof=pl.Series(out))
