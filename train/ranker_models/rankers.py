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
- clicks negatives are sampled per merged part (seeded), as the notebook sampled its parts
  A/B; the parts are read one at a time and the fold matrices built in batches.
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


class _Batches(xgb.DataIter):
    """Rows `idx` of X / y / qid in batches, so a fold's matrix never copies all its rows
    (the notebooks' IterLoadForDMatrix)."""

    def __init__(self, X, y, qid, idx, batch_size: int = 1 << 20):
        self.X, self.y, self.qid, self.idx, self.batch_size = X, y, qid, idx, batch_size
        self.it = 0
        super().__init__()

    def reset(self):
        self.it = 0

    def next(self, input_data):
        a = self.it * self.batch_size
        if a >= len(self.idx):
            return False
        i = self.idx[a : a + self.batch_size]
        input_data(data=self.X[i], label=self.y[i], qid=self.qid[i])
        self.it += 1
        return True


def load_training_rows(parts: list[Path], target: str, neg_frac: float | None, seed: int = 42):
    """The training sample of `parts` (all positives and `neg_frac` of each part's
    negatives, as the notebook sampled each of its parts A/B; all rows if None).

    Returns features, X (float32), y, users, and per part the kept row indices. Two
    passes (the target column first) so X is allocated once."""
    keep = []
    for k, path in enumerate(parts):
        y = pl.read_parquet(path, columns=[target])[target].to_numpy()
        keep.append(sample_negatives(y, neg_frac, seed + k) if neg_frac else np.arange(len(y)))
    features = feature_columns(pl.read_parquet_schema(parts[0]).names(), target)
    n = sum(len(k) for k in keep)
    X = np.empty((n, len(features)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    users = np.empty(n, dtype=np.int32)
    a = 0
    for path, idx in zip(parts, keep, strict=True):
        df = pl.read_parquet(path)[idx]
        b = a + len(idx)
        X[a:b] = df.select(features).to_numpy()
        y[a:b] = df[target].to_numpy()
        users[a:b] = df["user"].to_numpy()
        a = b
    return features, X, y, users, keep


def train(
    parts: list[Path],
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
    """Train the fold models on the merged rows in `parts` (sorted by user across parts)
    into `out`, and return the out-of-fold predictions `user, item, p, sampled` for every
    row (`sampled`: the row was in the training sample).

    Writes fold<k>.ubj, valid_users_f<k>.npy, features.json and importance.csv."""
    out.mkdir(parents=True, exist_ok=True)
    features, X, y, users, keep = load_training_rows(parts, target, neg_frac, seed)
    (out / "features.json").write_text(json.dumps(features))
    logger.info(f"{target}: training sample {len(y):,} rows x {len(features)} features")
    models, gains = [], []
    for fold, (tr, va) in enumerate(folds(users, n_folds)):
        t = time.perf_counter()
        np.save(out / f"valid_users_f{fold}.npy", np.unique(users[va]))
        dtrain = xgb.QuantileDMatrix(_Batches(X, y, users, tr), max_bin=256)
        dvalid = xgb.QuantileDMatrix(_Batches(X, y, users, va), max_bin=256, ref=dtrain)
        model = xgb.train(
            params(target, device, seed), dtrain, num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval,
        )
        model.save_model(out / f"fold{fold}.ubj")
        models.append(model)
        gains.append(model.get_score(importance_type="gain"))
        logger.info(f"{target} fold {fold}: best iteration {model.best_iteration}, "
                    f"valid map {model.best_score:.5f} ({time.perf_counter() - t:.0f}s)")
        del dtrain, dvalid
    del X
    imp = pl.DataFrame({"feature": features}).with_columns(
        pl.Series(f"gain_{k}", [g.get(f, 0.0) for f in features]) for k, g in enumerate(gains)
    ).with_columns(gain=pl.mean_horizontal(pl.exclude("feature"))).sort("gain", descending=True)
    imp.write_csv(out / "importance.csv")
    return out_of_fold(parts, out, models, features, keep)


def out_of_fold(parts, out: Path, models, features, keep) -> pl.DataFrame:
    """Every row of `parts` (sampled or not) scored by the model of its session's fold."""
    fold_users = [np.load(out / f"valid_users_f{k}.npy") for k in range(len(models))]
    all_users = np.concatenate(fold_users)
    fold_of = np.concatenate([np.full(len(u), k) for k, u in enumerate(fold_users)])
    order = np.argsort(all_users)
    all_users, fold_of = all_users[order], fold_of[order]
    res = []
    for path, idx in zip(parts, keep, strict=True):
        df = pl.read_parquet(path, columns=["user", "item", *features])
        users = df["user"].to_numpy()
        pos = np.clip(np.searchsorted(all_users, users), 0, len(all_users) - 1)
        fold = np.where(all_users[pos] == users, fold_of[pos], -1)  # -1: session not trained on
        X = df.select(features).to_numpy().astype(np.float32, copy=False)
        p = np.zeros(len(df), dtype=np.float32)
        for k, model in enumerate(models):
            m = fold == k
            if m.any():
                p[m] = predict(model, X[m])
        sampled = np.zeros(len(df), dtype=bool)
        sampled[idx] = True
        res.append(df.select("user", "item").with_columns(p=pl.Series(p), sampled=pl.Series(sampled)))
    return pl.concat(res)


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
