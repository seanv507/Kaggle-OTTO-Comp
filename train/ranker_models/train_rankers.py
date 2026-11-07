"""Train the XGBoost rankers of XGB-186-CLICKS / XGB-406-CARTS / XGB-412-ORDERS (see
rankers.py) on save_merged.py's rows, and print OTTO recall@20 of the out-of-fold
predictions.

Clicks runs first: its fold models give the carts and orders rows their `click_oof`
feature (step H of Make-172/174), written once as <rows>/train_v<model>_X/part-NNN.parquet.
Each target writes <out>/<target>/: fold<k>.ubj, valid_users_f<k>.npy, features.json,
importance.csv and oof.parquet (user, item, p, sampled).

Usage:
    python train/ranker_models/train_rankers.py   # local-validation split, all targets
    python train/ranker_models/train_rankers.py --target clicks --rounds 50 --device cpu
"""

import argparse
import time
from pathlib import Path

import polars as pl
import xgboost as xgb
from loguru import logger

import rankers

WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}


def _device(arg):
    if arg:
        return arg
    try:
        xgb.train({"device": "cuda", "tree_method": "hist"}, xgb.DMatrix([[0.0]], label=[0.0]), 1)
        return "cuda"
    except xgb.core.XGBoostError:
        return "cpu"


def add_click_oof(rows_dir: Path, model: str, clicks_models: Path, device: str, overwrite: bool) -> Path:
    out = rows_dir / f"train_v{model}_X"
    out.mkdir(exist_ok=True)
    for part in sorted((rows_dir / f"train_v{model}").glob("part-*.parquet")):
        path = out / part.name
        if path.exists() and not overwrite:
            continue
        rankers.click_oof(pl.read_parquet(part), clicks_models, device).write_parquet(path)
        logger.info(f"click_oof -> {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", nargs="+", choices=list(rankers.ROWS), default=list(rankers.ROWS))
    parser.add_argument("--rows", default="data/train_with_features_valid", help="save_merged.py output")
    parser.add_argument("--labels", default="data/valid_test_labels.parquet")
    parser.add_argument("--out", default="data/models_valid")
    parser.add_argument("--device", help="xgboost device (default: cuda if available)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10_000, help="max boosting rounds")
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", help="rebuild click_oof rows that exist")
    args = parser.parse_args()

    device = _device(args.device)
    rows_dir, out = Path(args.rows), Path(args.out)
    # every session of the rows (train_v152 has them all) counts in recall, with or without candidates
    sessions = pl.scan_parquet(rows_dir / "train_v152").select(session=pl.col("user").unique())
    labels = pl.scan_parquet(args.labels).join(sessions, on="session", how="semi").collect()
    scores = {}
    for target in [t for t in rankers.ROWS if t in args.target]:  # clicks first
        t = time.perf_counter()
        model = rankers.ROWS[target]
        if target == "clicks":
            src = rows_dir / f"train_v{model}"
        else:
            src = add_click_oof(rows_dir, model, out / "clicks", device, args.overwrite)
        rows = pl.read_parquet(src)
        logger.info(f"{target}: {rows.height:,} rows x {rows.width} columns from {src}, device {device}")
        oof = rankers.train(
            rows, target, out / target, device=device, n_folds=args.folds,
            neg_frac=0.4 if target == "clicks" else None, num_boost_round=args.rounds,
            early_stopping_rounds=args.early_stopping,
        )
        oof.write_parquet(out / target / "oof.parquet")
        scores[target] = rankers.recall_at_20(oof, labels, target)
        msg = f"{target}: recall@20 {scores[target]:.5f}"
        if target == "clicks":
            msg += f" (on the sampled rows only, as the notebook: {rankers.recall_at_20(oof.filter(pl.col("sampled")), labels, target):.5f})"
        logger.info(f"{msg} in {time.perf_counter() - t:.0f}s")
    if len(scores) == 3:
        logger.info(f"score {sum(WEIGHTS[k] * v for k, v in scores.items()):.5f}")


if __name__ == "__main__":
    main()
