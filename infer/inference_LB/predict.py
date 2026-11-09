"""LB inference and the submission, following Chris Deotte's infer/inference_LB notebooks
LB-21 (clicks, XGB-186), LB-11 (carts, XGB-406), LB-1 (orders, XGB-412) and Make-LB-Sub-3,
plus infer/make_parquets_LB/Make-175-G (`click_oof` on the LB rows).

Input: the LB rows of save_merged.py --labels none (<rows>/train_v152, train_v174,
train_v172) and the rankers of train_rankers.py (<models>/<target>/fold*.ubj). Each row is
scored by the mean of the target's fold models; the carts and orders rows first get
`click_oof`, the mean of the clicks models (rankers.click_oof(average=True)). Each
session's 20 highest-scoring items, best first, are its labels for the target (ties: the
lower candidate position first; the notebooks leave tie order to cudf's sort).

Output: <out> (default submission.csv), the Kaggle format `session_type,labels`
("12899779_clicks", "aid aid ..."), sorted by session_type as Make-LB-Sub-3 does.

Usage:
    python infer/inference_LB/predict.py --rows data/infer_with_features_lb --models data/models_valid
"""

import argparse
import time
from pathlib import Path

import polars as pl
from loguru import logger

from train.ranker_models import rankers

TOP = 20


def top_items(rows: pl.DataFrame, p, target: str, k: int = TOP) -> pl.DataFrame:
    """`session_type, labels` for each user of `rows`: its `k` best items by `p`."""
    return (
        rows.select("user", "item", "pos")
        .with_columns(p=pl.Series(p))
        .sort(["user", "p", "pos"], descending=[False, True, False])
        .group_by("user", maintain_order=True)
        .agg(pl.col("item").head(k))
        .select(
            session_type=pl.col("user").cast(pl.String) + f"_{target}",
            labels=pl.col("item").cast(pl.List(pl.String)).list.join(" "),
        )
    )


def predict_target(rows_dir: Path, models: Path, target: str, device: str) -> pl.DataFrame:
    """Labels of every session in the LB rows of `target`, part by part."""
    parts = sorted((rows_dir / f"train_v{rankers.ROWS[target]}").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parts in {rows_dir / f'train_v{rankers.ROWS[target]}'}")
    out = []
    for part in parts:
        t = time.perf_counter()
        rows = pl.read_parquet(part)
        if target != "clicks":
            rows = rankers.click_oof(rows, models / "clicks", device, average=True)
        out.append(top_items(rows, rankers.predict_folds(rows, models / target, device), target))
        logger.info(f"{target} {part.name}: {rows['user'].n_unique():,} sessions in {time.perf_counter() - t:.1f}s")
    return pl.concat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", default="data/infer_with_features_lb", help="save_merged.py --labels none output")
    parser.add_argument("--models", default="data/models_valid", help="train_rankers.py output")
    parser.add_argument("--out", default="data/submission.csv")
    parser.add_argument("--device", help="xgboost device (default: cuda if available)")
    args = parser.parse_args()

    device = rankers.default_device(args.device)
    rows_dir, models = Path(args.rows), Path(args.models)
    sub = pl.concat([predict_target(rows_dir, models, t, device) for t in rankers.ROWS]).sort("session_type")
    sessions = sub.select(pl.col("session_type").str.split("_").list.first()).n_unique()
    logger.info(f"{sub.height:,} rows ({sessions:,} sessions) -> {args.out}")
    sub.write_csv(args.out)


if __name__ == "__main__":
    main()
