"""Compute and save the item and user features, following Chris Deotte's
train/item_user_features/script.ipynb (the item-features-* / user-features-* notebooks).

Each `features.FEATURES` entry is saved as <out>/<name>.parquet (the notebook's
<name>.pqt), keyed by an `aid` or `session` column (the notebooks' index), sorted by it.
Item features read --train or --test as their notebook did; user features read --test.
The embeddings notebook (embeddings-for-train.ipynb, a PyTorch matrix factorisation) is
not converted here.

Usage:
    python train/item_user_features/save_features.py   # local-validation split
    python train/item_user_features/save_features.py --only item4 user7 --engine gpu
"""

import argparse
import time
from pathlib import Path

import polars as pl
from loguru import logger

import features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/valid_train.parquet", help="train events, sorted by session, ts")
    parser.add_argument("--test", default="data/valid_test.parquet", help="test events, sorted by session, ts")
    parser.add_argument("--out", default="data/item_user_features_valid", help="output directory")
    parser.add_argument("--only", nargs="+", metavar="NAME", help=f"features to make (default: all): {' '.join(features.FEATURES)}")
    parser.add_argument("--engine", choices=["streaming", "in-memory", "gpu"], default="streaming")
    parser.add_argument("--overwrite", action="store_true", help="recompute files that already exist")
    args = parser.parse_args()

    if unknown := set(args.only or []) - set(features.FEATURES):
        parser.error(f"unknown features: {sorted(unknown)}")
    engine = pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine
    events = {"train": pl.scan_parquet(args.train), "test": pl.scan_parquet(args.test)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = time.perf_counter()
    for name, (fn, which) in features.FEATURES.items():
        if args.only and name not in args.only:
            continue
        path = out / f"{name}.parquet"
        if path.exists() and not args.overwrite:
            logger.info(f"{name}: exists, skipping")
            continue
        t = time.perf_counter()
        df = fn(events[which]).collect(engine=engine)
        df = df.sort(df.columns[0])
        tmp = path.with_suffix(".tmp")
        df.write_parquet(tmp)
        tmp.rename(path)
        logger.info(f"{name}: {df.height:,} rows x {df.width - 1} features from {which} in {time.perf_counter() - t:.1f}s")
    logger.info(f"total {time.perf_counter() - total:.1f}s")


if __name__ == "__main__":
    main()
