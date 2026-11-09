"""Merge candidates, features and targets into the rankers' training rows, following
Chris Deotte's train/make_parquets/script-1.ipynb (Make-152, clicks) and script-2.ipynb
(Make-172 orders, Make-174 carts), steps A-G (see merge.py; step H, `click_oof`, needs
the clicks ranker and is left to train/ranker_models).

Each model is written as <out>/train_v<model>/part-NNN.parquet, one file per session
chunk (the notebooks split 152 into parts A/B to save memory; read the directory with
pl.scan_parquet). Rows within a part are sorted by user, pos.

Usage:
    python train/make_parquets/save_merged.py   # local-validation split, all three models
    python train/make_parquets/save_merged.py --model 172 --chunk 100000 --sessions 20000
    python train/make_parquets/save_merged.py --events data/lb_test.parquet --labels none \
        --candidates data/candidate_scores_lb --features data/item_user_features_lb --out data/infer_with_features_lb
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

import merge

FEATURE_NAMES = ["item4", "item10", "item12", "item13", "item20", "item21", "item22", "item40", "item41",
                 "user4", "user7", "user10", "user20", "user21"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=list(merge.TARGET), default=list(merge.TARGET))
    parser.add_argument("--events", default="data/valid_test.parquet", help="the sessions to rank, sorted by session, ts")
    parser.add_argument("--labels", default="data/valid_test_labels.parquet",
                        help="session, type, ground_truth; 'none' for the LB rows (no target column, all sessions)")
    parser.add_argument("--candidates", default="data/candidate_scores_valid", help="save_candidates.py output")
    parser.add_argument("--features", default="data/item_user_features_valid",
                        help="save_features.py output (and item_embed_32.npy from save_embeddings.py)")
    parser.add_argument("--embed", help="item embeddings .npy (default: <features>/item_embed_32.npy)")
    parser.add_argument("--giba-embed", help="Giba's embbedings_fm10.npy for step G (default: --embed, as the notebooks without it)")
    parser.add_argument("--out", default="data/train_with_features_valid", help="output directory")
    parser.add_argument("--chunk", type=int, default=200_000, help="session-id range per part")
    parser.add_argument("--sessions", type=int, help="only the first N sessions (smoke tests)")
    parser.add_argument("--engine", choices=["in-memory", "streaming", "gpu"], default="in-memory")
    parser.add_argument("--overwrite", action="store_true", help="recompute parts that already exist")
    args = parser.parse_args()

    engine = pl.GPUEngine(raise_on_fail=True) if args.engine == "gpu" else args.engine
    features = Path(args.features)
    feats = {name: pl.scan_parquet(features / f"{name}.parquet") for name in FEATURE_NAMES}
    embed = np.load(args.embed or features / "item_embed_32.npy")
    giba = np.load(args.giba_embed) if args.giba_embed else None
    labels = None if args.labels == "none" else pl.scan_parquet(args.labels)
    events = pl.scan_parquet(args.events).with_columns(pl.col("session", "aid").cast(pl.Int32))
    if args.sessions:
        first = events.select(pl.col("session").unique().sort().head(args.sessions)).collect()
        events = events.join(first.lazy(), on="session", how="semi")
    lo, hi = events.select(lo=pl.col("session").min(), hi=pl.col("session").max()).collect().row(0)

    total = time.perf_counter()
    for model in args.model:
        cands = {run: pl.scan_parquet(Path(args.candidates) / f"{run}.parquet") for run in merge.runs_needed(model)}
        out = Path(args.out) / f"train_v{model}"
        out.mkdir(parents=True, exist_ok=True)
        rows = 0
        for i, start in enumerate(range(lo, hi + 1, args.chunk)):
            path = out / f"part-{i:03d}.parquet"
            if path.exists() and not args.overwrite:
                logger.info(f"{model} part {i}: exists, skipping")
                continue
            t = time.perf_counter()
            chunk = events.filter(pl.col("session").is_between(start, start + args.chunk, closed="left"))
            df = merge.make(model, cands, labels, chunk, feats, embed, giba, engine=engine)
            tmp = path.with_suffix(".tmp")
            df.write_parquet(tmp)
            tmp.rename(path)
            rows += df.height
            logger.info(f"{model} part {i}: {df.height:,} rows x {df.width} columns in {time.perf_counter() - t:.1f}s")
        logger.info(f"{model}: {rows:,} rows written to {out}")
    logger.info(f"total {time.perf_counter() - total:.1f}s")


if __name__ == "__main__":
    main()
