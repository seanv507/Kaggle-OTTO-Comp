"""Train and save the 32-d item embeddings of embeddings-for-train.ipynb (see
embeddings.py), from the consecutive aids of the train and test sessions.

Writes <out>/item_embed_<dim>.npy, a (max aid + 1, dim) float32 array indexed by aid,
as the notebook's np.save.

Usage:
    python train/item_user_features/save_embeddings.py   # local-validation split
    python train/item_user_features/save_embeddings.py --epochs 2 --valid 10000000
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

import embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/valid_train.parquet", help="train events, sorted by session, ts")
    parser.add_argument("--test", default="data/valid_test.parquet", help="test events, sorted by session, ts")
    parser.add_argument("--out", default="data/item_user_features_valid", help="output directory")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--valid", type=int, default=0, help="hold out the last N pairs and log accuracy on them")
    parser.add_argument("--device", help="torch device (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dot", action="store_true",
                        help="train on the dot product, not per coordinate as the notebook did (see embeddings.py)")
    args = parser.parse_args()

    pairs = embeddings.next_pairs(pl.scan_parquet(args.train), pl.scan_parquet(args.test)).collect()
    n_aids = max(pl.scan_parquet(p).select(pl.col("aid").max()).collect().item() for p in [args.train, args.test]) + 1
    emb = embeddings.train_embeddings(
        pairs, n_aids=n_aids, dim=args.dim, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        n_valid=args.valid, device=args.device, seed=args.seed, per_coordinate=not args.dot,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"item_embed_{args.dim}.npy"
    np.save(path, emb)
    logger.info(f"saved {path}: {emb.shape}")


if __name__ == "__main__":
    main()
