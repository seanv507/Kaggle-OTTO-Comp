"""The leaderboard inputs as the local-validation split moved one week later.

Chris Deotte's infer/ notebooks rerun his train pipeline on all the data (his
data/infer_data), i.e. on the split one week later: train weeks 2-4 and the Kaggle test
week in place of train weeks 1-3 and the truncated week 4 (valid_train, valid_test). The
time constants of the train code (covisits.start_train, end_test, ...) are those of the
valid split, so this module moves the LB events back by exactly 7 days:

- lb_train: train.parquet from 2022-08-07 22:00 (week 1 dropped), ts - 7 days, so it
  spans 2022-07-31 22:00 to 2022-08-21 21:59:59, as valid_train;
- lb_test: test.parquet with ts - 7 days (2022-08-21 22:00 to 2022-08-28 21:59:51), as
  valid_test.

A whole number of weeks keeps day of week and hour of day. Every stage then runs
unchanged on lb_train / lb_test and the LB features mean what the rankers (trained on the
valid split) learnt, which is the choice made here. Chris's infer/ notebooks are an older
variant of the train ones (some matrices keep a different top k, v34 drops the self
pairs v115 keeps, list truncations differ, 'less'/'less2' are swapped) and keep week 1
in some steps; see docs/infer.md.

Usage:
    python infer/lb_data.py                     # data/train.parquet, data/test.parquet -> data/lb_*.parquet
"""

import argparse
import datetime
from pathlib import Path

import polars as pl
from loguru import logger

SHIFT = datetime.timedelta(days=7)
LB_START = datetime.datetime(2022, 8, 7, 22)  # start of train week 2 (valid_train's start + 7 days)


def lb_events(train: pl.LazyFrame | pl.DataFrame, test: pl.LazyFrame | pl.DataFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """(lb_train, lb_test): session/ts order and dtypes kept, ts moved back by SHIFT."""
    shift = pl.col("ts") - SHIFT
    lb_train = train.lazy().filter(pl.col("ts") >= LB_START).with_columns(shift)
    return lb_train, test.lazy().with_columns(shift)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/train.parquet")
    parser.add_argument("--test", default="data/test.parquet")
    parser.add_argument("--out", default="data", help="directory for lb_train.parquet and lb_test.parquet")
    args = parser.parse_args()

    out = Path(args.out)
    frames = lb_events(pl.scan_parquet(args.train), pl.scan_parquet(args.test))
    for name, lf in zip(["lb_train", "lb_test"], frames, strict=True):
        path, tmp = out / f"{name}.parquet", out / f"{name}.parquet.tmp"
        lf.sink_parquet(tmp)
        stats = pl.scan_parquet(tmp).select(pl.len(), pl.col("ts").min().alias("lo"), pl.col("ts").max().alias("hi")).collect()
        tmp.rename(path)
        logger.info(f"{path}: {stats['len'][0]:,} events, {stats['lo'][0]} to {stats['hi'][0]}")


if __name__ == "__main__":
    main()
