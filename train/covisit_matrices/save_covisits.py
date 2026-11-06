"""Compute and save all covisitation matrices, following Chris Deotte's
train/covisit_matrices/script.ipynb (same matrices, same order).

Each matrix is saved as <out>/<name>.parquet with columns aid, aid_y, value: the top
aid_y per aid, sorted by aid, value descending, aid_y. <name> is the reference pickle's
name without the `_<PART>` suffix (the reference split some matrices into aid ranges
to save memory; here each matrix is one file). Note the reference names don't always
match the number kept, e.g. top_40_aids_v220 keeps the top 80.

Usage:
    python train/covisit_matrices/save_covisits.py   # all matrices, A100 defaults
    python train/covisit_matrices/save_covisits.py --only 90 95 --engine streaming
    python train/covisit_matrices/save_covisits.py --chunk 116=100000 --overwrite

Engines and session chunk sizes default to what worked fastest on an A100-SXM4-80GB
(2026-11-05, see covisitations.ipynb). Some chunk sizes are not about speed: the
all-pairs self-joins of 116 and 217-235 exceed cudf's 2^31-row column limit on
larger chunks, on any GPU. On another machine, override with --engine / --chunk.
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

# RapidsMPF's pinned host memory fails to start when the process's NUMA node has no
# GPU ("No GPUs found on current NUMA node"), as on the RunPod A100 pods. Disabling it
# costs little; set RAPIDSMPF_PINNED_MEMORY=true to re-enable.
os.environ.setdefault("RAPIDSMPF_PINNED_MEMORY", "false")

import polars as pl  # noqa: E402
from loguru import logger  # noqa: E402

import covisits  # noqa: E402


@dataclass(frozen=True)
class Matrix:
    ver: str
    name: str  # reference file name, without the _<PART> suffix
    data: str  # "events" (train + test), "test" (test only) or "train_test" (both args)
    engine: str  # "gpu", "gpu-inmem" or "streaming"
    session_chunk: int | None


# In the order of script.ipynb.
MATRICES = [
    Matrix("115", "top_40_aids_v115", "events", "gpu", 4_000_000),  # no chunking overflows cudf
    Matrix("116", "top_40_aids_v116", "events", "gpu", 200_000),  # all pairs: 2^31-row limit
    Matrix("155", "top_40_aids_v155_d0", "test", "gpu", None),
    Matrix("157", "top_80_aids_v157_d0", "test", "gpu", None),
    # the default (streaming) GPU executor fails on a concat of parquet scans
    Matrix("165", "top_40_aids_v165", "train_test", "gpu-inmem", None),
    Matrix("166", "top_40_aids_v166", "train_test", "gpu-inmem", None),
    Matrix("167", "top_40_aids_v167", "events", "gpu", None),
    Matrix("168", "top_40_aids_v168", "events", "gpu", None),
    Matrix("217", "top_40_aids_v217", "events", "gpu", 250_000),  # 2^31-row limit
    Matrix("220", "top_40_aids_v220", "events", "gpu", 250_000),  # 2^31-row limit
    Matrix("226", "top_40_aids_v226", "events", "gpu", None),
    Matrix("232", "top_40_aids_v232", "events", "gpu", 250_000),  # 2^31-row limit
    Matrix("235", "top_40_aids_v235", "events", "gpu", 250_000),  # 2^31-row limit
    Matrix("239", "top_40_aids_v239", "train_test", "gpu", 2_000_000),
    Matrix("700", "top_40_aids_v700", "events", "gpu", None),
    Matrix("701", "top_40_aids_v701", "events", "gpu", None),
    Matrix("93", "top_80_aids_v93", "events", "gpu", 2_000_000),  # no chunking: out of GPU memory
    Matrix("90", "top_40_buy2buy_v90", "events", "gpu", None),
    Matrix("95", "top_40_orders_carts_v95", "events", "gpu", None),
    Matrix("99", "top_40_buy2buy_v99", "events", "gpu", None),
]


def make_engine(name: str):
    if name == "gpu":
        return pl.GPUEngine(raise_on_fail=True)
    if name == "gpu-inmem":
        return pl.GPUEngine(raise_on_fail=True, executor="in-memory")
    return name


def parse_chunk(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/train.parquet", help="train events, sorted by session, ts")
    parser.add_argument("--test", default="data/test.parquet", help="test events, sorted by session, ts")
    parser.add_argument("--out", default="data/covisit_matrices", help="output directory")
    parser.add_argument("--only", nargs="+", metavar="VER", help="matrix versions to make (default: all)")
    parser.add_argument("--engine", choices=["gpu", "gpu-inmem", "streaming", "in-memory"],
                        help="engine for every matrix, overriding the defaults")
    parser.add_argument("--chunk", nargs="+", default=[], metavar="VER=SESSIONS",
                        help="session chunk per matrix, e.g. 116=100000 or 90=none")
    parser.add_argument("--overwrite", action="store_true", help="recompute matrices that already exist")
    args = parser.parse_args()

    chunks = {}
    for item in args.chunk:
        ver, _, value = item.partition("=")
        chunks[ver] = parse_chunk(value)

    known = {m.ver for m in MATRICES}
    if unknown := set(args.only or []) - known | set(chunks) - known:
        parser.error(f"unknown matrix versions: {sorted(unknown)}")

    train = pl.scan_parquet(args.train)
    test = pl.scan_parquet(args.test)
    inputs = {
        "events": (pl.concat([train, test]),),
        "test": (test,),
        "train_test": (train, test),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = time.perf_counter()
    for m in MATRICES:
        if args.only and m.ver not in args.only:
            continue
        path = out / f"{m.name}.parquet"
        if path.exists() and not args.overwrite:
            logger.info(f"v{m.ver}: {path} exists, skipping (use --overwrite)")
            continue
        engine = args.engine or m.engine
        session_chunk = chunks.get(m.ver, m.session_chunk)
        logger.info(f"v{m.ver}: engine={engine} session_chunk={session_chunk}")
        start = time.perf_counter()
        make = getattr(covisits, f"make_covisits_{m.ver}")
        result = make(*inputs[m.data], session_chunk=session_chunk, engine=make_engine(engine))
        tmp = path.with_suffix(".parquet.tmp")
        result.write_parquet(tmp)
        tmp.rename(path)
        logger.info(f"v{m.ver}: {result.height:,} rows -> {path} in {time.perf_counter() - start:.1f}s")
        del result
    logger.info(f"done in {time.perf_counter() - total:.1f}s")


if __name__ == "__main__":
    main()
