"""Generate and save candidate lists and scores, following Chris Deotte's
train/candidates/script.ipynb (the model-564/612/614/620/709 notebooks).

Each run in `candidates.RUNS` is saved as <out>/<run>.parquet with columns
session, pos, aid, score: ITEM_CT rows per session in the notebook's order, where score
is what the run's -wgt notebook wrote (-1: history aid, -2: popular-item padding). This
one file replaces the notebook's CSV pair submission_v<VER>...csv / ..._wgt...csv. The
carts candidates of runs that wrote carts (564-F, 612, 614, 709) are their orders list.

Usage:
    python train/candidates/save_candidates.py \\
        --events data/valid_test.parquet --covisits data/covisit_matrices_valid \\
        --labels data/valid_test_labels.parquet
    python train/candidates/save_candidates.py --only 564_1 620 --sessions 20000

620 runs before 709, which reads 620's output (its first 10 aids per session) from --out.
Scores are summed exactly as the notebooks' Counter does, which needs a per-group
cumulative sum the GPU engine lacks, so this runs on the CPU engines.
"""

import argparse
import resource
import time
from pathlib import Path

import polars as pl
from loguru import logger

import candidates
import popular
from event_types import CARTS, CLICKS, ORDERS

# INFER name (key of the `matrices` dict) -> file written by save_covisits.py
MATRIX_FILES = {
    "top20": "top_80_aids_v93",
    "top20b": "top_40_aids_v217",
    "top20c": "top_40_aids_v220",
    "top20d": "top_40_aids_v226",
    "top20e": "top_40_aids_v232",
    "top20f": "top_40_aids_v235",
    "top20orders": "top_40_orders_carts_v95",
    "top20buy2buy": "top_40_buy2buy_v90",
    "top20buy2buy2": "top_40_buy2buy_v99",
    "top20test": "top_40_aids_v116",
    "top20test2": "top_40_aids_v115",
    "top20buy": "top_40_aids_v239",
    "top20new": "top_40_aids_v700",
    "top20new2": "top_40_aids_v701",
    "day": "top_40_aids_v155_d0",
    "day2": "top_80_aids_v157_d0",
    "less": "top_40_aids_v167",
    "more": "top_40_aids_v168",
    "less2": "top_40_aids_v165",
    "more2": "top_40_aids_v166",
}

# runs whose targets are scored against more than one label type
LABEL_TYPES = {"clicks": [CLICKS], "orders": [CARTS, ORDERS]}
TYPE_NAMES = {CLICKS: "clicks", CARTS: "carts", ORDERS: "orders"}


def run_rules(run: str) -> list[candidates.Rule]:
    family, target, include, _ = candidates.RUNS[run]
    hist, cov = candidates.RULES[family][target]
    rules = hist + cov
    return rules if include is None else [r for r in rules if r.include in include]


class Matrices:
    """Ranked covisitation matrices, loaded on first use and kept for later runs."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.cache: dict[tuple[str, int | None], pl.DataFrame] = {}

    def for_run(self, run: str) -> dict[str, pl.DataFrame]:
        rules = run_rules(run)
        out = {}
        for name in sorted({r.matrix for r in rules if not r.constant}):
            depth = candidates.max_rank(rules, name)
            # reuse a deeper copy if one is loaded
            hit = next(
                (m for (n, d), m in self.cache.items() if n == name and (d is None or (depth is not None and d >= depth))),
                None,
            )
            if hit is None:
                path = self.directory / f"{MATRIX_FILES[name]}.parquet"
                t = time.perf_counter()
                hit = candidates.ranked_matrix(pl.scan_parquet(path), depth)
                self.cache[name, depth] = hit
                logger.info(f"loaded {path.name} (rank < {depth}): {hit.height:,} rows in {time.perf_counter() - t:.1f}s")
            out[name] = hit
        return out


def session_chunks(events: pl.LazyFrame, chunk: int | None):
    if chunk is None:
        yield events
        return
    lo, hi = events.select(lo=pl.col("session").min(), hi=pl.col("session").max()).collect().row(0)
    for start in range(lo, hi + 1, chunk):
        yield events.filter(pl.col("session").is_between(start, start + chunk, closed="left"))


def recall(cands: pl.DataFrame, labels: pl.LazyFrame, label_type: int) -> float:
    """OTTO recall: sum over sessions of |ground truth & candidates| / min(len(gt), 20),
    over the labelled sessions that have candidates (as the notebooks' otto_metric)."""
    gt = labels.filter(pl.col("type") == label_type).join(
        cands.lazy().select("session").unique(), on="session", how="semi"
    )
    hits = (
        gt.select("session", aid=pl.col("ground_truth")).explode("aid", empty_as_null=False)
        .join(cands.lazy().select("session", "aid").unique(), on=["session", "aid"], how="semi")
        .select(pl.len()).collect().item()
    )
    total = gt.select(pl.col("ground_truth").list.len().clip(0, 20).sum()).collect().item()
    return hits / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", default="data/valid_test.parquet", help="sessions to score (session, ts sorted)")
    parser.add_argument("--covisits", default="data/covisit_matrices_valid", help="save_covisits.py output directory")
    parser.add_argument("--out", default="data/candidate_scores", help="output directory")
    parser.add_argument("--labels", help="labels parquet (session, type, ground_truth): print recall per run")
    parser.add_argument("--only", nargs="+", metavar="RUN", help=f"runs to make (default: all): {' '.join(candidates.RUNS)}")
    parser.add_argument("--popular", choices=["chris", "events"], default="chris",
                        help="padding lists: Chris's literals (local validation) or computed from --events")
    parser.add_argument("--engine", choices=["in-memory", "streaming"], default="in-memory")
    parser.add_argument("--chunk", type=int,
                        help="session-id range per chunk (default: all at once). Peak memory grows "
                             "~85 KB per session for the largest runs (564_F_orders, 612_orders)")
    parser.add_argument("--sessions", type=int, help="only the first N sessions (smoke tests)")
    parser.add_argument("--overwrite", action="store_true", help="recompute runs that already exist")
    args = parser.parse_args()

    if unknown := set(args.only or []) - set(candidates.RUNS):
        parser.error(f"unknown runs: {sorted(unknown)}")

    events = pl.scan_parquet(args.events).with_columns(pl.col("session", "aid").cast(pl.Int32))
    if args.sessions:
        first = events.select(pl.col("session").unique().sort().head(args.sessions)).collect()
        events = events.join(first.lazy(), on="session", how="semi")
    labels = None
    if args.labels:
        labels = pl.scan_parquet(args.labels)
    if args.popular == "chris":
        pad = popular.CHRIS
    else:
        pad = {t: popular.popular_from_events(events, t) for t in ["clicks", "orders"]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrices = Matrices(Path(args.covisits))

    total = time.perf_counter()
    for run, (family, target, include, branch) in candidates.RUNS.items():
        if args.only and run not in args.only:
            continue
        path = out / f"{run}.parquet"
        if path.exists() and not args.overwrite:
            logger.info(f"{run}: exists, skipping")
        else:
            t = time.perf_counter()
            mats = matrices.for_run(run)
            clicks620 = pl.scan_parquet(out / "620.parquet") if family == "709" else None
            if family == "709" and not (out / "620.parquet").exists():
                raise SystemExit("709 needs 620's output: run 620 first (or keep it in --out)")
            item_ct = 100 if family == "709" else 50
            parts = [
                candidates.suggest_564(
                    chunk, mats, target, include=include, branch=branch, item_ct=item_ct,
                    popular=pad[target], family=family, clicks620=clicks620,
                ).collect(engine=args.engine)
                for chunk in session_chunks(events, args.chunk)
            ]
            pl.concat(parts).write_parquet(path)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB -> GiB
            logger.info(
                f"{run}: {sum(p.height for p in parts):,} rows in {time.perf_counter() - t:.1f}s"
                f" (process peak RSS so far {peak:.1f} GiB)"
            )
        if labels is not None:
            cands = pl.read_parquet(path)
            scores = {TYPE_NAMES[ty]: recall(cands, labels, ty) for ty in LABEL_TYPES[target]}
            logger.info(f"{run}: recall " + ", ".join(f"{k} {v:.5f}" for k, v in scores.items()))
    logger.info(f"total {time.perf_counter() - total:.1f}s")


if __name__ == "__main__":
    main()
