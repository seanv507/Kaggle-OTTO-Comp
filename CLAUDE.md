# CLAUDE.md

## What this repo is

A replication of the Kaggle [OTTO recommender system](https://www.kaggle.com/competitions/otto-recommender-system/leaderboard) competition. The task: predict the next clicks, carts and orders for each session, 20 aids for each target.

The 3rd-place team (Benny, Chris, Giba and Theo) submitted an ensemble of their individual models. This repo is the user's fork of Chris's part (`seanv507/Kaggle-OTTO-Comp`), and also follows Benny's:

- **Chris Deotte's part** (the main reference): [cdeotte/Kaggle-OTTO-Comp](https://github.com/cdeotte/Kaggle-OTTO-Comp). His original notebooks are in this repo under `data/`, `train/` and `infer/`. A candidate rerank model: about 100 candidates per target from 20 covisitation matrices plus rules, then one XGBoost ranker per target. LB 0.601.
- **Benny Schifferer's part**: local copy at `../Kaggle-Otto-Comp-Benedikt`, the user's fork `seanv507/Kaggle-Otto-Comp-Benedikt`. An XGBoost reranker with Transformers4Rec, GRU and Word2Vec features and a 5-fold week-4 split. LB 0.601.

**Read [docs/solutions.md](docs/solutions.md) before working on a pipeline stage.** It summarises both write-ups: Chris's 20 covisitation matrix definitions, both candidate strategies, features, rankers, Benny's data split and his ablation table.

The main work is **converting their pandas/cuDF notebooks to Polars, using the GPU engine where possible.** The Polars code goes next to the notebook it replaces, following Chris's layout. The code was first written in `seanv507/kaggle-otto-multi-objective` and moved here with its git history.

Chris's pipeline, which runs once on train to fit models and once on all data for the LB:
1. covisitation matrices: `train/covisit_matrices/`, `infer/covisit_matrices_LB/`
2. candidates and scores: `train/candidates/`
3. item and user features: `train/item_user_features/`
4. merge candidates, features and targets: `train/make_parquets/`
5. XGBoost/CatBoost rankers: `train/ranker_models/`

When converting, read the matching reference notebook first. In `covisits.py`, function names map to notebook numbers. For example, `covisits.make_covisits_90` corresponds to `train/covisit_matrices/cpu-90.ipynb`, and `make_covisits_115` to `gpu-115.ipynb`.

## Layout

Chris's notebooks keep their places; the Polars code sits beside them. Tests (`test_*.py`) live next to the module they test.

- `train/covisit_matrices/covisits.py`: covisitation matrices, one `make_covisits_<VER>` per notebook run by `train/covisit_matrices/script.ipynb`, all built through `build_covisits`. The module docstring explains the shared conventions, including which copy of a duplicate pair is kept.
- `train/covisit_matrices/covisits_reference.py`: pandas ports of those notebooks, which the tests compare against.
- `train/covisit_matrices/save_covisits.py`: computes and saves all matrices. Run it from the repo root: `uv run python train/covisit_matrices/save_covisits.py`.
- `train/covisit_matrices/covisitations.ipynb`: exploration and timings, with benchmark results in markdown cells. It runs from its own directory, so data paths are `../../data/...`.
- `data/labels.py`, `data/testset.py`: ground-truth labels and the train/test split (being converted to Polars), the counterpart of `data/make_train_valid.ipynb`.
- `event_types.py` (root): integer codes for the `type` column, shared by all stages.
- `diary.md`: the user's working notes.
- `docs/solutions.md`: summaries of Chris's and Benny's write-ups.
- `data/`: Chris's tracked README skeleton, plus gitignored data files (`*.parquet`, `*.pkl`, `raw/`). Locally the input parquets are symlinks into `../kaggle-otto-multi-objective/data/`. On a cloud machine `data/` itself can't be a symlink, because it is tracked: link the individual files, or pass other paths (e.g. `save_covisits.py --train/--test/--out`).

## Environment

- Managed with `uv`. Run things from the repo root with `uv run python ...` and `uv run pytest`.
- The project is installed in editable mode with hatch `dev-mode-dirs = ["."]`, which puts the repo root on `sys.path`. That is how modules in `train/...` and `data/` import root modules such as `event_types`. Same-directory imports (`import covisits`) work because each script, notebook and test runs with its own directory on the path.
- **Polars is pinned to 1.44.x on purpose.** `cudf-polars-cu13` 26.10 needs `polars>=1.35,<1.45`. With Polars 2.x, uv quietly installs an empty placeholder cudf-polars (0.0.0a0) and the GPU engine stops working. Check RAPIDS support before upgrading Polars.
- Hardware: most heavy work will run on **cloud machines with A100 SXM (40/80 GB) or similar GPUs**. The local machine is small: RTX 2050 with 4 GB VRAM and 16 GB RAM. Use it for development, tests and small samples. The reference solutions ran on 32 GB V100s (Chris: 256 GB RAM).
  - Keep code portable between the two. Don't hard-code chunk sizes or memory limits for either machine; make them parameters.
  - Code must still run locally, even if slowly or on a sample, so it can be developed and tested there.
  - Benchmarks, best chunk sizes and memory limits depend on the machine. Record which machine a timing came from, and re-measure on the cloud GPU before relying on local results.
  - Check `nvidia-smi` to see which machine a session is on.
  - The project uses `cudf-polars-cu13`, so a cloud machine needs an NVIDIA driver that supports CUDA 13. CUDA 13 also drops Volta, so V100s won't work with it.

## Data (`data/*.parquet`)

| file | contents |
|---|---|
| `train.parquet` | 216.7M events, 2022-07-31 22:00 to 2022-08-28 21:59:59 |
| `test.parquet` | 6.9M events, 2022-08-28 22:00 to 2022-09-04 21:59:51 (the Kaggle test set) |
| `train_train.parquet`, `train_val.parquet`, `test_full.parquet` | local train/validation split |

Schema: `session` Int32, `aid` Int32, `ts` Datetime(ms), `type` UInt8. `test_full.parquet` uses Int64 for `session` and `aid`. Every file is sorted by `session`.

- `type` codes: 0 = clicks, 1 = carts, 2 = orders. Use the constants in `event_types.py` (`CLICKS`, `CARTS`, `ORDERS`) and never string literals. The Enum type was removed because the GPU engine can't handle it.
- The reference code uses `ts` in epoch **seconds**; here it is Datetime(ms). Translate constants such as `1659304800` into datetimes (see `start_train` and the related constants in `train/covisit_matrices/covisits.py`) and time gaps into `pl.duration(...)`.
- **Before rewriting a data file**: show the user an example of the conversion first. Then write to `<file>.tmp`, check it row by row against the original (row count, unchanged columns, mapped values, nulls), and only then `mv` it over. There are no backups.

## Writing Polars code here

- Use the lazy API: accept `pl.LazyFrame | pl.DataFrame`, call `.lazy()`, and finish with `.collect(engine=engine)`, where `engine` is a parameter. Prefer idiomatic expressions such as `.over(...)` and `top_k_by` to loops or Python UDFs.
- **GPU-engine limits found so far.** cudf-polars rejects the whole query if any one of these appears:
  - Enum/Categorical columns, including any cast of them.
  - `.abs()` on a Duration. Use `diff.is_between(-d, d)`, or convert to a float first with `diff.dt.total_seconds(fractional=True) / 3600`.
  - Dividing a Duration by a Duration, e.g. `diff / pl.duration(hours=1)`. Use `dt.total_seconds()` as above.
  - `top_k_by(...).over(...)`, and `int_range` / `cum_count` inside `.over()`.
  - Adding or subtracting a literal duration or datetime to a Datetime column, e.g. `pl.col("ts") + pl.duration(hours=2)` or `pl.col("ts") - some_datetime`. Comparing `ts` with a Python `datetime` works, as do `ts - ts_y` between two columns, `dt.hour()` and `dt.epoch("s")`.
- **GPU memory:** `.over(...)` inside a filter can't be split into partitions, so cudf runs it on the whole frame at once. A `rank().over("aid")` filter on the 117M-row 90 matrix ran out of memory on the 4 GB GPU. For per-group top-k use `sort` + `group_by(..., maintain_order=True).head(k)` (see `top_covisit`).
- `engine="gpu"` **quietly falls back to CPU** for unsupported queries. To confirm something really runs on the GPU, use `pl.GPUEngine(raise_on_fail=True)`.
- Chunking by session: `iterate_session_chunks` filters a session-sorted LazyFrame into session ranges. On the local RTX 2050 the GPU is fastest with no chunking. For `make_covisits_90` it took 21s on GPU versus 26 to 32s on streaming, while small chunks made the GPU slower. That trend should hold or strengthen on an A100, but it hasn't been measured there. Benchmark the chunk size for each new matrix rather than copying the reference's chunking.

## Testing

- `uv run pytest`. When converting a function, add a small synthetic test with hand-checked expected output. Parametrize it over `engine` (`"streaming"`, `"in-memory"`, and GPU, skipped when unavailable; see `train/covisit_matrices/test_covisits.py`) and over chunk sizes, so chunked and unchunked runs must agree.
- Synthetic test frames must use the same dtypes as the real data.
- Covisitation matrices also get a comparison test against a pandas port of the reference notebook in `covisits_reference.py` (`test_matches_reference`), run on random sessions.
- For full-data checks, compare GPU and streaming outputs with `polars.testing.assert_frame_equal` after sorting. Put benchmark scripts in the scratchpad, not the repo.
- Known broken: `data/labels.py::test_gt` calls `ground_truth_pl`, which doesn't exist.
