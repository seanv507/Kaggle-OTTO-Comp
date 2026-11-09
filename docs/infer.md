# LB inference (`infer/`)

How Chris's `infer/` notebooks map onto the Polars code, and where the conversion
deliberately departs from them. Decided 2026-11-09.

## Approach: the train code on the LB split

Chris's LB run is his train pipeline rerun one week later. Train weeks 2–4 replace train
weeks 1–3, and the Kaggle test week replaces the truncated week 4. The rankers trained on
the local-validation split then predict on the LB rows.

The conversion runs **our existing train modules unchanged** on LB inputs that are moved
back in time to line up with the valid split ([infer/lb_data.py](../infer/lb_data.py)):

| file | events | span after the move | as |
|---|---|---|---|
| `lb_train.parquet` | `train.parquet` from 2022-08-07 22:00 (week 1 dropped), ts − 7 days | 2022-07-31 22:00 – 08-21 21:59:59 | `valid_train` |
| `lb_test.parquet` | `test.parquet`, ts − 7 days | 2022-08-21 22:00 – 08-28 21:59:51 | `valid_test` |

All time constants in the train code (`covisits.start_train`, `end_test`, the
`in_last_days` windows, the time weights) then mean the same on both splits. Seven days
keeps day of week and hour of day, which the 167/168/180/181 matrices and some features
use. Session and aid ids are unchanged, so the submission uses the real test session ids.

New code, beyond running the stages with other paths:

| notebook(s) | Polars |
|---|---|
| (the `infer_data/` inputs) | `infer/lb_data.py` |
| `make_parquets_LB/Get-Users*` (4 user parts) | `save_merged.py --chunk` (session parts) |
| `make_parquets_LB/Make-161-A`, `Make-175-A` (candidates without targets) | `merge.make(labels=None)`, `save_merged.py --labels none` |
| `make_parquets_LB/Make-175-G` (`click_oof`, mean of the clicks folds) | `rankers.click_oof(average=True)` |
| `inference_LB/LB-1`, `LB-11`, `LB-21` (mean of 5 folds, top 20) | `infer/inference_LB/predict.py` |
| `inference_LB/Make-LB-Sub-3` (concatenate, sort) | `predict.py` |
| `script.ipynb` drivers | `SPLIT=lb bash train/run_a100.sh` |

The candidates' popular-item padding uses `--popular events`, which counts it on the LB
test week. Chris hard-coded a separate LB list.

## Notebook map

Train counterparts were found by diffing each LB notebook against the train notebooks (at
upstream `2bb8962`) and by the variable each candidate notebook loads it into.

**Covisitation matrices** (LB version → train version):

| LB | train | | LB | train |
|---|---|---|---|---|
| cpu-104 | gpu-93 | | gpu-162 | gpu-155 |
| cpu-23 | gpu-217 | | gpu-163 | gpu-157 |
| cpu-24 | gpu-220 | | gpu-178 | gpu-165 |
| cpu-28 | gpu-226 | | gpu-179 | gpu-166 |
| cpu-130 | gpu-232 | | gpu-180 | gpu-167 |
| cpu-132 | gpu-235 | | gpu-181 | gpu-168 |
| cpu-118 | cpu-95 | | gpu-800 | gpu-700 |
| cpu-16 | cpu-90 | | gpu-801 | gpu-701 |
| cpu-119 | cpu-99 | | | |
| cpu-35 | gpu-116 | | | |
| cpu-34 | gpu-115 | | | |
| cpu-37 | gpu-239 | | | |

**Candidates:** `model-11-<k>[-wgt]-LB` → `model-564-<k>[-wgt]`, `11-B` → `564-F`,
`13` → `612`, `14` → `614`, `620-LB` → `620`, `711[-wgt]-LB` → `709[-wgt]`,
`make-test-with-d` / `make-test-lists` → `make-valid-with-d` / `make-valid-lists`.

**Features:** each `item/user features-<n>-LB` → `item/user-features-<n>`, except
`item features-14b-LB` → `item-features-12`; `embeddings-for-test` → `embeddings-for-train`.

**Merged rows:** `Make-161-*` → `Make-152` (clicks), `Make-175-*` → `Make-172` (orders);
`-Copy1..3` are the user parts 1–3. `Make-175-A` is the train step A without targets,
reading the 711 candidates. B–F match `Make-172/174-C..G`, and `Make-175-G` matches
step H.

## Where Chris's LB run differs from his train run, and what we do

Everything below is in Chris's notebooks. We follow the train code each time, so the
LB features mean what the rankers learnt.

- **Matrix variants.**
  - LB 104 keeps the top 40 where train 93 keeps 80.
  - LB 132 keeps 40 where train 235 keeps 80.
  - LB 28 keeps 30 where train 226 keeps 80.
  - LB 34 drops the aid = aid_y self-pairs that train 115 keeps.
  - LB 35 joins only to carts/orders, where train 116 keeps click pairs with weight 0.
    This only changes the top k when an aid has fewer than 40 nonzero pairs.
  - The LB CPU notebooks break ties by `Counter.most_common` (insertion) order.
- **Truncated matrix lists in the candidates.** The LB candidate notebooks cut each
  matrix list when loading it, e.g. `top_20c` at 55, `top_20f` at 20, `top_20_orders` at
  30, `less`/`more` at 10. Where model 709 reads a list without a slice (`top_20[aid]`,
  `top_20c[aid]`, `top_20e[aid]`, `top_20f[aid]`, ...), the LB lists have other lengths
  than train's.
- **`less`/`less2` and `more`/`more2` swapped.** LB `top_40_less` is v178 (= 165) while
  train's is v167, and `less2` the reverse; the same holds for `more`/`more2`.
- **Week 1 kept in places.** The LB GPU matrices 162–181 and features 12/14b, 21, 22
  and 40 drop week 1 or keep the last 21 days. The CPU matrices and features 4 and 20 use
  all 5 weeks, and the embeddings are trained on all of them.
- **Carts rows.** The LB builds only the orders-based rows (`Make-175`, `USE='orders'`),
  and LB-11 predicts carts on them. In train, the carts ranker was trained on the
  carts-based rows (`Make-174`). We build all three (`152`, `172`, `174`), so carts
  predicts on carts rows.
- **Popular-item padding.** Chris hard-coded a separate list for the LB; we compute it
  from the LB test week.

## Running it

`SPLIT=lb` in `train/run_a100.sh`. It runs every stage on `lb_train` / `lb_test`, writes
the outputs under `data/*_lb`, downloads `models_valid/` from HF and writes
`data/submission_lb.csv`, which is uploaded to HF as `submission_lb/`. Submitting to
Kaggle is done by hand, e.g.
`kaggle competitions submit otto-recommender-system -f submission_lb.csv -m "..."`.
