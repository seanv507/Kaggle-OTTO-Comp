# Status (2026-11-09)

Where the Polars conversion stands, and what to do next.

## Done

All five stages of Chris's train pipeline are converted, tested against pandas ports of his notebooks, and have run end to end on the full local-validation split. That run used pod `gmr1vifevc2x4e`, an A100 SXM 80 GB in US-MD-1, on 2026-11-07. It took about 7.75 h and cost about $14.

| | ours | Chris |
|---|---|---|
| 709 candidates, recall@100: clicks / carts / orders | 0.67449 / 0.52620 / 0.71947 | 0.67459 / 0.52667 / 0.71981 |
| rankers, recall@20: clicks | 0.55845 | 0.55835 |
| carts | 0.44009 | 0.440095 |
| orders | 0.66858 | 0.668165 |
| **score** | **0.58902** | 0.5888 (0.590 with Giba's features) |

| step | time |
|---|---|
| covisits | 9 min |
| candidates | 92 min |
| merge | 11 min |
| rankers: clicks | 3 h 40 |
| rankers: carts | 68 min |
| rankers: orders | 30 min |

## Where things are

- **Code:** `main` was fast-forwarded to `ranker_models` (2a511a8) and pushed on 2026-11-09, so the two are identical. The `infer/` conversion is on branch `infer` (from `main`), which the pod's start command now fetches.
- **HF dataset** `seanv507/kaggle-otto-multi-objective`:

  | folder | contents |
  |---|---|
  | `covisit_matrices_valid/` | the 20 matrices |
  | `item_user_features_valid/` | features and `item_embed_32.npy` |
  | `models_valid/` | rankers (`.ubj`) and out-of-fold predictions |
  | `logs/valid_run/gmr1vifevc2x4e/` | logs, including `summary.txt` |

  **`candidate_scores_valid/` is missing:** its upload failed and left no trace in the logs. The exit trap now uploads the logs again last (commit `30923e3`), so a failure like this will show up. The merged rows were never uploaded (that is the default, `UPLOAD_ROWS=0`).
- **Pod `gmr1vifevc2x4e`** has been terminated; `candidate_scores_valid/` is lost. If it is needed again, regenerate it from the matrices on HF (about 1.5 h on an A100), then the merged rows (11 min).
- **How to launch a run:** see the header of `train/run_a100.sh` and `docs/runpod-launching.md`.
  - The pod's start command fetches the script from the `infer` branch (the URL in the start command and `BRANCH` in `run_a100.sh` must match).
  - Progress notifications go to ntfy. The topic is a Runpod secret, and locally it is in `~/.config/ntfy_topic`.

## LB inference (`infer/`)

Converted on 2026-11-09; see [infer.md](infer.md). The train modules run unchanged on
`lb_train` / `lb_test` (train weeks 2-4 and the test week moved back 7 days), and
`infer/inference_LB/predict.py` scores the rows with `models_valid` and writes the
submission. `SPLIT=lb bash train/run_a100.sh` runs it all and uploads `submission_lb/` to HF.

A local smoke run on a sample (1% of `lb_train` sessions, 2% of `lb_test`, 33,436
sessions) went through every stage and wrote a well-formed submission. That checks that
the stages run on the LB data and that the LB rows have every feature the rankers use; it
says nothing about the score.

**Prediction speed.** The fold models have about 3,100 trees each (clicks fold 0: best
iteration 3097). On the local machine (RTX 2050, 4 GB), one fold on 400k rows takes 2.5 s
on the CPU and 25 s on the GPU, so the GPU is 10x slower there. The full LB has about 1.67M
sessions, roughly 167M rows per target. At the local CPU rate that is about 1.5 h for
clicks, and about twice that each for carts and orders, which also compute `click_oof`. The
A100 GPU has not been measured. `predict.py` logs the time per part, so check the first
parts of the pod run and pass `--device cpu` if the GPU is slower there too.

## Next

1. Push branch `infer` (the pod fetches the script from GitHub).
2. Run `SPLIT=lb` on an A100 pod. Without ranker training it should take about 2-3 h plus prediction.
3. Submit `submission_lb.csv` to Kaggle by hand and compare with Chris's LB 0.601 (docs/solutions.md; his CV 0.601 includes features we lack, ours is 0.58902).
4. Optional:
   - simplify candidates (step 11 of the candidates plan: one run with a score column per rule);
   - speed up the clicks ranker (3 h 40);
   - CatBoost is not ported.
