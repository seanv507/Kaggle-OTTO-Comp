# Status (2026-11-08)

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

- **Code:** branch `ranker_models` (pushed) holds everything. The branches stack:
  `candidate_generation` → `item_user_features` → `make_parquets` → `ranker_models`.
  `main` is not updated yet. Locally it is 12 commits ahead of `origin/main`, unpushed, and 27 commits behind `ranker_models`.
- **HF dataset** `seanv507/kaggle-otto-multi-objective`:

  | folder | contents |
  |---|---|
  | `covisit_matrices_valid/` | the 20 matrices |
  | `item_user_features_valid/` | features and `item_embed_32.npy` |
  | `models_valid/` | rankers (`.ubj`) and out-of-fold predictions |
  | `logs/valid_run/gmr1vifevc2x4e/` | logs, including `summary.txt` |

  **`candidate_scores_valid/` is missing:** its upload failed and left no trace in the logs. The exit trap now uploads the logs again last (commit `30923e3`), so a failure like this will show up. The merged rows were never uploaded (that is the default, `UPLOAD_ROWS=0`).
- **Pod `gmr1vifevc2x4e` (`otto-valid-full`)** is EXITED, so only its 200 GB volume is billing.
  - The decision is to terminate it and not recover the candidates.
  - Terminating it from Claude was blocked by the auto-mode permission check, so terminate it in the Runpod console.
  - If the candidates are needed again, regenerate them from the matrices on HF (about 1.5 h on an A100). The merged rows then take another 11 min.
- **How to launch a run:** see the header of `train/run_a100.sh` and `docs/runpod-launching.md`.
  - The pod's start command fetches the script from the `ranker_models` branch.
  - Progress notifications go to ntfy. The topic is a Runpod secret, and locally it is in `~/.config/ntfy_topic`.
- `errors.log` (untracked) is an old log from the 2026-11-06 pod and can be deleted.

## Next

1. Terminate pod `gmr1vifevc2x4e` (console).
2. Merge `ranker_models` into `main`, which stacks all the branches above.
3. Convert `infer/` (the LB run: covisits on all data, candidates, features, rankers' predictions, the submission).
4. Optional:
   - simplify candidates (step 11 of the candidates plan: one run with a score column per rule);
   - speed up the clicks ranker (3 h 40);
   - CatBoost is not ported.
