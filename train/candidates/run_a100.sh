#!/usr/bin/env bash
# Covisitation matrices + candidates on the local-validation split, on a cloud GPU pod.
#
# Run on the pod (needs an NVIDIA driver with CUDA 13, e.g. an A100):
#   export HF_TOKEN=...            # write access to the HF dataset (for the uploads)
#   curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-OTTO-Comp/candidate_generation/train/candidates/run_a100.sh | bash
# or, from a checkout: bash train/candidates/run_a100.sh
#
# Steps (each skipped if its output exists, so a rerun resumes):
#   1. clone/update the repo, uv sync
#   2. download local_valid/ (valid_train, valid_test, valid_test_labels) from HF
#   3. save_covisits.py on valid_train + valid_test  -> data/covisit_matrices_valid
#   4. save_candidates.py on valid_test, with recall -> data/candidate_scores_valid
#   5. upload both output folders and the logs to HF
set -euo pipefail

REPO=${REPO:-https://github.com/seanv507/Kaggle-OTTO-Comp.git}
BRANCH=${BRANCH:-candidate_generation}
HF_DATASET=${HF_DATASET:-seanv507/kaggle-otto-multi-objective}
WORK=${WORK:-/workspace}
UPLOAD=${UPLOAD:-1}  # 0: keep the outputs on the pod only

export HF_XET_HIGH_PERFORMANCE=1
LOGS=$WORK/logs
mkdir -p "$LOGS"

# 1. code
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
cd "$WORK"
[ -d Kaggle-OTTO-Comp ] || git clone --branch "$BRANCH" "$REPO"
cd Kaggle-OTTO-Comp
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
uv sync
hf() { uvx --from huggingface_hub hf "$@"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$LOGS/gpu.txt"
git rev-parse HEAD | tee "$LOGS/commit.txt"

# 2. inputs (data/ is tracked, so link the files rather than the directory)
hf download "$HF_DATASET" --repo-type dataset --include "local_valid/*" --local-dir "$WORK/hf"
for f in valid_train valid_test valid_test_labels; do
    ln -sf "$WORK/hf/local_valid/$f.parquet" "data/$f.parquet"
done

# 3. covisitation matrices (existing ones are skipped unless --overwrite)
uv run python train/covisit_matrices/save_covisits.py \
    --train data/valid_train.parquet --test data/valid_test.parquet \
    --out data/covisit_matrices_valid 2>&1 | tee -a "$LOGS/covisits_valid.log"

# 4. candidates + recall (existing runs are skipped, recall is still printed)
uv run python train/candidates/save_candidates.py \
    --events data/valid_test.parquet --covisits data/covisit_matrices_valid \
    --out data/candidate_scores_valid --labels data/valid_test_labels.parquet \
    2>&1 | tee -a "$LOGS/candidates_valid.log"

# 5. outputs
if [ "$UPLOAD" = 1 ]; then
    hf upload "$HF_DATASET" data/covisit_matrices_valid covisit_matrices_valid --repo-type dataset \
        --commit-message "Covisitation matrices on the local-validation split ($(cat "$LOGS/commit.txt"))"
    hf upload "$HF_DATASET" data/candidate_scores_valid candidate_scores_valid --repo-type dataset \
        --commit-message "Candidates on the local-validation split ($(cat "$LOGS/commit.txt"))"
    hf upload "$HF_DATASET" "$LOGS" logs/valid_run --repo-type dataset --commit-message "Logs of the valid run"
fi
echo "done: recall per run is in $LOGS/candidates_valid.log"
