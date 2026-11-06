#!/usr/bin/env bash
# Covisitation matrices + candidates on the local-validation split, on a cloud GPU pod.
#
# Needs an NVIDIA driver with CUDA 13 (e.g. an A100; not a V100) and HF_TOKEN with write
# access to the HF dataset (for the uploads). Keep WORK on the pod's volume (/workspace)
# so outputs survive a stop.
#
# As the pod's start command (preferred: no SSH needed, the pod stops itself at the end).
# Set the env var HF_TOKEN={{ RUNPOD_SECRET_hf_token }} (a Runpod secret) and the start
# command to:
#   bash -c '(/start.sh >/workspace/start.log 2>&1 &); mkdir -p /workspace/logs; curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-OTTO-Comp/candidate_generation/train/candidates/run_a100.sh | bash 2>&1 | tee -a /workspace/logs/run.log; sleep infinity'
# /start.sh (in Runpod's images) keeps SSH and Jupyter available for debugging; `sleep
# infinity` stops a failed script from restart-looping the container if the self-stop
# below cannot run.
#
# Or by hand over SSH (inside tmux, so a dropped connection does not kill it):
#   export HF_TOKEN=...
#   curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-OTTO-Comp/candidate_generation/train/candidates/run_a100.sh | bash
# or, from a checkout: bash train/candidates/run_a100.sh
#
# Steps (each skipped if its output exists, so a rerun resumes):
#   1. clone/update the repo, uv sync
#   2. download local_valid/ (valid_train, valid_test, valid_test_labels) from HF
#   3. save_covisits.py on valid_train + valid_test  -> data/covisit_matrices_valid
#   4. save_candidates.py on valid_test, with recall -> data/candidate_scores_valid
#   5. upload both output folders and the logs to HF
# On exit, success or failure: upload the logs (if UPLOAD=1) and stop the pod (if
# STOP_POD=1 and on Runpod). A stopped pod keeps /workspace and bills only its storage.
set -euo pipefail

REPO=${REPO:-https://github.com/seanv507/Kaggle-OTTO-Comp.git}
BRANCH=${BRANCH:-candidate_generation}
HF_DATASET=${HF_DATASET:-seanv507/kaggle-otto-multi-objective}
WORK=${WORK:-/workspace}
UPLOAD=${UPLOAD:-1}  # 0: keep the outputs on the pod only
STOP_POD=${STOP_POD:-1}  # 0: leave the pod running at the end (e.g. to inspect it)

export HF_XET_HIGH_PERFORMANCE=1
LOGS=$WORK/logs
mkdir -p "$LOGS"

hf() { uvx --from huggingface_hub hf "$@"; }

stop_pod() {
    # runpodctl and RUNPOD_POD_ID are provided inside Runpod pods
    command -v runpodctl >/dev/null || { echo "runpodctl not found: stop the pod yourself"; return; }
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID" \
        || echo "could not stop pod $RUNPOD_POD_ID: stop it yourself"
}

on_exit() {
    local status=$?
    echo "run_a100.sh exited with status $status at $(date -u +%FT%TZ)" | tee -a "$LOGS/status.txt"
    if [ "$UPLOAD" = 1 ]; then
        hf upload "$HF_DATASET" "$LOGS" logs/valid_run --repo-type dataset \
            --commit-message "Logs of the valid run (exit $status)" || echo "log upload failed"
    fi
    if [ "$STOP_POD" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "stopping pod $RUNPOD_POD_ID"
        stop_pod
    fi
}
trap on_exit EXIT

# 1. code
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
cd "$WORK"
[ -d Kaggle-OTTO-Comp ] || git clone --branch "$BRANCH" "$REPO"
cd Kaggle-OTTO-Comp
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
uv sync
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
fi
echo "done: recall per run is in $LOGS/candidates_valid.log"
