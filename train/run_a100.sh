#!/usr/bin/env bash
# The whole train pipeline on the local-validation split, on a cloud GPU pod: covisitation
# matrices, candidates, features, embeddings, merged rows and the three rankers, ending
# with the out-of-fold recall@20 per target and the weighted score (Chris: 0.590).
#
# Needs an NVIDIA driver with CUDA 13 (e.g. an A100; not a V100), HF_TOKEN with write
# access to the HF dataset (for the uploads), >= 150 GB on the volume (merged rows ~60 GB,
# matrices ~10 GB, candidates ~7 GB, the environment) and >= 64 GB RAM (the clicks
# ranker's training sample is ~44 GB; 100 GB is safer). Keep WORK on the pod's volume
# (/workspace) so outputs survive a stop.
#
# As the pod's start command (preferred: no SSH needed, the pod stops itself at the end).
# Set the env vars HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }} and NTFY_TOPIC={{ RUNPOD_SECRET_NTFY_TOPIC }}
# (Runpod secrets) and the start command to:
#   bash -c '(/start.sh >/workspace/start.log 2>&1 &); mkdir -p /workspace/logs; curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-OTTO-Comp/ranker_models/train/run_a100.sh | bash 2>&1 | tee -a /workspace/logs/run.log; sleep infinity'
# /start.sh (in Runpod's images) keeps SSH and Jupyter available for debugging; `sleep
# infinity` stops a failed script from restart-looping the container if the self-stop
# below cannot run. The branch in the URL and BRANCH below must match.
#
# Or by hand over SSH (inside tmux, so a dropped connection does not kill it):
#   export HF_TOKEN=...
#   curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-OTTO-Comp/ranker_models/train/run_a100.sh | bash
# or, from a checkout: bash train/run_a100.sh
#
# Steps (each skips the outputs it already has, so a rerun resumes; the rankers retrain):
#   1. clone/update the repo, uv sync (with PyTorch for CUDA)
#   2. download local_valid/ (valid_train, valid_test, valid_test_labels) from HF
#   3. save_covisits.py on valid_train + valid_test   -> data/covisit_matrices_valid
#   4. save_candidates.py on valid_test, with recall  -> data/candidate_scores_valid
#   5. save_features.py                               -> data/item_user_features_valid
#   6. save_embeddings.py                             -> .../item_embed_32.npy
#   7. save_merged.py                                 -> data/train_with_features_valid
#   8. train_rankers.py, with recall@20 and the score -> data/models_valid
# Notifications (if NTFY_TOPIC is set): ntfy.sh on each completed step (with the candidate
# and ranker recalls), at the end, and with high priority on failure.
# Uploads to HF (if UPLOAD=1): the matrices after step 3; on exit, success or failure, the
# logs, candidates, features and models (not the merged rows: set UPLOAD_ROWS=1). Then the
# pod stops (if STOP_POD=1 and on Runpod); a stopped pod keeps /workspace and bills only
# its storage.
set -euo pipefail

REPO=${REPO:-https://github.com/seanv507/Kaggle-OTTO-Comp.git}
BRANCH=${BRANCH:-ranker_models}
HF_DATASET=${HF_DATASET:-seanv507/kaggle-otto-multi-objective}
WORK=${WORK:-/workspace}
UPLOAD=${UPLOAD:-1}  # 0: keep the outputs on the pod only
UPLOAD_ROWS=${UPLOAD_ROWS:-0}  # 1: also upload the merged rows (~60 GB)
STOP_POD=${STOP_POD:-1}  # 0: leave the pod running at the end (e.g. to inspect it)
# session ids per save_candidates chunk: unchunked, 564_F_orders needs ~150 GB RAM on the
# 1.8M valid_test sessions (it was OOM-killed on 2026-11-06); 250k keeps it near 20 GB
CHUNK=${CHUNK:-250000}
# session ids per merged part: ~3.3 GB RAM per 20k sessions on a local sample, so ~35 GB
MERGE_CHUNK=${MERGE_CHUNK:-200000}
TORCH_EXTRA=${TORCH_EXTRA:-cu132}  # uv extra for PyTorch (embeddings)
# ntfy.sh notifications on each completed step and at exit: set NTFY_TOPIC (on Runpod,
# NTFY_TOPIC={{ RUNPOD_SECRET_NTFY_TOPIC }}); unset, nothing is sent
NTFY_SERVER=${NTFY_SERVER:-https://ntfy.sh}

export HF_XET_HIGH_PERFORMANCE=1
LOGS=$WORK/logs
mkdir -p "$LOGS"
OUT=$WORK/Kaggle-OTTO-Comp/data

hf() { uvx --from huggingface_hub hf "$@"; }
run() { uv run --extra "$TORCH_EXTRA" python "$@"; }

upload() {  # upload <local dir> <path in the dataset> <message>
    [ -e "$1" ] || return 0
    hf upload "$HF_DATASET" "$1" "$2" --repo-type dataset --commit-message "$3" || echo "upload of $1 failed"
}

notify() {  # notify <message> [priority]
    [ -n "${NTFY_TOPIC:-}" ] || return 0
    curl -fsS -m 15 -o /dev/null -H "Title: OTTO valid run${RUNPOD_POD_ID:+ ($RUNPOD_POD_ID)}" \
        -H "Priority: ${2:-default}" -d "$1" "$NTFY_SERVER/$NTFY_TOPIC" || echo "ntfy failed"
}

stop_pod() {
    # runpodctl and RUNPOD_POD_ID are provided inside Runpod pods
    command -v runpodctl >/dev/null || { echo "runpodctl not found: stop the pod yourself"; return; }
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID" \
        || echo "could not stop pod $RUNPOD_POD_ID: stop it yourself"
}

on_exit() {
    local status=$?
    echo "run_a100.sh exited with status $status at $(date -u +%FT%TZ)" | tee -a "$LOGS/status.txt"
    if [ "$status" = 0 ]; then
        notify "Finished: $(cat "$LOGS/summary.txt" 2>/dev/null || echo 'no summary')"
    else
        notify "FAILED (exit $status) in step '${STEP:-?}' after $(( (SECONDS - STEP_T0) / 60 )) min. Last lines:
$(tail -n 5 "$LOGS/run.log" 2>/dev/null)" high
    fi
    if [ "$UPLOAD" = 1 ]; then
        # what exists so far (all of it on success); hf upload skips unchanged files
        upload "$LOGS" logs/valid_run "Logs of the valid run (exit $status)"
        upload "$OUT/candidate_scores_valid" candidate_scores_valid "Candidates on the local-validation split (exit $status)"
        upload "$OUT/item_user_features_valid" item_user_features_valid "Item/user features and embeddings on the local-validation split (exit $status)"
        upload "$OUT/models_valid" models_valid "Rankers on the local-validation split (exit $status)"
        if [ "$UPLOAD_ROWS" = 1 ]; then
            upload "$OUT/train_with_features_valid" train_with_features_valid "Merged rows on the local-validation split (exit $status)"
        fi
    fi
    if [ "$STOP_POD" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        echo "stopping pod $RUNPOD_POD_ID"
        stop_pod
    fi
}
trap on_exit EXIT

# step <name>: log the start of a step and notify that the previous one completed (with its
# duration and DETAIL, if a step set it)
STEP="" STEP_T0=$SECONDS DETAIL=""
step() {
    if [ -n "$STEP" ]; then
        notify "Done: $STEP in $(( (SECONDS - STEP_T0) / 60 )) min${DETAIL:+
$DETAIL}"
    fi
    STEP=$* STEP_T0=$SECONDS DETAIL=""
    echo "=== $* ($(date -u +%T))" | tee -a "$LOGS/steps.txt"
}

# 1. code
step "1. code"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
cd "$WORK"
[ -d Kaggle-OTTO-Comp ] || git clone --branch "$BRANCH" "$REPO"
cd Kaggle-OTTO-Comp
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
uv sync --extra "$TORCH_EXTRA"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$LOGS/gpu.txt"
git rev-parse HEAD | tee "$LOGS/commit.txt"
# Polars sizes its thread pool from the host's cores; in a container with a CPU quota that
# oversubscribes (a suspect for the pod's slow candidate runs), so cap it at the quota.
# Runpod's A100 hosts use cgroup v1 (2026-11-07: cfs quota 30.1 CPUs, nproc 252).
cpu_quota() {  # prints the quota in whole CPUs (rounded up), or nothing if unlimited
    local quota period
    if [ -r /sys/fs/cgroup/cpu.max ]; then  # cgroup v2
        read -r quota period < /sys/fs/cgroup/cpu.max
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then  # cgroup v1
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    fi
    if [ -n "${quota:-}" ] && [ "$quota" != max ] && [ "$quota" -gt 0 ]; then
        echo $(( (quota + period - 1) / period ))
    fi
}
if [ -z "${POLARS_MAX_THREADS:-}" ] && [ -n "$(cpu_quota)" ]; then
    POLARS_MAX_THREADS=$(cpu_quota)
    export POLARS_MAX_THREADS
fi
mem_limit=""  # cgroup v2 or v1; only one of the files exists (a failing cat would trip set -e)
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$f" ]; then mem_limit=$(cat "$f"); break; fi
done
{ echo "nproc $(nproc), CPU quota $(cpu_quota || true), POLARS_MAX_THREADS ${POLARS_MAX_THREADS:-unset}," \
       "memory limit ${mem_limit:-n/a} bytes (free below shows the host)"; free -g; df -h "$WORK"; } | tee "$LOGS/machine.txt"

# 2. inputs (data/ is tracked, so link the files rather than the directory)
step "2. inputs"
hf download "$HF_DATASET" --repo-type dataset --include "local_valid/*" --local-dir "$WORK/hf"
for f in valid_train valid_test valid_test_labels; do
    ln -sf "$WORK/hf/local_valid/$f.parquet" "data/$f.parquet"
done

# 3. covisitation matrices (existing ones are skipped unless --overwrite)
step "3. covisitation matrices"
run train/covisit_matrices/save_covisits.py \
    --train data/valid_train.parquet --test data/valid_test.parquet \
    --out data/covisit_matrices_valid 2>&1 | tee -a "$LOGS/covisits_valid.log"
# upload now, so a later failure does not lose them (unchanged files are skipped)
if [ "$UPLOAD" = 1 ]; then
    upload data/covisit_matrices_valid covisit_matrices_valid \
        "Covisitation matrices on the local-validation split ($(cat "$LOGS/commit.txt"))"
fi

# 4. candidates + recall (existing runs are skipped, recall is still printed)
step "4. candidates"
run train/candidates/save_candidates.py \
    --events data/valid_test.parquet --covisits data/covisit_matrices_valid \
    --out data/candidate_scores_valid --labels data/valid_test_labels.parquet \
    --chunk "$CHUNK" 2>&1 | tee -a "$LOGS/candidates_valid.log"
DETAIL=$(grep -E "(709_[a-z]+|564_F_[a-z]+): recall" "$LOGS/candidates_valid.log" | sed 's/.* - //' | sort -u || true)

# 5. item and user features (existing files are skipped)
step "5. features"
run train/item_user_features/save_features.py \
    --train data/valid_train.parquet --test data/valid_test.parquet \
    --out data/item_user_features_valid 2>&1 | tee -a "$LOGS/features_valid.log"

# 6. item embeddings (20 epochs on the GPU)
step "6. embeddings"
if [ ! -e data/item_user_features_valid/item_embed_32.npy ]; then
    run train/item_user_features/save_embeddings.py \
        --train data/valid_train.parquet --test data/valid_test.parquet \
        --out data/item_user_features_valid 2>&1 | tee -a "$LOGS/embeddings_valid.log"
fi

# 7. merged rows (existing parts are skipped)
step "7. merged rows"
run train/make_parquets/save_merged.py \
    --events data/valid_test.parquet --labels data/valid_test_labels.parquet \
    --candidates data/candidate_scores_valid --features data/item_user_features_valid \
    --out data/train_with_features_valid --chunk "$MERGE_CHUNK" 2>&1 | tee -a "$LOGS/merged_valid.log"

# 8. rankers, out-of-fold recall@20 and the score
step "8. rankers"
run train/ranker_models/train_rankers.py \
    --rows data/train_with_features_valid --labels data/valid_test_labels.parquet \
    --out data/models_valid 2>&1 | tee -a "$LOGS/rankers_valid.log"

{ grep -hE "recall@20|score " "$LOGS/rankers_valid.log" || true; } | sed 's/.* - //' | tee "$LOGS/summary.txt"
DETAIL=$(cat "$LOGS/summary.txt")
step "done"
