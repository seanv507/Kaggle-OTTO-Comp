#!/usr/bin/env bash
# The whole train pipeline on the local-validation split, on a cloud GPU pod: covisitation
# matrices, candidates, features, embeddings, merged rows and the three rankers, ending
# with the out-of-fold recall@20 per target and the weighted score (Chris: 0.590).
#
# SPLIT=lb runs the LB inference instead (Chris's infer/): the same stages on the LB
# inputs of infer/lb_data.py (train weeks 2-4 + the Kaggle test week, moved back 7 days
# so they line up with the valid split; see docs/infer.md), merged rows without targets,
# then the valid run's rankers (models_valid on HF) predict and write the submission,
# data/submission_lb.csv (uploaded to HF as submission_lb/). Submitting it to Kaggle is
# left to you. Its volume needs are about the same as the valid run's.
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
#   bash -c '(/start.sh >/workspace/start.log 2>&1 &); mkdir -p /workspace/logs; curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-OTTO-Comp/infer/train/run_a100.sh | bash 2>&1 | tee -a /workspace/logs/run.log; sleep infinity'
# /start.sh (in Runpod's images) keeps SSH and Jupyter available for debugging; `sleep
# infinity` stops a failed script from restart-looping the container if the self-stop
# below cannot run. The branch in the URL and BRANCH below must match.
#
# Or by hand over SSH (inside tmux, so a dropped connection does not kill it):
#   export HF_TOKEN=...
#   curl -sSL https://raw.githubusercontent.com/seanv507/Kaggle-OTTO-Comp/infer/train/run_a100.sh | bash
# or, from a checkout: bash train/run_a100.sh
#
# Steps (each skips the outputs it already has, so a rerun resumes; the rankers retrain):
#   1. clone/update the repo, uv sync (with PyTorch for CUDA)
#   2. download local_valid/ (valid_train, valid_test, valid_test_labels) from HF
#      (lb: train.parquet, test.parquet, then infer/lb_data.py -> data/lb_train, lb_test)
#   3. save_covisits.py on valid_train + valid_test   -> data/covisit_matrices_valid
#   4. save_candidates.py on valid_test, with recall  -> data/candidate_scores_valid
#   5. save_features.py                               -> data/item_user_features_valid
#   6. save_embeddings.py                             -> .../item_embed_32.npy
#   7. save_merged.py                                 -> data/train_with_features_valid
#   8. train_rankers.py, with recall@20 and the score -> data/models_valid
#      (lb: infer/inference_LB/predict.py with HF's models_valid -> data/submission_lb.csv)
# (lb: every _valid above is _lb, the inputs are lb_train / lb_test, step 4 pads with the
# LB test week's popular items and reports no recall, step 7 writes infer_with_features_lb)
# Notifications (if NTFY_TOPIC is set): ntfy.sh on each completed step (with the candidate
# and ranker recalls), at the end, and with high priority on failure.
# Uploads to HF (if UPLOAD=1): the matrices after step 3; on exit, success or failure, the
# logs (to logs/valid_run/<pod id>), candidates, features and models (not the merged rows:
# set UPLOAD_ROWS=1). Then the pod stops (if STOP_POD=1 and on Runpod); a stopped pod keeps
# /workspace and bills only its storage, but its GPU may be taken meanwhile.
set -euo pipefail

SPLIT=${SPLIT:-valid}  # valid: train the rankers; lb: the LB inference and submission
case "$SPLIT" in
    valid) TRAIN=valid_train TEST=valid_test ROWS=train_with_features_valid ;;
    lb) TRAIN=lb_train TEST=lb_test ROWS=infer_with_features_lb ;;
    *) echo "SPLIT must be valid or lb, not $SPLIT"; exit 2 ;;
esac
REPO=${REPO:-https://github.com/seanv507/Kaggle-OTTO-Comp.git}
BRANCH=${BRANCH:-infer}
HF_DATASET=${HF_DATASET:-seanv507/kaggle-otto-multi-objective}
WORK=${WORK:-/workspace}
UPLOAD=${UPLOAD:-1}  # 0: keep the outputs on the pod only
UPLOAD_ROWS=${UPLOAD_ROWS:-0}  # 1: also upload the merged rows (~60 GB)
STOP_POD=${STOP_POD:-1}  # 0: leave the pod running at the end (e.g. to inspect it)
# after a failure, wait this long before stopping the pod: a stopped pod's GPU is soon taken
# by another renter (twice on 2026-11-07), so a fix is applied by restarting the running pod
FAIL_GRACE_MIN=${FAIL_GRACE_MIN:-20}
# session ids per save_candidates chunk: unchunked, 564_F_orders needs ~150 GB RAM on the
# 1.8M valid_test sessions (it was OOM-killed on 2026-11-06); 250k keeps it near 20 GB
CHUNK=${CHUNK:-250000}
# session ids per merged part: ~3.3 GB RAM per 20k sessions on a local sample, so ~35 GB
MERGE_CHUNK=${MERGE_CHUNK:-200000}
TORCH_EXTRA=${TORCH_EXTRA:-cu132}  # uv extra for PyTorch (embeddings)
# ntfy.sh notifications on each completed step and at exit: set NTFY_TOPIC (on Runpod,
# NTFY_TOPIC={{ RUNPOD_SECRET_NTFY_TOPIC }}); unset, nothing is sent
NTFY_SERVER=${NTFY_SERVER:-https://ntfy.sh}
# Runpod's /workspace volume is a network filesystem (MooseFS), where uv's tens of thousands
# of small writes can fail ("Stale file handle", 2026-11-07); the venv need not survive a
# stop, so keep it and uv's cache on the container disk
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/root/otto-venv}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
# HF folder for this run's logs (logs/<split>_run/<id>): one per pod, so runs on different pods do not mix
RUN_ID=${RUN_ID:-${RUNPOD_POD_ID:-local}}

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
    curl -fsS -m 15 -o /dev/null -H "Title: OTTO $SPLIT run${RUNPOD_POD_ID:+ ($RUNPOD_POD_ID)}" \
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
        notify "FAILED (exit $status) in step '${STEP:-?}' after $(( (SECONDS - STEP_T0) / 60 )) min.$(
            [ "$STOP_POD" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ] && echo " The pod stops in $FAIL_GRACE_MIN min unless restarted.")
Last lines:
$(tail -n 5 "$LOGS/run.log" 2>/dev/null)" high
    fi
    if [ "$UPLOAD" = 1 ]; then
        # what exists so far (all of it on success); hf upload skips unchanged files
        upload "$LOGS" "logs/${SPLIT}_run/$RUN_ID" "Logs of the $SPLIT run $RUN_ID (exit $status)"
        upload "$OUT/candidate_scores_$SPLIT" "candidate_scores_$SPLIT" "Candidates on the $SPLIT split (exit $status)"
        upload "$OUT/item_user_features_$SPLIT" "item_user_features_$SPLIT" "Item/user features and embeddings on the $SPLIT split (exit $status)"
        if [ "$SPLIT" = valid ]; then
            upload "$OUT/models_valid" models_valid "Rankers on the local-validation split (exit $status)"
        elif [ -e "$OUT/submission_lb.csv" ]; then
            mkdir -p "$WORK/submission_lb" && cp "$OUT/submission_lb.csv" "$WORK/submission_lb/submission_lb.csv"
            upload "$WORK/submission_lb" submission_lb "LB submission (exit $status, commit $(cat "$LOGS/commit.txt" 2>/dev/null))"
        fi
        if [ "$UPLOAD_ROWS" = 1 ]; then
            upload "$OUT/$ROWS" "$ROWS" "Merged rows on the $SPLIT split (exit $status)"
        fi
        # again, so run.log on HF records how the uploads above went (a failed candidates
        # upload on 2026-11-07 left no trace: the logs had gone up first)
        upload "$LOGS" "logs/${SPLIT}_run/$RUN_ID" "Logs of the $SPLIT run $RUN_ID after the uploads (exit $status)"
    fi
    if [ "$STOP_POD" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        if [ "$status" != 0 ] && [ "$FAIL_GRACE_MIN" -gt 0 ]; then
            echo "waiting $FAIL_GRACE_MIN min before stopping (restart the pod to retry on its GPU)"
            sleep $(( FAIL_GRACE_MIN * 60 ))
        fi
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
for attempt in 1 2 3; do  # retried: see UV_PROJECT_ENVIRONMENT above
    if uv sync --extra "$TORCH_EXTRA"; then break; fi
    if [ "$attempt" = 3 ]; then exit 1; fi
    echo "uv sync failed (attempt $attempt), retrying"
    sleep 10
done
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
# the same for OpenMP / BLAS pools (PyTorch, numpy, XGBoost): PyTorch ran 128 threads on a
# 14-CPU quota and starved the GPU (391 s per embedding epoch at 1% GPU use, 2026-11-07)
if [ -n "${POLARS_MAX_THREADS:-}" ]; then
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$POLARS_MAX_THREADS}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$POLARS_MAX_THREADS}
fi
mem_limit=""  # cgroup v2 or v1; only one of the files exists (a failing cat would trip set -e)
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$f" ]; then mem_limit=$(cat "$f"); break; fi
done
{ echo "nproc $(nproc), CPU quota $(cpu_quota || true), POLARS_MAX_THREADS ${POLARS_MAX_THREADS:-unset}, OMP_NUM_THREADS ${OMP_NUM_THREADS:-unset}," \
       "memory limit ${mem_limit:-n/a} bytes (free below shows the host)"; free -g; df -h "$WORK"; } | tee "$LOGS/machine.txt"

# 2. inputs (data/ is tracked, so link the files rather than the directory)
step "2. inputs"
if [ "$SPLIT" = valid ]; then
    hf download "$HF_DATASET" --repo-type dataset --include "local_valid/*" --local-dir "$WORK/hf"
    for f in valid_train valid_test valid_test_labels; do
        ln -sf "$WORK/hf/local_valid/$f.parquet" "data/$f.parquet"
    done
else
    hf download "$HF_DATASET" train.parquet test.parquet --repo-type dataset --local-dir "$WORK/hf"
    if [ ! -e "$WORK/hf/lb_test.parquet" ]; then
        run infer/lb_data.py --train "$WORK/hf/train.parquet" --test "$WORK/hf/test.parquet" \
            --out "$WORK/hf" 2>&1 | tee -a "$LOGS/lb_data.log"
    fi
    for f in lb_train lb_test; do
        ln -sf "$WORK/hf/$f.parquet" "data/$f.parquet"
    done
fi

# 3. covisitation matrices (existing ones are skipped unless --overwrite)
step "3. covisitation matrices"
run train/covisit_matrices/save_covisits.py \
    --train "data/$TRAIN.parquet" --test "data/$TEST.parquet" \
    --out "data/covisit_matrices_$SPLIT" 2>&1 | tee -a "$LOGS/covisits_$SPLIT.log"
# upload now, so a later failure does not lose them (unchanged files are skipped)
if [ "$UPLOAD" = 1 ]; then
    upload "data/covisit_matrices_$SPLIT" "covisit_matrices_$SPLIT" \
        "Covisitation matrices on the $SPLIT split ($(cat "$LOGS/commit.txt"))"
fi

# 4. candidates + recall (existing runs are skipped, recall is still printed)
step "4. candidates"
if [ "$SPLIT" = valid ]; then
    CAND_ARGS=(--labels data/valid_test_labels.parquet)  # Chris's popular lists (of valid_test)
else
    CAND_ARGS=(--popular events)  # the LB test week's popular items; no labels
fi
run train/candidates/save_candidates.py \
    --events "data/$TEST.parquet" --covisits "data/covisit_matrices_$SPLIT" \
    --out "data/candidate_scores_$SPLIT" "${CAND_ARGS[@]}" \
    --chunk "$CHUNK" 2>&1 | tee -a "$LOGS/candidates_$SPLIT.log"
DETAIL=$(grep -E "(709_[a-z]+|564_F_[a-z]+): recall" "$LOGS/candidates_$SPLIT.log" | sed 's/.* - //' | sort -u || true)

# 5. item and user features (existing files are skipped)
step "5. features"
run train/item_user_features/save_features.py \
    --train "data/$TRAIN.parquet" --test "data/$TEST.parquet" \
    --out "data/item_user_features_$SPLIT" 2>&1 | tee -a "$LOGS/features_$SPLIT.log"

# 6. item embeddings (20 epochs on the GPU)
step "6. embeddings"
if [ ! -e "data/item_user_features_$SPLIT/item_embed_32.npy" ]; then
    run train/item_user_features/save_embeddings.py \
        --train "data/$TRAIN.parquet" --test "data/$TEST.parquet" \
        --out "data/item_user_features_$SPLIT" 2>&1 | tee -a "$LOGS/embeddings_$SPLIT.log"
fi

# 7. merged rows (existing parts are skipped)
step "7. merged rows"
run train/make_parquets/save_merged.py \
    --events "data/$TEST.parquet" --labels "$([ "$SPLIT" = valid ] && echo data/valid_test_labels.parquet || echo none)" \
    --candidates "data/candidate_scores_$SPLIT" --features "data/item_user_features_$SPLIT" \
    --out "data/$ROWS" --chunk "$MERGE_CHUNK" 2>&1 | tee -a "$LOGS/merged_$SPLIT.log"

if [ "$SPLIT" = valid ]; then
    # 8. rankers, out-of-fold recall@20 and the score
    step "8. rankers"
    run train/ranker_models/train_rankers.py \
        --rows data/train_with_features_valid --labels data/valid_test_labels.parquet \
        --out data/models_valid 2>&1 | tee -a "$LOGS/rankers_valid.log"
    { grep -hE "recall@20|score " "$LOGS/rankers_valid.log" || true; } | sed 's/.* - //' | tee "$LOGS/summary.txt"
else
    # 8. the valid run's rankers predict the LB rows; the submission
    step "8. predictions"
    hf download "$HF_DATASET" --repo-type dataset --include "models_valid/*" --local-dir "$WORK/hf"
    run infer/inference_LB/predict.py --rows "data/$ROWS" --models "$WORK/hf/models_valid" \
        --out data/submission_lb.csv 2>&1 | tee -a "$LOGS/predict_lb.log"
    { grep -hE "rows .* sessions" "$LOGS/predict_lb.log" || true; } | sed 's/.* - //' | tee "$LOGS/summary.txt"
fi
DETAIL=$(cat "$LOGS/summary.txt")
step "done"
