#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/ExtractedFeatures}"
SYN_ROOT="${SYN_ROOT:-/root/autodl-tmp/results/seed_eval_source_syn_knn_5runs}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/results/seed_dan_s2_s14_eval}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-10}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
N_RUNS="${N_RUNS:-1}"

mkdir -p "${OUT_DIR}"

for subj in "$@"; do
  syn_path="${SYN_ROOT}/s${subj}/knn_filtered/generated_anchor_ranked_r0p100.npz"
  log_path="${OUT_DIR}/s${subj}_source_knn_r0p100.log"
  if [[ ! -f "${syn_path}" ]]; then
    echo "[s${subj}] missing synthetic file: ${syn_path}" | tee -a "${OUT_DIR}/missing.log"
    continue
  fi

  echo "================================================================"
  echo "[s${subj}] DAN source-only vs KNN0.1 synthetic"
  echo "log: ${log_path}"
  echo "================================================================"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u eval_cross_subject_adaptation.py \
    --data_root "${DATA_ROOT}" \
    --test_subject "${subj}" \
    --synthetic_path "${syn_path}" \
    --methods source_only synthetic \
    --syn_ratio 0.1 \
    --model dan \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --n_runs "${N_RUNS}" \
    --gpu 0 \
    --patience "${PATIENCE}" \
    --val_interval "${VAL_INTERVAL}" \
    2>&1 | tee "${log_path}"
done
