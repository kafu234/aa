#!/usr/bin/env bash
set -euo pipefail

SUBJECTS="$1"
OUT="$2"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/ExtractedFeatures}"
SYN_ROOT="${SYN_ROOT:-/root/autodl-tmp/results/seed_eval_source_syn_knn_5runs}"
SYN_RATIO="${SYN_RATIO:-0.1}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"

mkdir -p "${OUT}"
echo "[$(date -Is)] nsal_dgat worker start: ${SUBJECTS}" | tee "${OUT}/worker_${SUBJECTS// /_}.log"

for SUBJ in ${SUBJECTS}; do
  echo "[$(date -Is)] s${SUBJ} source-only start" | tee -a "${OUT}/worker_${SUBJECTS// /_}.log"
  python eval_de_classifier.py \
    --data_root "${DATA_ROOT}" \
    --split_mode subject \
    --test_subject "${SUBJ}" \
    --real_only \
    --model nsal_dgat \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --gpu "${GPU}" \
    --seed "${SEED}" \
    --n_runs 1 \
    2>&1 | tee "${OUT}/s${SUBJ}_source.log"

  SYN="${SYN_ROOT}/s${SUBJ}/knn_filtered/generated_anchor_ranked_r0p100.npz"
  echo "[$(date -Is)] s${SUBJ} knn0.1 synthetic start" | tee -a "${OUT}/worker_${SUBJECTS// /_}.log"
  python eval_cross_subject_adaptation.py \
    --dataset seed \
    --data_root "${DATA_ROOT}" \
    --test_subject "${SUBJ}" \
    --synthetic_path "${SYN}" \
    --methods synthetic \
    --syn_ratio "${SYN_RATIO}" \
    --model nsal_dgat \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --gpu "${GPU}" \
    --seed "${SEED}" \
    --n_runs 1 \
    2>&1 | tee "${OUT}/s${SUBJ}_knn_r0p100.log"
done

echo "[$(date -Is)] nsal_dgat worker done: ${SUBJECTS}" | tee -a "${OUT}/worker_${SUBJECTS// /_}.log"
