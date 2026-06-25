#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd /root/FlowTS-main/FMTS

export DATASET=seed
export GPU=0
export DATA_ROOT=/root/autodl-tmp/ExtractedFeatures

# 这里填你前面生成数据的结果目录
export GEN_ROOT=/root/autodl-tmp/results/seed_1_15_no_eval

# 这里是本次 source-only vs source+syn 对比实验输出目录
export EVAL_ROOT=/root/autodl-tmp/results/seed_eval_source_syn_knn_5runs

export SUBJECTS="10 11 12 13 14 "
export RATIOS="0.10 0.15 0.20"

# 每个设置跑 5 次
export N_RUNS=5

# 下游分类器训练参数
export EVAL_EPOCHS=200
export EVAL_BATCH_SIZE=4096
export EVAL_PATIENCE=20

# kNN 筛选 synthetic 的邻居数
export KNN_K=11

mkdir -p "${EVAL_ROOT}"

echo "Evaluation started: $(date)"
echo "GEN_ROOT=${GEN_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "SUBJECTS=${SUBJECTS}"
echo "RATIOS=${RATIOS}"
echo "N_RUNS=${N_RUNS}"
echo "KNN_K=${KNN_K}"

for SUBJ in ${SUBJECTS}; do
    echo "============================================================"
    echo "Target subject ${SUBJ}"
    echo "============================================================"

    SUBJ_DIR="${EVAL_ROOT}/s${SUBJ}"
    mkdir -p "${SUBJ_DIR}"

    PSEUDO_FILE="${GEN_ROOT}/s${SUBJ}/target_pseudo.npz"
    ADAPT_DIR="${GEN_ROOT}/s${SUBJ}/target_adapted"
    GEN_FILE=$(find "${ADAPT_DIR}" -maxdepth 1 -name 'generated_*.npz' -print -quit)

    if [ -z "${GEN_FILE}" ]; then
        echo "ERROR: no generated file found in ${ADAPT_DIR}"
        exit 1
    fi

    if [ ! -f "${PSEUDO_FILE}" ]; then
        echo "ERROR: pseudo file not found: ${PSEUDO_FILE}"
        exit 1
    fi

    echo "Synthetic file: ${GEN_FILE}"
    echo "Pseudo file: ${PSEUDO_FILE}"

    # 计算当前 LOSO 设置下的 source 样本数，用于 ratio=source_count*ratio
    SOURCE_COUNT=$(python - <<PY | tail -n 1
from eval_de_classifier import load_de_data
source_x, source_y, target_x, target_y, source_subjects = load_de_data(
    "${DATA_ROOT}",
    42,
    "subject",
    test_subject=${SUBJ},
    return_subjects=True,
)
print(len(source_y))
PY
)

    echo "SOURCE_COUNT=${SOURCE_COUNT}"

    # 先生成 kNN anchor-ranked synthetic 子集：
    # 会一次性生成 0.10 / 0.15 / 0.20 三个比例的 npz
    KNN_DIR="${SUBJ_DIR}/knn_filtered"
    mkdir -p "${KNN_DIR}"

    python filter_synthetic_anchor_neighbors.py \
        --synthetic_path "${GEN_FILE}" \
        --anchor_path "${PSEUDO_FILE}" \
        --output_dir "${KNN_DIR}" \
        --source_count "${SOURCE_COUNT}" \
        --ratios 0.10 0.15 0.20 \
        --neighbors "${KNN_K}" \
        2>&1 | tee "${SUBJ_DIR}/knn_filter.log"

    for RATIO in ${RATIOS}; do
        RATIO_NAME=$(printf "%.3f" "${RATIO}" | sed 's/\./p/g')

        echo "------------------------------------------------------------"
        echo "Subject ${SUBJ}, ratio=${RATIO}, raw synthetic"
        echo "------------------------------------------------------------"

        python eval_cross_subject_adaptation.py \
            --dataset "${DATASET}" \
            --data_root "${DATA_ROOT}" \
            --test_subject "${SUBJ}" \
            --synthetic_path "${GEN_FILE}" \
            --methods source_only synthetic \
            --syn_ratio "${RATIO}" \
            --epochs "${EVAL_EPOCHS}" \
            --n_runs "${N_RUNS}" \
            --batch_size "${EVAL_BATCH_SIZE}" \
            --patience "${EVAL_PATIENCE}" \
            --gpu "${GPU}" \
            --seed 42 \
            2>&1 | tee "${SUBJ_DIR}/raw_syn_r${RATIO_NAME}.log"

        KNN_FILE="${KNN_DIR}/generated_anchor_ranked_r${RATIO_NAME}.npz"

        if [ ! -f "${KNN_FILE}" ]; then
            echo "ERROR: kNN filtered file not found: ${KNN_FILE}"
            exit 1
        fi

        echo "------------------------------------------------------------"
        echo "Subject ${SUBJ}, ratio=${RATIO}, kNN-filtered synthetic"
        echo "------------------------------------------------------------"

        python eval_cross_subject_adaptation.py \
            --dataset "${DATASET}" \
            --data_root "${DATA_ROOT}" \
            --test_subject "${SUBJ}" \
            --synthetic_path "${KNN_FILE}" \
            --methods source_only synthetic \
            --syn_ratio "${RATIO}" \
            --epochs "${EVAL_EPOCHS}" \
            --n_runs "${N_RUNS}" \
            --batch_size "${EVAL_BATCH_SIZE}" \
            --patience "${EVAL_PATIENCE}" \
            --gpu "${GPU}" \
            --seed 42 \
            2>&1 | tee "${SUBJ_DIR}/knn_syn_r${RATIO_NAME}.log"
    done
done

echo "All evaluations finished: $(date)"