#!/bin/bash
# ============================================================
# SEED-IV 分类别独立训练 → 合并 → 诊断 → 评估
# ============================================================
# 每个类别训练一个独立的无条件生成模型，避免条件模式坍缩
# ============================================================

GPU=0
CONFIG="./Config/seed4_de_gen.yaml"
DATA_ROOT="/root/autodl-tmp/eeg_feature_smooth"
FEAT_ROOT="/root/autodl-tmp/eeg_feature_smooth"

TRAIN_TRIALS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
TEST_TRIALS="16,17,18,19,20,21,22,23"

PRETRAIN_DIR="/root/autodl-tmp/results/seed4_per_class_pretrain"
FT_BASE="/root/autodl-tmp/results/seed4_per_class"

PRETRAIN_EPOCHS=5000
FT_EPOCHS=2000
SAMPLES_PER_CLASS=1250

# ============================================================
# 第一步: 每个类别全被试预训练 (共 4 个模型)
# ============================================================
echo "====== [第一步] 分类别全被试预训练 ======"
for LABEL in 1 2 3; do
    LABEL_NAMES=("neutral" "sad" "fear" "happy")
    echo "--- 预训练 class ${LABEL} (${LABEL_NAMES[$LABEL]}) ---"

    python train_seed4.py --config ${CONFIG} --gpu ${GPU} \
        --split_mode trial \
        --train_trials ${TRAIN_TRIALS} --test_trials ${TEST_TRIALS} \
        --target_label ${LABEL} \
        --max_epochs ${PRETRAIN_EPOCHS} \
        --results_dir ${PRETRAIN_DIR}/class${LABEL}
done

# ============================================================
# 第二步: 逐被试微调 + 生成 + 合并 + 诊断 + 评估
# ============================================================
echo "====== [第二步] 逐被试微调 + 生成 + 评估 ======"
for SUBJ in $(seq 1 15); do
    echo ""
    echo "============================================"
    echo "  被试 ${SUBJ}"
    echo "============================================"

    SUBJ_DIR="${FT_BASE}/s${SUBJ}"
    mkdir -p ${SUBJ_DIR}

    # ---- 2a. 每个类别微调 + 生成 ----
    for LABEL in 0 1 2 3; do
        echo "--- 被试 ${SUBJ}, class ${LABEL} 微调+生成 ---"

        python train_seed4.py --config ${CONFIG} --gpu ${GPU} \
            --split_mode trial --subject ${SUBJ} \
            --train_trials ${TRAIN_TRIALS} --test_trials ${TEST_TRIALS} \
            --target_label ${LABEL} \
            --checkpoint ${PRETRAIN_DIR}/class${LABEL}/checkpoint-best.pt \
            --finetune --max_epochs ${FT_EPOCHS} \
            --num_samples ${SAMPLES_PER_CLASS} \
            --results_dir ${SUBJ_DIR}/class${LABEL}
    done

    # ---- 2b. 合并 4 个类别的生成结果 ----
    echo "--- 被试 ${SUBJ}: 合并生成结果 ---"
    python -c "
import numpy as np
all_data, all_labels = [], []
for c in range(4):
    import glob
    files = glob.glob('${SUBJ_DIR}/class' + str(c) + '/generated_*.npz')
    if not files:
        print(f'  WARNING: class {c} 无生成文件')
        continue
    bundle = np.load(files[0])
    data = bundle['data']
    all_data.append(data)
    all_labels.append(np.full(len(data), c, dtype=np.int64))
    print(f'  class {c}: {len(data)} samples')
if all_data:
    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    np.savez('${SUBJ_DIR}/generated_merged.npz', data=data, labels=labels)
    print(f'  合并完成: {data.shape}, labels: {dict(zip(*np.unique(labels, return_counts=True)))}')
else:
    print('  ERROR: 无任何生成文件')
"

    # ---- 2c. 诊断 ----
    MERGED="${SUBJ_DIR}/generated_merged.npz"
    if [ ! -f "${MERGED}" ]; then
        echo "  [WARNING] 被试 ${SUBJ}: 合并文件不存在, 跳过"
        continue
    fi

    echo "--- 被试 ${SUBJ}: 诊断 ---"
    python eval_de_classifier_seed4.py \
        --data_root ${FEAT_ROOT} \
        --synthetic_path ${MERGED} \
        --subject ${SUBJ} --split_mode trial \
        --train_trials ${TRAIN_TRIALS} \
        --test_trials ${TEST_TRIALS} \
        --mode diagnose --model dgcnn --epochs 200 --gpu ${GPU}

    # ---- 2d. 评估 ----
    echo "--- 被试 ${SUBJ}: 对比评估 ---"
    python eval_de_classifier_seed4.py \
        --data_root ${FEAT_ROOT} \
        --synthetic_path ${MERGED} \
        --subject ${SUBJ} --split_mode trial \
        --train_trials ${TRAIN_TRIALS} \
        --test_trials ${TEST_TRIALS} \
        --mode compare --syn_ratio 0.25 --n_runs 3 \
        --model dgcnn --gpu ${GPU}
done

echo ""
echo "============================================"
echo "  全部完成"
echo "============================================"
