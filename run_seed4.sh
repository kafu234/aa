#!/bin/bash
# ============================================================
# SEED-IV 完整流程: 预训练 → 逐被试微调 → 生成 → 评估
# ============================================================
# 参数与 SEED 保持一致:
#   classifier_weight = 0.5
# Trial 划分 (遵循 PGCN 等文献的标准协议):
#   SEED:    15 trials → train 0-8  (9个, 60%) / test 9-14  (6个, 40%)
#   SEED-IV: 24 trials → train 0-15 (16个, 67%) / test 16-23 (8个, 33%)
# ============================================================

GPU=0
CONFIG="./Config/seed4_de_gen.yaml"
DATA_ROOT="/root/autodl-tmp/eeg_feature_smooth"
FEAT_ROOT="/root/autodl-tmp/eeg_feature_smooth"

TRAIN_TRIALS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"   # 前 16 个 trial 训练
TEST_TRIALS="16,17,18,19,20,21,22,23"                    # 后 8 个 trial 测试

PRETRAIN_DIR="/root/autodl-tmp/results/seed4_de_pretrain"
FT_BASE="/root/autodl-tmp/results/seed4_de_ft"

# ============================================================
# 第一步: 全被试预训练 (只跑一次, 所有被试的 DE 数据)
# ============================================================
echo "====== [第一步] 全被试预训练 ======"
python train_seed4.py --config ${CONFIG} --gpu ${GPU} \
    --conditional --classifier_weight 0.5 \
    --split_mode trial \
    --train_trials ${TRAIN_TRIALS} --test_trials ${TEST_TRIALS} \
    --results_dir ${PRETRAIN_DIR}

# ============================================================
# 第二步: 逐被试微调 + 生成 + 评估
# ============================================================
echo "====== [第二步] 逐被试微调 + 生成 + 评估 ======"
for SUBJ in $(seq 1 15); do
    echo "===== 被试 ${SUBJ} ====="

    # 微调
    python train_seed4.py --config ${CONFIG} --gpu ${GPU} \
        --conditional --classifier_weight 0.5 \
        --split_mode trial --subject ${SUBJ} \
        --train_trials ${TRAIN_TRIALS} --test_trials ${TEST_TRIALS} \
        --checkpoint ${PRETRAIN_DIR}/checkpoint-best.pt \
        --finetune --max_epochs 2000 \
        --num_samples 5000 \
        --results_dir ${FT_BASE}/s${SUBJ}

    # 找生成文件
    GEN=$(ls ${FT_BASE}/s${SUBJ}/generated_*.npz 2>/dev/null | head -1)
    if [ -z "${GEN}" ]; then
        echo "[WARNING] 被试 ${SUBJ}: 未找到生成文件, 跳过评估"
        continue
    fi

    # 评估
    python eval_de_classifier_seed4.py \
        --data_root ${FEAT_ROOT} \
        --synthetic_path ${GEN} \
        --subject ${SUBJ} --split_mode trial \
        --train_trials ${TRAIN_TRIALS} \
        --test_trials ${TEST_TRIALS} \
        --mode compare --syn_ratio 0.25 --n_runs 3
done

echo "====== 全部完成 ======"