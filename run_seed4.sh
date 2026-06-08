#!/bin/bash
# ============================================================
# SEED-IV 完整流程: 预训练 → 逐被试微调 → 生成 → 评估
# ============================================================
# Guidance 强度消融:
#   默认先跑代表被试 2/3/4/8, 验证 noise-aware guidance + margin 是否能改善标签错配
#   先用较短 epoch 快速诊断, 通过后再全量长跑
# Trial 划分 (遵循 PGCN 等文献的标准协议):
#   SEED:    15 trials → train 0-8  (9个, 60%) / test 9-14  (6个, 40%)
#   SEED-IV: 24 trials → train 0-15 (16个, 67%) / test 16-23 (8个, 33%)
# ============================================================

GPU=${GPU:-0}
CONFIG=${CONFIG:-"./Config/seed4_de_gen.yaml"}
CLASSIFIER_WEIGHT=${CLASSIFIER_WEIGHT:-0.1}
CONDITION_MARGIN_WEIGHT=${CONDITION_MARGIN_WEIGHT:-0.05}
CONDITION_MARGIN=${CONDITION_MARGIN:-0.02}
CONDITION_MARGIN_MAX_T=${CONDITION_MARGIN_MAX_T:-0.8}
SUBJECTS=${SUBJECTS:-"1 5 6 7"}
SYN_RATIO=${SYN_RATIO:-0.25}
N_RUNS=${N_RUNS:-3}
CLS_EPOCHS=${CLS_EPOCHS:-100}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-3000}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-1500}
RUN_PRETRAIN=${RUN_PRETRAIN:-1}
SAMPLE_MODE=${SAMPLE_MODE:-anchored}
ANCHOR_T_START=${ANCHOR_T_START:-0.75}
EXP_NAME=${EXP_NAME:-"seed4_de_noiseaware_cw01_margin005_fast"}
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/eeg_feature_smooth"}
FEAT_ROOT=${FEAT_ROOT:-"/root/autodl-tmp/eeg_feature_smooth"}

TRAIN_TRIALS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"   # 前 16 个 trial 训练
TEST_TRIALS="16,17,18,19,20,21,22,23"                    # 后 8 个 trial 测试

PRETRAIN_DIR=${PRETRAIN_DIR:-"/root/autodl-tmp/results/${EXP_NAME}_pretrain"}
FT_BASE=${FT_BASE:-"/root/autodl-tmp/results/${EXP_NAME}_ft"}

# ============================================================
# 第一步: 全被试预训练 (只跑一次, 所有被试的 DE 数据)
# ============================================================
echo "====== 实验配置 ======"
echo "EXP_NAME=${EXP_NAME}"
echo "SUBJECTS=${SUBJECTS}"
echo "CLASSIFIER_WEIGHT=${CLASSIFIER_WEIGHT}"
echo "CONDITION_MARGIN_WEIGHT=${CONDITION_MARGIN_WEIGHT}"
echo "CLS_EPOCHS=${CLS_EPOCHS}"
echo "PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS}"
echo "FINETUNE_EPOCHS=${FINETUNE_EPOCHS}"
echo "SAMPLE_MODE=${SAMPLE_MODE}"
echo "ANCHOR_T_START=${ANCHOR_T_START}"
echo "PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "FT_BASE=${FT_BASE}"

if [ "${RUN_PRETRAIN}" = "1" ]; then
    echo "====== [第一步] 全被试预训练 ======"
    python train_seed4.py --config ${CONFIG} --gpu ${GPU} \
        --conditional --classifier_weight ${CLASSIFIER_WEIGHT} \
        --cls_epochs ${CLS_EPOCHS} \
        --condition_margin_weight ${CONDITION_MARGIN_WEIGHT} \
        --condition_margin ${CONDITION_MARGIN} --condition_margin_max_t ${CONDITION_MARGIN_MAX_T} \
        --split_mode trial \
        --train_trials ${TRAIN_TRIALS} --test_trials ${TEST_TRIALS} \
        --max_epochs ${PRETRAIN_EPOCHS} \
        --sample_mode ${SAMPLE_MODE} --anchor_t_start ${ANCHOR_T_START} \
        --results_dir ${PRETRAIN_DIR}
else
    echo "====== [第一步] 跳过预训练, 使用已有 checkpoint ======"
fi

# ============================================================
# 第二步: 逐被试微调 + 生成 + 评估
# ============================================================
echo "====== [第二步] 逐被试微调 + 生成 + 评估 ======"
for SUBJ in ${SUBJECTS}; do
    echo "===== 被试 ${SUBJ} ====="

    # 微调
    python train_seed4.py --config ${CONFIG} --gpu ${GPU} \
        --conditional --classifier_weight ${CLASSIFIER_WEIGHT} \
        --cls_epochs ${CLS_EPOCHS} \
        --condition_margin_weight ${CONDITION_MARGIN_WEIGHT} \
        --condition_margin ${CONDITION_MARGIN} --condition_margin_max_t ${CONDITION_MARGIN_MAX_T} \
        --split_mode trial --subject ${SUBJ} \
        --train_trials ${TRAIN_TRIALS} --test_trials ${TEST_TRIALS} \
        --checkpoint ${PRETRAIN_DIR}/checkpoint-best.pt \
        --finetune --max_epochs ${FINETUNE_EPOCHS} \
        --num_samples 5000 \
        --sample_mode ${SAMPLE_MODE} --anchor_t_start ${ANCHOR_T_START} \
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
        --mode compare --syn_ratio ${SYN_RATIO} --n_runs ${N_RUNS}
done

echo "====== 全部完成 ======"
