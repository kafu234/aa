#!/bin/bash
# ============================================================
# 跨 Subject (LOSO) 实验: 14人训练, 1人测试
# 流程: 条件训练(14人) → 无标签微调(留出被试) → 生成 → 评估
# ============================================================

DATA_ROOT="/root/autodl-tmp/ExtractedFeatures"
CONFIG="./Config/seed_de_gen_full.yaml"  # 14人数据量大, 用大模型
GPU=0
TRAIN_EPOCHS=10000
FT_EPOCHS=1000
CLS_WEIGHT=0.5
SYN_RATIO=0.25
N_RUNS=3

if [ $# -gt 0 ]; then
    SUBJECTS="$@"
else
    SUBJECTS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
fi

echo "============================================"
echo "  跨 Subject (LOSO) 实验"
echo "  训练: 14 人, 测试: 留出 1 人"
echo "  被试: $SUBJECTS"
echo "============================================"

for SUBJ in $SUBJECTS; do
    echo ""
    echo "========================================"
    echo "  留出被试 $SUBJ (其余 14 人训练)"
    echo "========================================"

    BASE_DIR="/root/autodl-tmp/results/cross_subject/s${SUBJ}"
    ADAPT_DIR="${BASE_DIR}_adapted"

    # ---- 1. 条件训练 (14 人) ----
    echo "[1/4] 条件训练 (14人, ${TRAIN_EPOCHS} epochs)..."
    python train_seed.py --config $CONFIG --gpu $GPU \
        --conditional --classifier_weight $CLS_WEIGHT \
        --split_mode subject --test_subject $SUBJ \
        --max_epochs $TRAIN_EPOCHS \
        --results_dir $BASE_DIR

    # ---- 2. 无标签微调 (留出被试的数据, 不用标签) ----
    echo "[2/4] 无标签微调 (被试${SUBJ}数据, ${FT_EPOCHS} epochs)..."
    python train_seed.py --config $CONFIG --gpu $GPU \
        --conditional \
        --split_mode subject --test_subject $SUBJ \
        --checkpoint ${BASE_DIR}/checkpoint-best.pt \
        --finetune --max_epochs $FT_EPOCHS \
        --unlabeled_finetune --use_test_period \
        --results_dir $ADAPT_DIR

    # ---- 3. 生成 ----
    echo "[3/4] 生成..."
    python train_seed.py --config $CONFIG --gpu $GPU \
        --conditional \
        --split_mode subject --test_subject $SUBJ \
        --checkpoint ${ADAPT_DIR}/checkpoint-best.pt \
        --sample_only \
        --results_dir $ADAPT_DIR

    GEN_FILE=$(ls ${ADAPT_DIR}/generated_*.npz 2>/dev/null | head -1)
    if [ -z "$GEN_FILE" ]; then
        echo "  ⚠️ 生成失败, 跳过"
        continue
    fi

    # ---- 4. 评估 ----
    echo "[4/4] 评估..."
    python eval_de_classifier.py \
        --data_root $DATA_ROOT \
        --synthetic_path $GEN_FILE \
        --split_mode subject --test_subject $SUBJ \
        --mode compare --model dgcnn \
        --syn_ratio $SYN_RATIO --n_runs $N_RUNS

done

echo ""
echo "============================================"
echo "  跨 Subject (LOSO) 实验完成!"
echo "============================================"