#!/bin/bash
# ============================================================
# 跨 Session 实验: 所有被试 session 1+2 训练, session 3 测试
# 流程: 条件训练(session1+2) → 无标签微调(session3) → 生成 → 评估
# 注意: 该流程显式使用无标签测试域数据，属于 transductive adaptation。
# ============================================================

DATA_ROOT="/root/autodl-tmp/ExtractedFeatures"
CONFIG="./Config/seed_de_gen_full.yaml"  # 所有被试数据量大, 用大模型
GPU=0
TRAIN_EPOCHS=10000
FT_EPOCHS=1000
CLS_WEIGHT=0.5
SYN_RATIO=0.25
N_RUNS=3

BASE_DIR="/root/autodl-tmp/results/cross_session"
ADAPT_DIR="${BASE_DIR}_adapted"

echo "============================================"
echo "  跨 Session 实验"
echo "  训练: 所有被试 session 1+2"
echo "  测试: 所有被试 session 3"
echo "============================================"

# ---- 1. 条件训练 (所有被试 session 1+2) ----
echo "[1/4] 条件训练 (所有被试 session 1+2, ${TRAIN_EPOCHS} epochs)..."
python train_seed.py --config $CONFIG --gpu $GPU \
    --conditional --classifier_weight $CLS_WEIGHT \
    --split_mode session \
    --max_epochs $TRAIN_EPOCHS \
    --results_dir $BASE_DIR

# ---- 2. 无标签微调 (所有被试 session 3) ----
echo "[2/4] 无标签微调 (所有被试 session 3, ${FT_EPOCHS} epochs)..."
python train_seed.py --config $CONFIG --gpu $GPU \
    --conditional \
    --split_mode session \
    --checkpoint ${BASE_DIR}/checkpoint-best.pt \
    --finetune --max_epochs $FT_EPOCHS \
    --unlabeled_finetune --use_test_period \
    --allow_transductive_test_adaptation \
    --results_dir $ADAPT_DIR

# ---- 3. 生成 ----
echo "[3/4] 生成..."
python train_seed.py --config $CONFIG --gpu $GPU \
    --conditional \
    --split_mode session \
    --checkpoint ${ADAPT_DIR}/checkpoint-best.pt \
    --sample_only \
    --results_dir $ADAPT_DIR

GEN_FILE=$(ls ${ADAPT_DIR}/generated_*.npz 2>/dev/null | head -1)
if [ -z "$GEN_FILE" ]; then
    echo "⚠️ 生成失败"
    exit 1
fi

# ---- 4. 评估 ----
echo "[4/4] 评估..."
python eval_de_classifier.py \
    --data_root $DATA_ROOT \
    --synthetic_path $GEN_FILE \
    --split_mode session \
    --mode compare --model dgcnn \
    --syn_ratio $SYN_RATIO --n_runs $N_RUNS

echo ""
echo "============================================"
echo "  跨 Session 实验完成!"
echo "============================================"