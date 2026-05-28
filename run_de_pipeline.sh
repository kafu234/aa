#!/bin/bash
# ============================================================
# run_de_pipeline.sh — DE 空间生成完整流水线
# ============================================================
# 用法:
#   bash run_de_pipeline.sh           # 跑全部 15 个被试
#   bash run_de_pipeline.sh 1 2 3     # 只跑被试 1, 2, 3
# ============================================================

DATA_ROOT="/root/autodl-tmp/ExtractedFeatures"
CONFIG="./Config/seed_de_gen.yaml"
GPU=0
N_SAMPLES=5000           # 每类生成样本数 (总共 N_SAMPLES)
CLASSIFIER_WEIGHT=0.5
SYN_RATIOS="0.1 0.25 0.5 1.0"   # 测试多个混合比例
N_RUNS=3

# 被试列表
if [ $# -gt 0 ]; then
    SUBJECTS="$@"
else
    SUBJECTS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
fi

echo "============================================"
echo "  DE 空间生成流水线"
echo "  被试: $SUBJECTS"
echo "  生成量: $N_SAMPLES 样本"
echo "============================================"

for SUBJ in $SUBJECTS; do
    echo ""
    echo "========================================"
    echo "  被试 $SUBJ"
    echo "========================================"

    RESULT_DIR="./results/de_full/s${SUBJ}"

    # ---- 1. 训练生成模型 ----
    echo "[1/3] 训练生成模型..."
    python train_seed.py \
        --config $CONFIG \
        --gpu $GPU \
        --conditional \
        --classifier_weight $CLASSIFIER_WEIGHT \
        --cfg_dropout 0.2 \
        --guidance_scale 2.0 \
        --split_mode trial \
        --subject $SUBJ \
        --train_trials 0,1,2,3,4,5,6,7,8 \
        --test_trials 9,10,11,12,13,14 \
        --num_samples $N_SAMPLES \
        --results_dir $RESULT_DIR

    # 找到生成的 npz
    GEN_FILE=$(ls ${RESULT_DIR}/generated_*.npz 2>/dev/null | head -1)
    if [ -z "$GEN_FILE" ]; then
        echo "  ⚠️  被试 $SUBJ 生成失败, 跳过"
        continue
    fi
    echo "  生成文件: $GEN_FILE"

    # ---- 2. 诊断 ----
    echo "[2/3] 诊断合成数据质量..."
    python eval_de_classifier.py \
        --data_root $DATA_ROOT \
        --synthetic_path $GEN_FILE \
        --subject $SUBJ \
        --split_mode trial \
        --train_trials 0,1,2,3,4,5,6,7,8 \
        --test_trials 9,10,11,12,13,14 \
        --mode diagnose \
        --epochs 200

    # ---- 3. 对比评估 (多个 syn_ratio) ----
    echo "[3/3] 对比评估..."
    for RATIO in $SYN_RATIOS; do
        echo "  --- syn_ratio=$RATIO ---"
        python eval_de_classifier.py \
            --data_root $DATA_ROOT \
            --synthetic_path $GEN_FILE \
            --subject $SUBJ \
            --split_mode trial \
            --train_trials 0,1,2,3,4,5,6,7,8 \
            --test_trials 9,10,11,12,13,14 \
            --mode compare \
            --syn_ratio $RATIO \
            --n_runs $N_RUNS \
            --epochs 200
    done
done

echo ""
echo "============================================"
echo "  全部完成!"
echo "============================================"