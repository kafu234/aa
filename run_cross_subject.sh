#!/bin/bash
# Transductive LOSO experiment.
# Source labels are used for training; target labels are used only by final evaluation.
set -euo pipefail
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-seed}                 # seed or seed4
GPU=${GPU:-0}
SOURCE_EPOCHS=${SOURCE_EPOCHS:-10000}
ADAPT_EPOCHS=${ADAPT_EPOCHS:-1000}
SCORER_EPOCHS=${SCORER_EPOCHS:-200}
SCORE_RUNS=${SCORE_RUNS:-3}
EVAL_EPOCHS=${EVAL_EPOCHS:-200}
N_RUNS=${N_RUNS:-5}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1024}
EVAL_VAL_INTERVAL=${EVAL_VAL_INTERVAL:-5}
EVAL_PATIENCE=${EVAL_PATIENCE:-20}
PSEUDO_THRESHOLD=${PSEUDO_THRESHOLD:-0.8}
PSEUDO_MIN_AGREEMENT=${PSEUDO_MIN_AGREEMENT:-0.67}
PSEUDO_MIN_PER_CLASS=${PSEUDO_MIN_PER_CLASS:-100}
PSEUDO_MAX_PER_CLASS=${PSEUDO_MAX_PER_CLASS:-1000}
PSEUDO_RATIO=${PSEUDO_RATIO:-0.1}
SYN_RATIO=${SYN_RATIO:-0.10}
NUM_SYNTHETIC=${NUM_SYNTHETIC:-30000}
ANCHOR_T_START=${ANCHOR_T_START:-0.45}
CLASSIFIER_WEIGHT=${CLASSIFIER_WEIGHT:-0.1}
RUN_SOURCE=${RUN_SOURCE:-1}
RUN_PSEUDO=${RUN_PSEUDO:-1}
RUN_ADAPT=${RUN_ADAPT:-1}

if [ "$DATASET" = "seed4" ]; then
    DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/eeg_feature_smooth}
    CONFIG=${CONFIG:-./Config/seed4_de_gen.yaml}
    TRAIN_SCRIPT=train_seed4.py
else
    DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/ExtractedFeatures}
    CONFIG=${CONFIG:-./Config/seed_de_gen_full.yaml}
    TRAIN_SCRIPT=train_seed.py
fi

RESULT_ROOT=${RESULT_ROOT:-/root/autodl-tmp/results/${DATASET}_cross_subject_adaptation}
SUBJECTS=${SUBJECTS:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"}
LOG_FILE=${LOG_FILE:-${RESULT_ROOT}/run.log}

mkdir -p "${RESULT_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Run started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Log file: ${LOG_FILE}"
echo "NUM_SYNTHETIC=${NUM_SYNTHETIC}; PSEUDO_RATIO=${PSEUDO_RATIO}; SYN_RATIO=${SYN_RATIO}"

for SUBJ in $SUBJECTS; do
    echo "============================================================"
    echo "Target subject ${SUBJ}; dataset=${DATASET}; transductive LOSO"
    echo "============================================================"
    BASE_DIR=${RESULT_ROOT}/s${SUBJ}/source
    ADAPT_DIR=${RESULT_ROOT}/s${SUBJ}/target_adapted
    PSEUDO_FILE=${RESULT_ROOT}/s${SUBJ}/target_pseudo.npz
    mkdir -p ${RESULT_ROOT}/s${SUBJ}

    if [ "$RUN_SOURCE" = "1" ]; then
        python ${TRAIN_SCRIPT} --config ${CONFIG} --gpu ${GPU} \
            --conditional --classifier_weight ${CLASSIFIER_WEIGHT} \
            --split_mode subject --test_subject ${SUBJ} \
            --validation_level subject --validation_ratio 0.05 \
            --max_epochs ${SOURCE_EPOCHS} \
            --skip_generation --results_dir ${BASE_DIR}
    fi

    if [ "$RUN_PSEUDO" = "1" ]; then
        python pseudo_label_target.py --dataset ${DATASET} \
            --data_root ${DATA_ROOT} --test_subject ${SUBJ} \
            --score_runs ${SCORE_RUNS} --epochs ${SCORER_EPOCHS} \
            --threshold ${PSEUDO_THRESHOLD} --min_agreement ${PSEUDO_MIN_AGREEMENT} \
            --min_per_class ${PSEUDO_MIN_PER_CLASS} \
            --max_per_class ${PSEUDO_MAX_PER_CLASS} \
            --gpu ${GPU} --output ${PSEUDO_FILE}
    fi

    if [ "$RUN_ADAPT" = "1" ]; then
        python ${TRAIN_SCRIPT} --config ${CONFIG} --gpu ${GPU} \
            --conditional --split_mode subject --test_subject ${SUBJ} \
            --checkpoint ${BASE_DIR}/checkpoint-best.pt \
            --finetune --max_epochs ${ADAPT_EPOCHS} \
            --unlabeled_finetune --use_test_period \
            --allow_transductive_test_adaptation \
            --anchor_bundle ${PSEUDO_FILE} \
            --sample_mode anchored --anchor_t_start ${ANCHOR_T_START} \
            --num_samples ${NUM_SYNTHETIC} --results_dir ${ADAPT_DIR}
    fi

    GEN_FILE=$(find ${ADAPT_DIR} -maxdepth 1 -name 'generated_*.npz' -print -quit)
    if [ -z "$GEN_FILE" ]; then
        echo "No generated file found for subject ${SUBJ}" >&2
        exit 1
    fi

    echo "===== Target subject ${SUBJ}: syn_ratio=${SYN_RATIO} ====="
    python eval_cross_subject_adaptation.py --dataset ${DATASET} \
        --data_root ${DATA_ROOT} --test_subject ${SUBJ} \
        --pseudo_path ${PSEUDO_FILE} \
        --synthetic_path ${GEN_FILE} \
        --pseudo_ratio ${PSEUDO_RATIO} --syn_ratio ${SYN_RATIO} \
        --epochs ${EVAL_EPOCHS} --n_runs ${N_RUNS} \
        --batch_size ${EVAL_BATCH_SIZE} --val_interval ${EVAL_VAL_INTERVAL} \
        --patience ${EVAL_PATIENCE} --gpu ${GPU}
done
