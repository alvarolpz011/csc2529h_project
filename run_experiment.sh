#!/bin/bash

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET="materials"
DEGRADE=true
MAX_STEPS=2000  # Define this here so we can use it for checkpoint path

# ==========================================
# 2. AUTOMATED SETUP
# ==========================================
echo "Checking and preparing data for $DATASET..."
python3 src/download_data.py --dataset $DATASET --data_dir data

DATASET_PATH="data/${DATASET}"

if [ "$DEGRADE" = true ]; then
    MODE="degraded"
    DEGRADE_FLAG="--degrade_inputs"
else
    MODE="clean"
    DEGRADE_FLAG=""
fi

# Args for TRAINING
COMMON_TRAIN_ARGS="--mixed_precision=bf16 src/train_difix.py \
    --dataset_path=$DATASET_PATH \
    --max_train_steps $MAX_STEPS \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name difix"

# Args for EVAL (Simple python execution)
COMMON_EVAL_ARGS="src/eval_metrics.py \
    --dataset_path=$DATASET_PATH \
    --resolution=512"

echo "========================================================"
echo "Running Experiment Suite: $DATASET | Mode: $MODE"
echo "========================================================"

VARIANTS=(
    "base|"
    "depth|--use_depth"
    "canny|--use_canny"
    "depth_canny|--use_depth --use_canny"
)

for item in "${VARIANTS[@]}"; do
    VARIANT_NAME="${item%%|*}"
    VARIANT_FLAGS="${item##*|}"

    RUN_NAME="${DATASET}-${MODE}-${VARIANT_NAME}"
    OUTPUT_DIR="./outputs/${DATASET}/${MODE}/${VARIANT_NAME}"
    
    CKPT_PATH="${OUTPUT_DIR}/checkpoints/model_${MAX_STEPS}.pkl"

    if [ -f "$CKPT_PATH" ]; then
        echo "--> [SKIP] Variant $VARIANT_NAME already completed (Found $CKPT_PATH)"
        continue
    fi
    
    # ----------------------------------------
    # STEP 1: TRAINING
    # ----------------------------------------
    echo "--> [TRAIN] Starting Variant: $VARIANT_NAME"
    
    accelerate launch $COMMON_TRAIN_ARGS \
        --output_dir="$OUTPUT_DIR" \
        --tracker_run_name="$RUN_NAME" \
        $DEGRADE_FLAG \
        $VARIANT_FLAGS
        
    # ----------------------------------------
    # STEP 2: EVALUATION
    # ----------------------------------------
    echo "--> [EVAL] Evaluating Variant: $VARIANT_NAME"
    
    python3 $COMMON_EVAL_ARGS \
        --checkpoint_path="$CKPT_PATH" \
        --run_name="$RUN_NAME" \
        $DEGRADE_FLAG \
        $VARIANT_FLAGS
        
    echo "Done with $VARIANT_NAME"
    echo "--------------------------------------------------------"
done