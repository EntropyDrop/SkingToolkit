#!/bin/bash

# Configuration paths (Defaulting to folders relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="../../black-forest-labs/FLUX.2-klein-base-4B"
TEXT_ENCODER_PATH="../../Qwen/Qwen3-4B"
PHOTOS_DIR="./control_imgs"
TARGET_IMGS_DIR="./target_imgs"
OUTPUT_DIR="./output/flux_inverse_uv_lora"
VALIDATION_PHOTOS_DIR="../../SkingDataset/DDJ_real2render/test_output"
VALIDATION_STEPS=100

# Hyperparameters
LR=1e-4
BATCH_SIZE=1
EPOCHS=1000
SAVE_EVERY_EPOCHS=10
PRECISION="bf16"
RESOLUTION=512

# LoRA config
LORA_LINEAR_RANK=32
LORA_LINEAR_ALPHA=32
LORA_CONV_RANK=16
LORA_CONV_ALPHA=16

# Print training configuration
echo "=========================================================="
echo "Starting Flux Inverse UV LoRA Fine-Tuning..."
echo "Model Path:        $MODEL_PATH"
echo "Text Encoder Path: $TEXT_ENCODER_PATH"
echo "Photos Dir:        $PHOTOS_DIR"
echo "Target Imgs Dir:   $TARGET_IMGS_DIR"
echo "Output Dir:        $OUTPUT_DIR"
echo "Batch Size:        $BATCH_SIZE"
echo "Epochs:            $EPOCHS"
echo "Precision:         $PRECISION"
echo "Resolution:        $RESOLUTION"
echo "LoRA Rank/Alpha:   rank=$LORA_LINEAR_RANK alpha=$LORA_LINEAR_ALPHA"
echo "=========================================================="

# Run training via Accelerate
accelerate launch train.py \
    --model_path "$MODEL_PATH" \
    --text_encoder_path "$TEXT_ENCODER_PATH" \
    --photos_dir "$PHOTOS_DIR" \
    --target_imgs_dir "$TARGET_IMGS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --mixed_precision "$PRECISION" \
    --resolution "$RESOLUTION" \
    --lora_rank "$LORA_LINEAR_RANK" \
    --lora_alpha "$LORA_LINEAR_ALPHA" \
    --lora_conv_rank "$LORA_CONV_RANK" \
    --lora_conv_alpha "$LORA_CONV_ALPHA" \
    --lora_target_modules "qkv,linear1,linear2,proj" \
    --validation_photos_dir "$VALIDATION_PHOTOS_DIR" \
    --validation_steps "$VALIDATION_STEPS"
