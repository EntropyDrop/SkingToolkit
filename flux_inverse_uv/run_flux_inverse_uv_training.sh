#!/bin/bash

# Configuration paths (Defaulting to folders relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="../../black-forest-labs/FLUX.2-klein-base-4B"
TEXT_ENCODER_PATH="../../Qwen/Qwen3-4B"
CONTROL_IMGS_DIR="./control_imgs"
TARGET_IMGS_DIR="./target_imgs"
OUTPUT_DIR="./output/flux_inverse_uv_lora_v3"
VALIDATION_DIR="../../SkingDataset/DDJ_real2render/test_output"
VALIDATION_STEPS=500

# Hyperparameters
LR=1e-4
BATCH_SIZE=1
EPOCHS=1000
SAVE_EVERY_EPOCHS=1
PRECISION="bf16"
RESOLUTION=512

# Auxiliary Pixel Loss hyperparameters (Set lambda_pixel > 0 to enable, requires more VRAM)
LAMBDA_PIXEL=1.0
LAMBDA_DOT_WEIGHT=15.0
LAMBDA_UNIFORMITY=1.0

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
echo "Control Imgs Dir:  $CONTROL_IMGS_DIR"
echo "Target Imgs Dir:   $TARGET_IMGS_DIR"
echo "Validation Dir:    $VALIDATION_DIR"
echo "Output Dir:        $OUTPUT_DIR"
echo "Batch Size:        $BATCH_SIZE"
echo "Epochs:            $EPOCHS"
echo "Precision:         $PRECISION"
echo "Resolution:        $RESOLUTION"
echo "Pixel Loss Weight: lambda_pixel=$LAMBDA_PIXEL (dot_weight=$LAMBDA_DOT_WEIGHT, uniformity=$LAMBDA_UNIFORMITY)"
echo "LoRA Rank/Alpha:   rank=$LORA_LINEAR_RANK alpha=$LORA_LINEAR_ALPHA"
echo "=========================================================="

# Run training via Accelerate
accelerate launch train.py \
    --model_path "$MODEL_PATH" \
    --text_encoder_path "$TEXT_ENCODER_PATH" \
    --control_imgs_dir "$CONTROL_IMGS_DIR" \
    --target_imgs_dir "$TARGET_IMGS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --mixed_precision "$PRECISION" \
    --resolution "$RESOLUTION" \
    --lambda_pixel "$LAMBDA_PIXEL" \
    --lambda_dot_weight "$LAMBDA_DOT_WEIGHT" \
    --lambda_uniformity "$LAMBDA_UNIFORMITY" \
    --lora_rank "$LORA_LINEAR_RANK" \
    --lora_alpha "$LORA_LINEAR_ALPHA" \
    --lora_conv_rank "$LORA_CONV_RANK" \
    --lora_conv_alpha "$LORA_CONV_ALPHA" \
    --lora_target_modules "qkv,linear1,linear2,proj" \
    --validation_dir "$VALIDATION_DIR" \
    --validation_steps "$VALIDATION_STEPS"
