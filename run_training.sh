#!/bin/bash

# Configuration paths (Defaulting to folders relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="../black-forest-labs/FLUX.2-klein-base-4B"  # Path containing Flux2Klein weights (e.g. flux-2-klein-base-4b.safetensors and VAE)
TEXT_ENCODER_PATH="../Qwen/Qwen3-4B" # Path to Qwen text encoder model
DATA_DIR="../SkingDataset/skins"
PHOTOS_DIR="../SkingDataset/control_imgs"
OUTPUT_DIR="output/flux_skin_lora"
VALIDATION_PHOTOS_DIR="../SkingValidation" # 测试验证图根目录（包含 front/ 和 back/）
VALIDATION_STEPS=100

# Hyperparameters
LR=1e-4
BATCH_SIZE=1
EPOCHS=1000
SAVE_EVERY_EPOCHS=10
PRECISION="bf16"
RESOLUTION=1024

# Loss weighting coefficients. Keep UV/render disabled for ai-toolkit-style latent MSE training.
LAMBDA_LATENT=1.0
LAMBDA_UV=50.0
LAMBDA_RENDER=50.0
LAMBDA_LPIPS=0.5
VIEWS="static_front,static_back"
FOREGROUND_WEIGHT=1.0
RENDER_WARMUP_EPOCHS=200

# LoRA config aligned with ai-toolkit linear=32/linear_alpha=32. Conv knobs are
# passed through for parity, but this Flux2 transformer has no Conv2d modules.
LORA_LINEAR_RANK=32
LORA_LINEAR_ALPHA=32
LORA_CONV_RANK=16
LORA_CONV_ALPHA=16

MAPPINGS_DIR="../github/differentiable_minecraft_renderer/mappings"

# Print info
echo "=========================================================="
echo "Starting Flux2Klein ai-toolkit-compatible latent training..."
echo "Model Path:        $MODEL_PATH"
echo "Text Encoder Path: $TEXT_ENCODER_PATH"
echo "Dataset Dir:       $DATA_DIR"
echo "Photos Dir:        $PHOTOS_DIR"
echo "Mappings Dir:      $MAPPINGS_DIR"
echo "Output Dir:        $OUTPUT_DIR"
echo "Batch Size:        $BATCH_SIZE"
echo "Epochs:            $EPOCHS"
echo "Precision:         $PRECISION"
echo "Target Canvas:     $((RESOLUTION / 2))x$RESOLUTION"
echo "Loss Weights:      latent=$LAMBDA_LATENT uv=$LAMBDA_UV render=$LAMBDA_RENDER lpips=$LAMBDA_LPIPS"
echo "LoRA:              linear=$LORA_LINEAR_RANK alpha=$LORA_LINEAR_ALPHA conv=$LORA_CONV_RANK conv_alpha=$LORA_CONV_ALPHA"
echo "Views for loss:    $VIEWS"
echo "=========================================================="

# Run the training script via Hugging Face Accelerate or Python directly
accelerate launch train.py \
    --model_path "$MODEL_PATH" \
    --text_encoder_path "$TEXT_ENCODER_PATH" \
    --data_dir "$DATA_DIR" \
    --photos_dir "$PHOTOS_DIR" \
    --mappings_dir "$MAPPINGS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --mixed_precision "$PRECISION" \
    --resolution "$RESOLUTION" \
    --lambda_latent "$LAMBDA_LATENT" \
    --lambda_uv "$LAMBDA_UV" \
    --lambda_render "$LAMBDA_RENDER" \
    --lambda_lpips "$LAMBDA_LPIPS" \
    --views "$VIEWS" \
    --foreground_weight "$FOREGROUND_WEIGHT" \
    --render_warmup_epochs "$RENDER_WARMUP_EPOCHS" \
    --lora_rank "$LORA_LINEAR_RANK" \
    --lora_alpha "$LORA_LINEAR_ALPHA" \
    --lora_conv_rank "$LORA_CONV_RANK" \
    --lora_conv_alpha "$LORA_CONV_ALPHA" \
    --lora_target_modules "qkv,linear1,linear2,proj" \
    --validation_photos_dir "$VALIDATION_PHOTOS_DIR" \
    --validation_steps "$VALIDATION_STEPS"
