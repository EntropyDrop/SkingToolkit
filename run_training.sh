#!/bin/bash

# Configuration paths (Defaulting to folders relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="../black-forest-labs/FLUX.2-klein-base-4B"  # Path containing Flux2Klein weights (e.g. flux-2-klein-base-4b.safetensors and VAE)
TEXT_ENCODER_PATH="../Qwen/Qwen3-4B" # Path to Qwen text encoder model
DATA_DIR="../SkingDataset/skins"
PHOTOS_DIR="../SkingDataset/control_imgs"
MAPPINGS_DIR="${MAPPINGS_DIR:-}"
if [ -z "$MAPPINGS_DIR" ]; then
    for candidate in \
        "../differentiable_minecraft_renderer/mappings" \
        "./mappings" \
        "../github/differentiable_minecraft_renderer/mappings" \
        "$HOME/llms/differentiable_minecraft_renderer/mappings" \
        "$HOME/Documents/entropydrop_website/differentiable_minecraft_renderer/mappings"
    do
        if [ -d "$candidate" ]; then
            MAPPINGS_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$MAPPINGS_DIR" ] || [ ! -d "$MAPPINGS_DIR" ]; then
    echo "[X] Could not find differentiable renderer mappings."
    echo "    Expected files like static_front_mapping.pt and static_back_mapping.pt."
    echo "    Set MAPPINGS_DIR explicitly, for example:"
    echo "    MAPPINGS_DIR=/path/to/differentiable_minecraft_renderer/mappings ./run_training.sh"
    exit 1
fi
OUTPUT_DIR="output/flux_skin_lora"
VALIDATION_PHOTOS_DIR="../SkingValidation" # 测试验证图根目录（包含 front/ 和 back/）
VALIDATION_STEPS=100

# Hyperparameters
LR=1e-4
BATCH_SIZE=2
EPOCHS=1000
SAVE_EVERY_EPOCHS=500
PRECISION="bf16"

# Loss weighting coefficients
LAMBDA_LATENT=1.0
LAMBDA_UV=1.0
LAMBDA_RENDER=1.0
LAMBDA_LPIPS=0.1
VIEWS="static_front,static_back"
FOREGROUND_WEIGHT=1.0

# Print info
echo "=========================================================="
echo "Starting Flux2Klein Differentiable Training..."
echo "Model Path:        $MODEL_PATH"
echo "Text Encoder Path: $TEXT_ENCODER_PATH"
echo "Dataset Dir:       $DATA_DIR"
echo "Photos Dir:        $PHOTOS_DIR"
echo "Mappings Dir:      $MAPPINGS_DIR"
echo "Output Dir:        $OUTPUT_DIR"
echo "Batch Size:        $BATCH_SIZE"
echo "Epochs:            $EPOCHS"
echo "Precision:         $PRECISION"
echo "Views for loss:    $VIEWS"
echo "=========================================================="

# Run the training script via Hugging Face Accelerate or Python directly
accelerate launch train.py \
    --model_path "$MODEL_PATH" \
    --model_type "flux2klein" \
    --text_encoder_type "qwen" \
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
    --lambda_latent "$LAMBDA_LATENT" \
    --lambda_uv "$LAMBDA_UV" \
    --lambda_render "$LAMBDA_RENDER" \
    --lambda_lpips "$LAMBDA_LPIPS" \
    --use_lpips \
    --views "$VIEWS" \
    --foreground_weight "$FOREGROUND_WEIGHT" \
    --use_lora True \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "qkv,linear1,linear2,proj" \
    --validation_photos_dir "$VALIDATION_PHOTOS_DIR" \
    --validation_steps "$VALIDATION_STEPS"
