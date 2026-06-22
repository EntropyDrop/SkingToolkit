#!/bin/bash

# Configuration paths (Defaulting to folders in the current workspace)
MODEL_PATH="models/flux2klein4b"  # Path containing Flux2Klein weights (e.g. flux-2-klein-base-4b.safetensors and VAE)
TEXT_ENCODER_PATH="Qwen/Qwen3-4B" # Path to Qwen text encoder model
DATA_DIR="../Sking/skins"
PHOTOS_DIR="../Sking/control_imgs"
MAPPINGS_DIR="../differentiable_minecraft_renderer/mappings"
OUTPUT_DIR="output/flux_skin_lora"

# Hyperparameters
LR=1e-4
BATCH_SIZE=2
EPOCHS=100
SAVE_EVERY_EPOCHS=5
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
    --lora_target_modules "qkv,linear1,linear2,proj"

