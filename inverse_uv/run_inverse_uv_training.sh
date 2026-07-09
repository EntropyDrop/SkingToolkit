#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Limit PyTorch CPU parallelism; very high thread counts can cause lock contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

MODEL="${MODEL:-full}"
if [[ -z "${RUN_NAME:-}" ]]; then
  v=1
  while [[ -d "runs/inverse_uv_${MODEL}_v${v}" ]]; do
    ((v++))
  done
  RUN_NAME="inverse_uv_${MODEL}_v${v}"
fi
DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_SIZE="${MAPPINGS_SIZE:-256x512}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings_${MAPPINGS_SIZE}}"
VIEWS="${VIEWS:-walk_front_both_layer_ortho,walk_back_both_layer_ortho}"
MAX_SAMPLES="${MAX_SAMPLES:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
RESUME="${RESUME:-}"
RESUME_LR="${RESUME_LR:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-true}"
UNPROJECT_MODE="${UNPROJECT_MODE:-mean}"
BEST_METRIC="${BEST_METRIC:-loss_recon_total}"
SCHEDULER="${SCHEDULER:-cosine}"
MIN_LR="${MIN_LR:-1e-5}"
LOG_EVERY="${LOG_EVERY:-50}"

# --- Augmentation (pose robustness) ---
AUGMENT="${AUGMENT:-false}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.015}"
SCALE_RANGE="${SCALE_RANGE:-0.015}"
PERSPECTIVE_SCALE="${PERSPECTIVE_SCALE:-0.0}"

# --- PatchGAN loss ---
# Keep the default color-first. For a later texture-sharpening finetune,
# resume from best.pt with a very small value such as LAMBDA_GAN=0.005.
LAMBDA_GAN="${LAMBDA_GAN:-0.0}"

# --- Loss weights ---
LAMBDA_RGB="${LAMBDA_RGB:-2.0}"
LAMBDA_ALPHA="${LAMBDA_ALPHA:-0.8}"
LAMBDA_ALPHA_DICE="${LAMBDA_ALPHA_DICE:-0.5}"
LAMBDA_ALPHA_EDGE="${LAMBDA_ALPHA_EDGE:-0.5}"
LAMBDA_RENDER="${LAMBDA_RENDER:-0.2}"
LAMBDA_RENDER_ALPHA="${LAMBDA_RENDER_ALPHA:-0.4}"
LAMBDA_EDGE="${LAMBDA_EDGE:-1.0}"

resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args=(--resume "$RESUME")
fi
resume_lr_args=()
if [[ -n "$RESUME_LR" ]]; then
  resume_lr_args=(--resume_lr "$RESUME_LR")
fi

augment_args=()
if [[ "$AUGMENT" == "true" ]]; then
  augment_args=(
    --augment
    --translation_scale "$TRANSLATION_SCALE"
    --scale_range "$SCALE_RANGE"
    --perspective_scale "$PERSPECTIVE_SCALE"
  )
fi
cudnn_args=()
if [[ "$CUDNN_BENCHMARK" == "true" ]]; then
  cudnn_args=(--cudnn_benchmark)
else
  cudnn_args=(--no_cudnn_benchmark)
fi

python train.py \
  --data_dir "$DATA_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "runs/$RUN_NAME" \
  --views "$VIEWS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --val_split 0.1 \
  --mappings_dir "$MAPPINGS_DIR" \
  --save_every 1 \
  --preview_every 1 \
  --mixed_precision "$MIXED_PRECISION" \
  --matmul_precision "$MATMUL_PRECISION" \
  --unproject_mode "$UNPROJECT_MODE" \
  --best_metric "$BEST_METRIC" \
  --scheduler "$SCHEDULER" \
  --min_lr "$MIN_LR" \
  --log_every "$LOG_EVERY" \
  --lambda_gan "$LAMBDA_GAN" \
  --lambda_rgb "$LAMBDA_RGB" \
  --lambda_alpha "$LAMBDA_ALPHA" \
  --lambda_alpha_dice "$LAMBDA_ALPHA_DICE" \
  --lambda_alpha_edge "$LAMBDA_ALPHA_EDGE" \
  --lambda_render "$LAMBDA_RENDER" \
  --lambda_render_alpha "$LAMBDA_RENDER_ALPHA" \
  --lambda_edge "$LAMBDA_EDGE" \
  ${augment_args[@]+"${augment_args[@]}"} \
  ${cudnn_args[@]+"${cudnn_args[@]}"} \
  ${resume_lr_args[@]+"${resume_lr_args[@]}"} \
  ${resume_args[@]+"${resume_args[@]}"}
