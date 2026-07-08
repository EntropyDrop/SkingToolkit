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
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings_256x512}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
RESUME="${RESUME:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
UNPROJECT_MODE="${UNPROJECT_MODE:-mode}"

# --- Augmentation (pose robustness) ---
AUGMENT="${AUGMENT:-true}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.03}"
SCALE_RANGE="${SCALE_RANGE:-0.03}"
PERSPECTIVE_SCALE="${PERSPECTIVE_SCALE:-0.008}"

# --- PatchGAN loss ---
LAMBDA_GAN="${LAMBDA_GAN:-0.1}"

# --- Loss weights ---
LAMBDA_RGB="${LAMBDA_RGB:-1.0}"
LAMBDA_ALPHA="${LAMBDA_ALPHA:-0.5}"
LAMBDA_RENDER="${LAMBDA_RENDER:-0.1}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.25}"

resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args=(--resume "$RESUME")
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

python train.py \
  --data_dir "$DATA_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "runs/$RUN_NAME" \
  --views walk_front_both_layer_ortho,walk_back_both_layer_ortho \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --val_split 0.1 \
  --mappings_dir "$MAPPINGS_DIR" \
  --save_every 1 \
  --preview_every 1 \
  --mixed_precision "$MIXED_PRECISION" \
  --unproject_mode "$UNPROJECT_MODE" \
  --lambda_gan "$LAMBDA_GAN" \
  --lambda_rgb "$LAMBDA_RGB" \
  --lambda_alpha "$LAMBDA_ALPHA" \
  --lambda_render "$LAMBDA_RENDER" \
  --lambda_edge "$LAMBDA_EDGE" \
  ${augment_args[@]+"${augment_args[@]}"} \
  ${resume_args[@]+"${resume_args[@]}"}
