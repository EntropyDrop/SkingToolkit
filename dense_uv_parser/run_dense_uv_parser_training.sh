#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

if [[ -z "${RUN_NAME:-}" ]]; then
  v=1
  while [[ -d "runs/dense_uv_parser_v${v}" ]]; do
    ((v++))
  done
  RUN_NAME="dense_uv_parser_v${v}"
fi

DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_SIZE="${MAPPINGS_SIZE:-256x512}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings_${MAPPINGS_SIZE}}"
VIEWS="${VIEWS:-walk_front_both_layer_ortho,walk_back_both_layer_ortho}"
MAX_SAMPLES="${MAX_SAMPLES:-30000}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-true}"
LOG_EVERY="${LOG_EVERY:-50}"

AUGMENT="${AUGMENT:-true}"
AUGMENT_VALIDATION="${AUGMENT_VALIDATION:-true}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.03}"
SCALE_RANGE="${SCALE_RANGE:-0.03}"
BACKGROUND_AUGMENT="${BACKGROUND_AUGMENT:-true}"
BACKGROUND_AUGMENT_PROB="${BACKGROUND_AUGMENT_PROB:-0.9}"

LAMBDA_FOREGROUND="${LAMBDA_FOREGROUND:-1.0}"
LAMBDA_LAYER="${LAMBDA_LAYER:-1.0}"
LAMBDA_PART="${LAMBDA_PART:-0.5}"
LAMBDA_FACE="${LAMBDA_FACE:-0.5}"
LAMBDA_UV="${LAMBDA_UV:-0.25}"
LAMBDA_UV_CLASS="${LAMBDA_UV_CLASS:-1.0}"
UV_CLASSIFICATION="${UV_CLASSIFICATION:-true}"

augment_args=()
if [[ "$AUGMENT" == "true" ]]; then
  augment_args=(
    --augment
    --translation_scale "$TRANSLATION_SCALE"
    --scale_range "$SCALE_RANGE"
  )
else
  augment_args=(--no_augment)
fi
if [[ "$AUGMENT_VALIDATION" == "true" ]]; then
  augment_args+=(--augment_validation)
else
  augment_args+=(--no_augment_validation)
fi
uv_class_args=()
if [[ "$UV_CLASSIFICATION" == "true" ]]; then
  uv_class_args=(--uv_classification)
else
  uv_class_args=(--no_uv_classification)
fi
background_args=()
if [[ "$BACKGROUND_AUGMENT" == "true" ]]; then
  background_args=(
    --background_augment
    --background_augment_prob "$BACKGROUND_AUGMENT_PROB"
  )
else
  background_args=(--no_background_augment)
fi
cudnn_args=()
if [[ "$CUDNN_BENCHMARK" == "true" ]]; then
  cudnn_args=(--cudnn_benchmark)
else
  cudnn_args=(--no_cudnn_benchmark)
fi

python train.py \
  --data_dir "$DATA_DIR" \
  --output_dir "runs/$RUN_NAME" \
  --mappings_dir "$MAPPINGS_DIR" \
  --views "$VIEWS" \
  --max_samples "$MAX_SAMPLES" \
  --base_channels "$BASE_CHANNELS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --mixed_precision "$MIXED_PRECISION" \
  --matmul_precision "$MATMUL_PRECISION" \
  --log_every "$LOG_EVERY" \
  --lambda_foreground "$LAMBDA_FOREGROUND" \
  --lambda_layer "$LAMBDA_LAYER" \
  --lambda_part "$LAMBDA_PART" \
  --lambda_face "$LAMBDA_FACE" \
  --lambda_uv "$LAMBDA_UV" \
  --lambda_uv_class "$LAMBDA_UV_CLASS" \
  "${augment_args[@]}" \
  "${background_args[@]}" \
  "${uv_class_args[@]}" \
  "${cudnn_args[@]}"
