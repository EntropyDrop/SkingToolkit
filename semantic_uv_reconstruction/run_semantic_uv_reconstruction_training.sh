#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

RESUME="${RESUME:-}"
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

if [[ -n "$RESUME" && -z "${RUN_NAME:-}" ]]; then
  OUTPUT_DIR="$(dirname "$RESUME")"
  RUN_NAME="$(basename "$OUTPUT_DIR")"
elif [[ -z "${RUN_NAME:-}" ]]; then
  v=1
  while [[ -d "runs/semantic_uv_reconstruction_v${v}" ]]; do
    ((v++))
  done
  RUN_NAME="semantic_uv_reconstruction_v${v}"
  OUTPUT_DIR="runs/$RUN_NAME"
else
  OUTPUT_DIR="runs/$RUN_NAME"
fi

DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_SIZE="${MAPPINGS_SIZE:-256x512}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../differentiable_minecraft_renderer/mappings_${MAPPINGS_SIZE}}"
VIEWS="${VIEWS:-walk_front_both_layer_ortho,walk_back_both_layer_ortho}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SEMANTIC_LABELS_DIR="${SEMANTIC_LABELS_DIR:-}"
SEMANTIC_CLASSES="${SEMANTIC_CLASSES:-13}"
SEMANTIC_BACKBONE="${SEMANTIC_BACKBONE:-siglip2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-base-patch16-224}"
SIGLIP_LOCAL_FILES_ONLY="${SIGLIP_LOCAL_FILES_ONLY:-false}"

BASE_CHANNELS="${BASE_CHANNELS:-32}"
TOKEN_CHANNELS="${TOKEN_CHANNELS:-128}"
QUERY_SIZE="${QUERY_SIZE:-16}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
ATTENTION_LAYERS="${ATTENTION_LAYERS:-2}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-4}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEVICE="${DEVICE:-auto}"

LAMBDA_UV_RGB="${LAMBDA_UV_RGB:-1.0}"
LAMBDA_OUTER_ALPHA="${LAMBDA_OUTER_ALPHA:-1.0}"
LAMBDA_OUTER_DICE="${LAMBDA_OUTER_DICE:-0.5}"
LAMBDA_SEMANTIC_UV="${LAMBDA_SEMANTIC_UV:-0.25}"
LAMBDA_SEMANTIC_PRESENCE="${LAMBDA_SEMANTIC_PRESENCE:-0.25}"
LAMBDA_SEMANTIC_COVERAGE="${LAMBDA_SEMANTIC_COVERAGE:-0.25}"
LAMBDA_SEMANTIC_COLOR="${LAMBDA_SEMANTIC_COLOR:-0.25}"
LAMBDA_RENDER_RGB="${LAMBDA_RENDER_RGB:-0.5}"
LAMBDA_RENDER_ALPHA="${LAMBDA_RENDER_ALPHA:-0.5}"
LAMBDA_SIGLIP_RENDER="${LAMBDA_SIGLIP_RENDER:-0.1}"

optional_args=()
if [[ -n "$MAX_SAMPLES" ]]; then
  optional_args+=(--max_samples "$MAX_SAMPLES")
fi
if [[ -n "$SEMANTIC_LABELS_DIR" ]]; then
  optional_args+=(--semantic_labels_dir "$SEMANTIC_LABELS_DIR")
fi
if [[ -n "$RESUME" ]]; then
  optional_args+=(--resume "$RESUME")
fi
if [[ "$SIGLIP_LOCAL_FILES_ONLY" == "true" ]]; then
  optional_args+=(--siglip_local_files_only)
fi

python train_semantic_uv_reconstruction.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mappings_dir "$MAPPINGS_DIR" \
  --views "$VIEWS" \
  --semantic_classes "$SEMANTIC_CLASSES" \
  --semantic_backbone "$SEMANTIC_BACKBONE" \
  --siglip_model "$SIGLIP_MODEL" \
  --base_channels "$BASE_CHANNELS" \
  --token_channels "$TOKEN_CHANNELS" \
  --query_size "$QUERY_SIZE" \
  --attention_heads "$ATTENTION_HEADS" \
  --attention_layers "$ATTENTION_LAYERS" \
  --attention_dropout "$ATTENTION_DROPOUT" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --min_lr_ratio "$MIN_LR_RATIO" \
  --mixed_precision "$MIXED_PRECISION" \
  --device "$DEVICE" \
  --lambda_uv_rgb "$LAMBDA_UV_RGB" \
  --lambda_outer_alpha "$LAMBDA_OUTER_ALPHA" \
  --lambda_outer_dice "$LAMBDA_OUTER_DICE" \
  --lambda_semantic_uv "$LAMBDA_SEMANTIC_UV" \
  --lambda_semantic_presence "$LAMBDA_SEMANTIC_PRESENCE" \
  --lambda_semantic_coverage "$LAMBDA_SEMANTIC_COVERAGE" \
  --lambda_semantic_color "$LAMBDA_SEMANTIC_COLOR" \
  --lambda_render_rgb "$LAMBDA_RENDER_RGB" \
  --lambda_render_alpha "$LAMBDA_RENDER_ALPHA" \
  --lambda_siglip_render "$LAMBDA_SIGLIP_RENDER" \
  "${optional_args[@]}"
