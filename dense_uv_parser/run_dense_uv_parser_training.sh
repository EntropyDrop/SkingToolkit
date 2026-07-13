#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

find_latest_checkpoint() {
  local best_v=-1
  local best_checkpoint=""
  local dir base suffix v

  shopt -s nullglob
  for dir in runs/dense_uv_parser_v*; do
    [[ -d "$dir" && -f "$dir/latest.pt" ]] || continue
    base="$(basename "$dir")"
    suffix="${base#dense_uv_parser_v}"
    [[ "$suffix" =~ ^[0-9]+$ ]] || continue
    v=$((10#$suffix))
    if (( v > best_v )); then
      best_v="$v"
      best_checkpoint="$dir/latest.pt"
    fi
  done
  shopt -u nullglob

  printf '%s\n' "$best_checkpoint"
}

RESUME="${RESUME:-}"
if [[ "$RESUME" == "latest" ]]; then
  RESUME="$(find_latest_checkpoint)"
  if [[ -z "$RESUME" ]]; then
    echo "No runs/dense_uv_parser_v*/latest.pt checkpoint found to resume." >&2
    exit 1
  fi
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

if [[ -n "$RESUME" && -z "${RUN_NAME:-}" ]]; then
  OUTPUT_DIR="$(dirname "$RESUME")"
  RUN_NAME="$(basename "$OUTPUT_DIR")"
elif [[ -z "${RUN_NAME:-}" ]]; then
  v=1
  while [[ -d "runs/dense_uv_parser_v${v}" ]]; do
    ((v++))
  done
  RUN_NAME="dense_uv_parser_v${v}"
  OUTPUT_DIR="runs/$RUN_NAME"
else
  OUTPUT_DIR="runs/$RUN_NAME"
fi

DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_SIZE="${MAPPINGS_SIZE:-512x1024}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings_${MAPPINGS_SIZE}}"
VIEWS="${VIEWS:-walk_front_both_layer_ortho,walk_back_both_layer_ortho}"
PARSER_MODE="${PARSER_MODE:-geometry_fit}"
MAX_SAMPLES="${MAX_SAMPLES:-30000}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-true}"
LOG_EVERY="${LOG_EVERY:-50}"
BEST_METRIC="${BEST_METRIC:-loss_geometry}"

AUGMENT="${AUGMENT:-true}"
AUGMENT_VALIDATION="${AUGMENT_VALIDATION:-true}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.03}"
SCALE_RANGE="${SCALE_RANGE:-0.03}"
BACKGROUND_AUGMENT="${BACKGROUND_AUGMENT:-true}"
BACKGROUND_AUGMENT_PROB="${BACKGROUND_AUGMENT_PROB:-0.9}"
SEMANTIC_GATE="${SEMANTIC_GATE:-true}"
AFFINE_REFINE="${AFFINE_REFINE:-true}"
AFFINE_REFINE_TRANSLATION_PX="${AFFINE_REFINE_TRANSLATION_PX:-8.0}"
AFFINE_REFINE_SCALE="${AFFINE_REFINE_SCALE:-0.0}"
ROUTE_CONFIDENCE_THRESHOLD="${ROUTE_CONFIDENCE_THRESHOLD:-0.0}"
ROUTE_MARGIN_THRESHOLD="${ROUTE_MARGIN_THRESHOLD:-0.0}"
OUTER_ROUTE_CONFIDENCE_THRESHOLD="${OUTER_ROUTE_CONFIDENCE_THRESHOLD:-0.10}"
OUTER_ROUTE_MARGIN_THRESHOLD="${OUTER_ROUTE_MARGIN_THRESHOLD:-0.20}"
OUTER_UV_MIN_COVERAGE="${OUTER_UV_MIN_COVERAGE:-0.5}"
SPLAT_COLOR_AGGREGATION="${SPLAT_COLOR_AGGREGATION:-exact_mode}"
ALLOW_SEMANTIC_FALLBACK="${ALLOW_SEMANTIC_FALLBACK:-false}"

LAMBDA_FOREGROUND="${LAMBDA_FOREGROUND:-1.0}"
LAMBDA_LAYER="${LAMBDA_LAYER:-1.0}"
LAMBDA_PART="${LAMBDA_PART:-0.5}"
LAMBDA_FACE="${LAMBDA_FACE:-0.5}"
LAMBDA_LAYER_FACE="${LAMBDA_LAYER_FACE:-1.0}"
LAMBDA_UV="${LAMBDA_UV:-0.25}"
LAMBDA_UV_CLASS="${LAMBDA_UV_CLASS:-1.0}"
LAMBDA_AFFINE="${LAMBDA_AFFINE:-1.0}"
LAMBDA_SURFACE="${LAMBDA_SURFACE:-1.0}"
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
semantic_gate_args=()
if [[ "$SEMANTIC_GATE" == "true" ]]; then
  semantic_gate_args=(--semantic_gate)
else
  semantic_gate_args=(--no_semantic_gate)
fi
affine_refine_args=()
if [[ "$AFFINE_REFINE" == "true" ]]; then
  affine_refine_args=(--affine_refine)
else
  affine_refine_args=(--no_affine_refine)
fi
fallback_args=()
if [[ "$ALLOW_SEMANTIC_FALLBACK" == "true" ]]; then
  fallback_args=(--allow_semantic_fallback)
fi
cudnn_args=()
if [[ "$CUDNN_BENCHMARK" == "true" ]]; then
  cudnn_args=(--cudnn_benchmark)
else
  cudnn_args=(--no_cudnn_benchmark)
fi
resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args=(--resume "$RESUME")
fi

python train.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mappings_dir "$MAPPINGS_DIR" \
  --views "$VIEWS" \
  --parser_mode "$PARSER_MODE" \
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
  --best_metric "$BEST_METRIC" \
  --lambda_foreground "$LAMBDA_FOREGROUND" \
  --lambda_layer "$LAMBDA_LAYER" \
  --lambda_part "$LAMBDA_PART" \
  --lambda_face "$LAMBDA_FACE" \
  --lambda_layer_face "$LAMBDA_LAYER_FACE" \
  --lambda_uv "$LAMBDA_UV" \
  --lambda_uv_class "$LAMBDA_UV_CLASS" \
  --lambda_affine "$LAMBDA_AFFINE" \
  --lambda_surface "$LAMBDA_SURFACE" \
  --affine_refine_translation_px "$AFFINE_REFINE_TRANSLATION_PX" \
  --affine_refine_scale "$AFFINE_REFINE_SCALE" \
  --route_confidence_threshold "$ROUTE_CONFIDENCE_THRESHOLD" \
  --route_margin_threshold "$ROUTE_MARGIN_THRESHOLD" \
  --outer_route_confidence_threshold "$OUTER_ROUTE_CONFIDENCE_THRESHOLD" \
  --outer_route_margin_threshold "$OUTER_ROUTE_MARGIN_THRESHOLD" \
  --outer_uv_min_coverage "$OUTER_UV_MIN_COVERAGE" \
  --splat_color_aggregation "$SPLAT_COLOR_AGGREGATION" \
  "${augment_args[@]}" \
  "${background_args[@]}" \
  "${semantic_gate_args[@]}" \
  "${affine_refine_args[@]}" \
  "${fallback_args[@]}" \
  "${uv_class_args[@]}" \
  "${cudnn_args[@]}" \
  "${resume_args[@]}"
