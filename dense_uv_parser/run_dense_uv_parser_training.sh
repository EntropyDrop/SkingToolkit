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
MAPPINGS_SIZE="${MAPPINGS_SIZE:-256x512}"
VIEWS="${VIEWS:-walk_front_both_layer_ortho,walk_back_both_layer_ortho}"

resolve_mappings_dir() {
  local requested="${MAPPINGS_DIR:-}"
  local name="mappings_${MAPPINGS_SIZE}"
  local candidate=""
  local discovered=""
  if [[ -n "$requested" ]]; then
    [[ -d "$requested" ]] || { echo "MAPPINGS_DIR does not exist: $requested" >&2; return 1; }
    MAPPINGS_DIR="$requested"
    return 0
  fi
  for candidate in \
    "../differentiable_minecraft_renderer/$name" \
    "../../differentiable_minecraft_renderer/$name" \
    "../../github/differentiable_minecraft_renderer/$name" \
    "../$name"; do
    if [[ -d "$candidate" ]]; then
      MAPPINGS_DIR="$candidate"
      return 0
    fi
  done
  discovered="$(find ../.. -maxdepth 5 -type d -name "$name" -print -quit 2>/dev/null || true)"
  [[ -n "$discovered" ]] || { echo "Could not find $name from $(pwd)." >&2; return 1; }
  MAPPINGS_DIR="$discovered"
}
resolve_mappings_dir

PARSER_MODE="${PARSER_MODE:-geometry_fit}"
MAX_SAMPLES="${MAX_SAMPLES:-30000}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.10}"
SEMANTIC_BACKBONE="${SEMANTIC_BACKBONE:-siglip2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-base-patch16-224}"
SIGLIP_LOCAL_FILES_ONLY="${SIGLIP_LOCAL_FILES_ONLY:-false}"
CACHE_SIGLIP_GLOBALS="${CACHE_SIGLIP_GLOBALS:-true}"
SIGLIP_CACHE_DIR="${SIGLIP_CACHE_DIR:-cache/semantic_dense_parser_siglip2_${MAPPINGS_SIZE}_${MAX_SAMPLES}}"
SIGLIP_CACHE_BATCH_SIZE="${SIGLIP_CACHE_BATCH_SIZE:-32}"
SEMANTIC_CHANNELS="${SEMANTIC_CHANNELS:-128}"
SEMANTIC_ATTENTION_HEADS="${SEMANTIC_ATTENTION_HEADS:-4}"
SEMANTIC_LAYERS="${SEMANTIC_LAYERS:-1}"
SEMANTIC_DROPOUT="${SEMANTIC_DROPOUT:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-true}"
LOG_EVERY="${LOG_EVERY:-50}"
BEST_METRIC="${BEST_METRIC:-loss_hard_uv_selection}"

AUGMENT="${AUGMENT:-false}"
AUGMENT_VALIDATION="${AUGMENT_VALIDATION:-false}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.0}"
SCALE_RANGE="${SCALE_RANGE:-0.0}"
BACKGROUND_AUGMENT="${BACKGROUND_AUGMENT:-true}"
BACKGROUND_AUGMENT_PROB="${BACKGROUND_AUGMENT_PROB:-0.9}"
SEMANTIC_GATE="${SEMANTIC_GATE:-true}"
AFFINE_REFINE="${AFFINE_REFINE:-false}"
AFFINE_REFINE_TRANSLATION_PX="${AFFINE_REFINE_TRANSLATION_PX:-0.0}"
AFFINE_REFINE_SCALE="${AFFINE_REFINE_SCALE:-0.0}"
# Do not reject ordinary inner-layer evidence during parser model selection.
# The asymmetric outer gates below are the controls that suppress the costly
# inner-to-outer routing errors. Runtime inference can still opt into the fully
# conservative profile when the input itself is noisy.
ROUTE_CONFIDENCE_THRESHOLD="${ROUTE_CONFIDENCE_THRESHOLD:-0.0}"
ROUTE_MARGIN_THRESHOLD="${ROUTE_MARGIN_THRESHOLD:-0.0}"
OUTER_ROUTE_CONFIDENCE_THRESHOLD="${OUTER_ROUTE_CONFIDENCE_THRESHOLD:-0.80}"
OUTER_ROUTE_MARGIN_THRESHOLD="${OUTER_ROUTE_MARGIN_THRESHOLD:-0.55}"
OUTER_UV_MIN_COVERAGE="${OUTER_UV_MIN_COVERAGE:-0.25}"
OUTER_GEOMETRY_RESCUE="${OUTER_GEOMETRY_RESCUE:-true}"
OUTER_RESCUE_CONFIDENCE_THRESHOLD="${OUTER_RESCUE_CONFIDENCE_THRESHOLD:-0.60}"
OUTER_RESCUE_MARGIN_THRESHOLD="${OUTER_RESCUE_MARGIN_THRESHOLD:-0.25}"
OUTER_RESCUE_MIN_COVERAGE="${OUTER_RESCUE_MIN_COVERAGE:-0.10}"
GEOMETRY_ROUTE_TEXEL_CONSENSUS="${GEOMETRY_ROUTE_TEXEL_CONSENSUS:-true}"
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
LAMBDA_OUTER_FALSE_POSITIVE="${LAMBDA_OUTER_FALSE_POSITIVE:-1.50}"
LAMBDA_OUTER_FALSE_NEGATIVE="${LAMBDA_OUTER_FALSE_NEGATIVE:-0.40}"
LAMBDA_ROUTE_CONFIDENCE="${LAMBDA_ROUTE_CONFIDENCE:-0.25}"
LAMBDA_SEMANTIC_PRESENCE="${LAMBDA_SEMANTIC_PRESENCE:-0.25}"
LAMBDA_SEMANTIC_COVERAGE="${LAMBDA_SEMANTIC_COVERAGE:-0.25}"
OUTER_FALSE_POSITIVE_GAMMA="${OUTER_FALSE_POSITIVE_GAMMA:-2.0}"
OUTER_FALSE_NEGATIVE_GAMMA="${OUTER_FALSE_NEGATIVE_GAMMA:-2.0}"
ROUTE_CLASS_WEIGHT_FLOOR="${ROUTE_CLASS_WEIGHT_FLOOR:-0.75}"
ROUTE_OUTER_CLASS_WEIGHT_CAP="${ROUTE_OUTER_CLASS_WEIGHT_CAP:-0.75}"
LAMBDA_SOFT_UV_RGB="${LAMBDA_SOFT_UV_RGB:-0.25}"
LAMBDA_SOFT_UV_ALPHA="${LAMBDA_SOFT_UV_ALPHA:-0.35}"
LAMBDA_SOFT_UV_INNER_RECALL="${LAMBDA_SOFT_UV_INNER_RECALL:-0.50}"
LAMBDA_SOFT_UV_OUTER_RECALL="${LAMBDA_SOFT_UV_OUTER_RECALL:-0.50}"
SOFT_UV_RECALL_HARD_FRACTION="${SOFT_UV_RECALL_HARD_FRACTION:-0.10}"
SOFT_UV_RECALL_HARD_WEIGHT="${SOFT_UV_RECALL_HARD_WEIGHT:-0.50}"
LAMBDA_RENDER_RGB="${LAMBDA_RENDER_RGB:-0.20}"
LAMBDA_RENDER_ALPHA="${LAMBDA_RENDER_ALPHA:-0.25}"
OUTER_SELECTION_PRECISION_WEIGHT="${OUTER_SELECTION_PRECISION_WEIGHT:-1.50}"
OUTER_SELECTION_RECALL_WEIGHT="${OUTER_SELECTION_RECALL_WEIGHT:-0.50}"
OUTER_SELECTION_IOU_WEIGHT="${OUTER_SELECTION_IOU_WEIGHT:-0.5}"
INNER_SELECTION_RECALL_WEIGHT="${INNER_SELECTION_RECALL_WEIGHT:-0.5}"
RENDER_SOFTMAX_TEMPERATURE="${RENDER_SOFTMAX_TEMPERATURE:-1.0}"
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
routing_consensus_args=()
if [[ "$GEOMETRY_ROUTE_TEXEL_CONSENSUS" == "true" ]]; then
  routing_consensus_args=(--geometry_route_texel_consensus)
else
  routing_consensus_args=(--no_geometry_route_texel_consensus)
fi
outer_rescue_args=()
if [[ "$OUTER_GEOMETRY_RESCUE" == "true" ]]; then
  outer_rescue_args=(--outer_geometry_rescue)
else
  outer_rescue_args=(--no_outer_geometry_rescue)
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

semantic_args=(--semantic_backbone "$SEMANTIC_BACKBONE")
if [[ "$SEMANTIC_BACKBONE" == "siglip2" ]]; then
  cache_args=()
  if [[ "$SIGLIP_LOCAL_FILES_ONLY" == "true" ]]; then
    cache_args+=(--siglip_local_files_only)
    semantic_args+=(--siglip_local_files_only)
  fi
  if [[ "$CACHE_SIGLIP_GLOBALS" == "true" ]]; then
    python ../semantic_uv_reconstruction/cache_siglip_globals.py \
      --data_dir "$DATA_DIR" \
      --cache_dir "$SIGLIP_CACHE_DIR" \
      --mappings_dir "$MAPPINGS_DIR" \
      --views "$VIEWS" \
      --siglip_model "$SIGLIP_MODEL" \
      --max_samples "$MAX_SAMPLES" \
      --batch_size "$SIGLIP_CACHE_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --prefetch_factor "$PREFETCH_FACTOR" \
      --mixed_precision "$MIXED_PRECISION" \
      --device "${DEVICE:-auto}" \
      "${cache_args[@]}"
    semantic_args+=(--siglip_cache_dir "$SIGLIP_CACHE_DIR")
  fi
  semantic_args+=(
    --siglip_model "$SIGLIP_MODEL"
    --semantic_channels "$SEMANTIC_CHANNELS"
    --semantic_attention_heads "$SEMANTIC_ATTENTION_HEADS"
    --semantic_layers "$SEMANTIC_LAYERS"
    --semantic_dropout "$SEMANTIC_DROPOUT"
  )
fi

python train.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mappings_dir "$MAPPINGS_DIR" \
  --views "$VIEWS" \
  --parser_mode "$PARSER_MODE" \
  --max_samples "$MAX_SAMPLES" \
  --base_channels "$BASE_CHANNELS" \
  --feature_dropout "$FEATURE_DROPOUT" \
  "${semantic_args[@]}" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --lr_schedule "$LR_SCHEDULE" \
  --min_lr_ratio "$MIN_LR_RATIO" \
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
  --lambda_outer_false_positive "$LAMBDA_OUTER_FALSE_POSITIVE" \
  --lambda_outer_false_negative "$LAMBDA_OUTER_FALSE_NEGATIVE" \
  --lambda_route_confidence "$LAMBDA_ROUTE_CONFIDENCE" \
  --lambda_semantic_presence "$LAMBDA_SEMANTIC_PRESENCE" \
  --lambda_semantic_coverage "$LAMBDA_SEMANTIC_COVERAGE" \
  --outer_false_positive_gamma "$OUTER_FALSE_POSITIVE_GAMMA" \
  --outer_false_negative_gamma "$OUTER_FALSE_NEGATIVE_GAMMA" \
  --route_class_weight_floor "$ROUTE_CLASS_WEIGHT_FLOOR" \
  --route_outer_class_weight_cap "$ROUTE_OUTER_CLASS_WEIGHT_CAP" \
  --lambda_soft_uv_rgb "$LAMBDA_SOFT_UV_RGB" \
  --lambda_soft_uv_alpha "$LAMBDA_SOFT_UV_ALPHA" \
  --lambda_soft_uv_inner_recall "$LAMBDA_SOFT_UV_INNER_RECALL" \
  --lambda_soft_uv_outer_recall "$LAMBDA_SOFT_UV_OUTER_RECALL" \
  --soft_uv_recall_hard_fraction "$SOFT_UV_RECALL_HARD_FRACTION" \
  --soft_uv_recall_hard_weight "$SOFT_UV_RECALL_HARD_WEIGHT" \
  --lambda_render_rgb "$LAMBDA_RENDER_RGB" \
  --lambda_render_alpha "$LAMBDA_RENDER_ALPHA" \
  --outer_selection_precision_weight "$OUTER_SELECTION_PRECISION_WEIGHT" \
  --outer_selection_recall_weight "$OUTER_SELECTION_RECALL_WEIGHT" \
  --outer_selection_iou_weight "$OUTER_SELECTION_IOU_WEIGHT" \
  --inner_selection_recall_weight "$INNER_SELECTION_RECALL_WEIGHT" \
  --render_softmax_temperature "$RENDER_SOFTMAX_TEMPERATURE" \
  --affine_refine_translation_px "$AFFINE_REFINE_TRANSLATION_PX" \
  --affine_refine_scale "$AFFINE_REFINE_SCALE" \
  --route_confidence_threshold "$ROUTE_CONFIDENCE_THRESHOLD" \
  --route_margin_threshold "$ROUTE_MARGIN_THRESHOLD" \
  --outer_route_confidence_threshold "$OUTER_ROUTE_CONFIDENCE_THRESHOLD" \
  --outer_route_margin_threshold "$OUTER_ROUTE_MARGIN_THRESHOLD" \
  --outer_uv_min_coverage "$OUTER_UV_MIN_COVERAGE" \
  --outer_rescue_confidence_threshold "$OUTER_RESCUE_CONFIDENCE_THRESHOLD" \
  --outer_rescue_margin_threshold "$OUTER_RESCUE_MARGIN_THRESHOLD" \
  --outer_rescue_min_coverage "$OUTER_RESCUE_MIN_COVERAGE" \
  --splat_color_aggregation "$SPLAT_COLOR_AGGREGATION" \
  "${augment_args[@]}" \
  "${background_args[@]}" \
  "${semantic_gate_args[@]}" \
  "${affine_refine_args[@]}" \
  "${fallback_args[@]}" \
  "${routing_consensus_args[@]}" \
  "${outer_rescue_args[@]}" \
  "${uv_class_args[@]}" \
  "${cudnn_args[@]}" \
  "${resume_args[@]}"
