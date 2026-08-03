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
VIEWS="${VIEWS:-front_left,back_left}"

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
MAX_SAMPLES="${MAX_SAMPLES:-180000}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.10}"
ROUTE_ROLE_SPATIAL_PRIOR="${ROUTE_ROLE_SPATIAL_PRIOR:-true}"
ROUTE_PRIOR_HEIGHT="${ROUTE_PRIOR_HEIGHT:-32}"
ROUTE_PRIOR_WIDTH="${ROUTE_PRIOR_WIDTH:-16}"
ROUTE_PRIOR_LOGIT_CAP="${ROUTE_PRIOR_LOGIT_CAP:-1.5}"
ROUTE_PRIOR_DROPOUT="${ROUTE_PRIOR_DROPOUT:-0.10}"
SEMANTIC_BACKBONE="${SEMANTIC_BACKBONE:-siglip2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-base-patch16-224}"
SIGLIP_LOCAL_FILES_ONLY="${SIGLIP_LOCAL_FILES_ONLY:-false}"
CACHE_SIGLIP_FEATURES="${CACHE_SIGLIP_FEATURES:-${CACHE_SIGLIP_GLOBALS:-true}}"
SIGLIP_CACHE_SPATIAL="${SIGLIP_CACHE_SPATIAL:-true}"
SIGLIP_CACHE_DIR="${SIGLIP_CACHE_DIR:-cache/semantic_dense_parser_siglip2_spatial_${MAPPINGS_SIZE}_${MAX_SAMPLES}}"
SIGLIP_CACHE_BATCH_SIZE="${SIGLIP_CACHE_BATCH_SIZE:-32}"
TIPSV2_MODEL="${TIPSV2_MODEL:-google/tipsv2-b14}"
TIPSV2_LOCAL_FILES_ONLY="${TIPSV2_LOCAL_FILES_ONLY:-false}"
SEMANTIC_CHANNELS="${SEMANTIC_CHANNELS:-128}"
SEMANTIC_ATTENTION_HEADS="${SEMANTIC_ATTENTION_HEADS:-4}"
SEMANTIC_LAYERS="${SEMANTIC_LAYERS:-1}"
SEMANTIC_DROPOUT="${SEMANTIC_DROPOUT:-0.05}"
SEMANTIC_SPATIAL_CHANNELS="${SEMANTIC_SPATIAL_CHANNELS:-64}"
SEMANTIC_RUNTIME_BATCH_SIZE="${SEMANTIC_RUNTIME_BATCH_SIZE:-32}"
PREDICT_OUTER_UV_OCCUPANCY="${PREDICT_OUTER_UV_OCCUPANCY:-false}"
OUTER_UV_FEATURE_CHANNELS="${OUTER_UV_FEATURE_CHANNELS:-32}"
OUTER_UV_TOPOLOGY_CHANNELS="${OUTER_UV_TOPOLOGY_CHANNELS:-64}"
OUTER_UV_TOPOLOGY_LAYERS="${OUTER_UV_TOPOLOGY_LAYERS:-3}"
OUTER_UV_TOPOLOGY_DROPOUT="${OUTER_UV_TOPOLOGY_DROPOUT:-0.05}"
OUTER_UV_ROUTE_EVIDENCE_DROPOUT="${OUTER_UV_ROUTE_EVIDENCE_DROPOUT:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-1234}"
REPRODUCIBLE="${REPRODUCIBLE:-false}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-false}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-true}"
LOG_EVERY="${LOG_EVERY:-50}"
BEST_METRIC="${BEST_METRIC:-loss_hard_uv_color_selection}"

if [[ "$REPRODUCIBLE" == "true" ]]; then
  export PYTHONHASHSEED="$SEED"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

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
BACKGROUND_COLOR_TOLERANCE="${BACKGROUND_COLOR_TOLERANCE:-0.25}"
COLOR_BACKGROUND_TOLERANCE="${COLOR_BACKGROUND_TOLERANCE:-0.031372549}"
COLOR_FOREGROUND_INSET="${COLOR_FOREGROUND_INSET:-1}"
OUTER_ROUTE_CONFIDENCE_THRESHOLD="${OUTER_ROUTE_CONFIDENCE_THRESHOLD:-0.80}"
OUTER_ROUTE_MARGIN_THRESHOLD="${OUTER_ROUTE_MARGIN_THRESHOLD:-0.55}"
OUTER_UV_MIN_COVERAGE="${OUTER_UV_MIN_COVERAGE:-0.25}"
OUTER_UV_MIN_SOURCE_PIXELS="${OUTER_UV_MIN_SOURCE_PIXELS:-15}"
OUTER_SILHOUETTE_CONSISTENCY="${OUTER_SILHOUETTE_CONSISTENCY:-true}"
OUTER_SILHOUETTE_MIN_COVERAGE="${OUTER_SILHOUETTE_MIN_COVERAGE:-0.50}"
OUTER_SILHOUETTE_DILATION="${OUTER_SILHOUETTE_DILATION:-1}"
OUTER_SILHOUETTE_MIN_PIXELS="${OUTER_SILHOUETTE_MIN_PIXELS:-4}"
OUTER_GEOMETRY_RESCUE="${OUTER_GEOMETRY_RESCUE:-true}"
OUTER_RESCUE_CONFIDENCE_THRESHOLD="${OUTER_RESCUE_CONFIDENCE_THRESHOLD:-0.60}"
OUTER_RESCUE_MARGIN_THRESHOLD="${OUTER_RESCUE_MARGIN_THRESHOLD:-0.25}"
OUTER_RESCUE_MIN_COVERAGE="${OUTER_RESCUE_MIN_COVERAGE:-0.10}"
GEOMETRY_ROUTE_TEXEL_CONSENSUS="${GEOMETRY_ROUTE_TEXEL_CONSENSUS:-true}"
GEOMETRY_ROUTE_TEXEL_CONSENSUS_WEIGHT="${GEOMETRY_ROUTE_TEXEL_CONSENSUS_WEIGHT:-0.60}"
GEOMETRY_ROUTE_PRESERVE_OUTER_CONFIDENCE="${GEOMETRY_ROUTE_PRESERVE_OUTER_CONFIDENCE:-0.80}"
GEOMETRY_ROUTE_PRESERVE_OUTER_MARGIN="${GEOMETRY_ROUTE_PRESERVE_OUTER_MARGIN:-0.35}"
GEOMETRY_ROUTE_CONSENSUS_OUTER_CONFIDENCE="${GEOMETRY_ROUTE_CONSENSUS_OUTER_CONFIDENCE:-0.70}"
GEOMETRY_ROUTE_CONSENSUS_OUTER_MARGIN="${GEOMETRY_ROUTE_CONSENSUS_OUTER_MARGIN:-0.20}"
GEOMETRY_CROSS_VIEW_OUTER_CONSISTENCY="${GEOMETRY_CROSS_VIEW_OUTER_CONSISTENCY:-false}"
GEOMETRY_CROSS_VIEW_OUTER_WEIGHT="${GEOMETRY_CROSS_VIEW_OUTER_WEIGHT:-0.50}"
GEOMETRY_CROSS_VIEW_OUTER_POSITIVE_CONFIDENCE="${GEOMETRY_CROSS_VIEW_OUTER_POSITIVE_CONFIDENCE:-0.70}"
GEOMETRY_CROSS_VIEW_OUTER_POSITIVE_MARGIN="${GEOMETRY_CROSS_VIEW_OUTER_POSITIVE_MARGIN:-0.20}"
GEOMETRY_CROSS_VIEW_OUTER_NEGATIVE_CONFIDENCE="${GEOMETRY_CROSS_VIEW_OUTER_NEGATIVE_CONFIDENCE:-0.70}"
GEOMETRY_CROSS_VIEW_OUTER_NEGATIVE_MARGIN="${GEOMETRY_CROSS_VIEW_OUTER_NEGATIVE_MARGIN:-0.20}"
GEOMETRY_CROSS_VIEW_OUTER_BACKGROUND_MAX_COVERAGE="${GEOMETRY_CROSS_VIEW_OUTER_BACKGROUND_MAX_COVERAGE:-0.25}"
GEOMETRY_CROSS_VIEW_OUTER_MIN_VIEWS="${GEOMETRY_CROSS_VIEW_OUTER_MIN_VIEWS:-2}"
OUTER_UV_OCCUPANCY_BLEND_WEIGHT="${OUTER_UV_OCCUPANCY_BLEND_WEIGHT:-0.0}"
OUTER_UV_OCCUPANCY_GATE_THRESHOLD="${OUTER_UV_OCCUPANCY_GATE_THRESHOLD:-0.15}"
OUTER_UV_OCCUPANCY_RESCUE_THRESHOLD="${OUTER_UV_OCCUPANCY_RESCUE_THRESHOLD:-0.70}"
OUTER_UV_OCCUPANCY_RESCUE_ROUTE_THRESHOLD="${OUTER_UV_OCCUPANCY_RESCUE_ROUTE_THRESHOLD:-0.30}"
OUTER_UV_OCCUPANCY_ROUTING="${OUTER_UV_OCCUPANCY_ROUTING:-false}"
OUTER_UV_COMPONENT_ROUTING="${OUTER_UV_COMPONENT_ROUTING:-false}"
OUTER_UV_COMPONENT_SEED_THRESHOLD="${OUTER_UV_COMPONENT_SEED_THRESHOLD:-0.80}"
OUTER_UV_COMPONENT_GROW_THRESHOLD="${OUTER_UV_COMPONENT_GROW_THRESHOLD:-0.50}"
OUTER_UV_COMPONENT_MIN_SIZE="${OUTER_UV_COMPONENT_MIN_SIZE:-2}"
SPLAT_COLOR_AGGREGATION="${SPLAT_COLOR_AGGREGATION:-grid_mode}"
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
LAMBDA_OUTER_FALSE_POSITIVE="${LAMBDA_OUTER_FALSE_POSITIVE:-1.0}"
LAMBDA_OUTER_FALSE_NEGATIVE="${LAMBDA_OUTER_FALSE_NEGATIVE:-0.75}"
LAMBDA_ROUTE_CONFIDENCE="${LAMBDA_ROUTE_CONFIDENCE:-0.25}"
LAMBDA_PRIMARY_ROUTE_SWAP="${LAMBDA_PRIMARY_ROUTE_SWAP:-1.0}"
LAMBDA_ROUTE_TEXEL_CONSISTENCY="${LAMBDA_ROUTE_TEXEL_CONSISTENCY:-0.25}"
LAMBDA_ROUTE_TEXEL_SUPERVISION="${LAMBDA_ROUTE_TEXEL_SUPERVISION:-0.0}"
LAMBDA_CROSS_VIEW_OUTER_VISIBILITY="${LAMBDA_CROSS_VIEW_OUTER_VISIBILITY:-0.0}"
CROSS_VIEW_OUTER_CONSISTENCY_LOSS_WEIGHT="${CROSS_VIEW_OUTER_CONSISTENCY_LOSS_WEIGHT:-0.25}"
OUTER_VISIBILITY_HARD_NEGATIVE_FRACTION="${OUTER_VISIBILITY_HARD_NEGATIVE_FRACTION:-0.20}"
OUTER_VISIBILITY_HARD_NEGATIVE_WEIGHT="${OUTER_VISIBILITY_HARD_NEGATIVE_WEIGHT:-0.75}"
ROUTE_TEXEL_CENTER_POWER="${ROUTE_TEXEL_CENTER_POWER:-2.0}"
LAMBDA_ROUTE_PRIOR_REGULARIZATION="${LAMBDA_ROUTE_PRIOR_REGULARIZATION:-0.001}"
LAMBDA_SEMANTIC_PRESENCE="${LAMBDA_SEMANTIC_PRESENCE:-0.25}"
LAMBDA_SEMANTIC_COVERAGE="${LAMBDA_SEMANTIC_COVERAGE:-0.25}"
LAMBDA_OUTER_UV_OCCUPANCY="${LAMBDA_OUTER_UV_OCCUPANCY:-0.0}"
OUTER_UV_OCCUPANCY_DICE_WEIGHT="${OUTER_UV_OCCUPANCY_DICE_WEIGHT:-0.50}"
OUTER_UV_OCCUPANCY_POSITIVE_BALANCE="${OUTER_UV_OCCUPANCY_POSITIVE_BALANCE:-0.60}"
OUTER_HARD_POSITIVE_FRACTION="${OUTER_HARD_POSITIVE_FRACTION:-0.25}"
OUTER_HARD_POSITIVE_WEIGHT="${OUTER_HARD_POSITIVE_WEIGHT:-0.50}"
OUTER_HARD_NEGATIVE_FRACTION="${OUTER_HARD_NEGATIVE_FRACTION:-0.02}"
OUTER_HARD_NEGATIVE_WEIGHT="${OUTER_HARD_NEGATIVE_WEIGHT:-0.25}"
LAMBDA_OUTER_COMPONENT_RECALL="${LAMBDA_OUTER_COMPONENT_RECALL:-0.0}"
LAMBDA_OUTER_COMPONENT_FALSE_POSITIVE="${LAMBDA_OUTER_COMPONENT_FALSE_POSITIVE:-0.0}"
LAMBDA_OUTER_TOPOLOGY="${LAMBDA_OUTER_TOPOLOGY:-0.0}"
LAMBDA_OUTER_NEGATIVE_TOPOLOGY="${LAMBDA_OUTER_NEGATIVE_TOPOLOGY:-0.0}"
LAMBDA_ROUTE_OCCUPANCY_AGREEMENT="${LAMBDA_ROUTE_OCCUPANCY_AGREEMENT:-0.0}"
OUTER_OCCUPANCY_AGREEMENT_WARMUP_FRACTION="${OUTER_OCCUPANCY_AGREEMENT_WARMUP_FRACTION:-0.25}"
OUTER_OCCUPANCY_AGREEMENT_CONFIDENCE_THRESHOLD="${OUTER_OCCUPANCY_AGREEMENT_CONFIDENCE_THRESHOLD:-0.80}"
OUTER_FALSE_POSITIVE_GAMMA="${OUTER_FALSE_POSITIVE_GAMMA:-3.0}"
OUTER_FALSE_NEGATIVE_GAMMA="${OUTER_FALSE_NEGATIVE_GAMMA:-2.0}"
PRIMARY_ROUTE_SWAP_GAMMA="${PRIMARY_ROUTE_SWAP_GAMMA:-2.0}"
ROUTE_PRIOR_TV_WEIGHT="${ROUTE_PRIOR_TV_WEIGHT:-1.0}"
ROUTE_CLASS_WEIGHT_FLOOR="${ROUTE_CLASS_WEIGHT_FLOOR:-0.75}"
ROUTE_OUTER_CLASS_WEIGHT_CAP="${ROUTE_OUTER_CLASS_WEIGHT_CAP:-0.90}"
LAMBDA_SOFT_UV_RGB="${LAMBDA_SOFT_UV_RGB:-0.25}"
LAMBDA_SOFT_UV_ALPHA="${LAMBDA_SOFT_UV_ALPHA:-0.35}"
LAMBDA_SOFT_UV_INNER_RECALL="${LAMBDA_SOFT_UV_INNER_RECALL:-0.50}"
LAMBDA_SOFT_UV_OUTER_RECALL="${LAMBDA_SOFT_UV_OUTER_RECALL:-0.50}"
SOFT_UV_RECALL_HARD_FRACTION="${SOFT_UV_RECALL_HARD_FRACTION:-0.10}"
SOFT_UV_RECALL_HARD_WEIGHT="${SOFT_UV_RECALL_HARD_WEIGHT:-0.50}"
LAMBDA_RENDER_RGB="${LAMBDA_RENDER_RGB:-0.20}"
LAMBDA_RENDER_ALPHA="${LAMBDA_RENDER_ALPHA:-0.25}"
LAMBDA_OUTER_PROJECTION_FALSE_POSITIVE="${LAMBDA_OUTER_PROJECTION_FALSE_POSITIVE:-0.0}"
LAMBDA_OUTER_PROJECTION_FALSE_NEGATIVE="${LAMBDA_OUTER_PROJECTION_FALSE_NEGATIVE:-0.0}"
LAMBDA_OUTER_PROJECTION_DICE="${LAMBDA_OUTER_PROJECTION_DICE:-0.0}"
LAMBDA_OUTER_PROJECTED_AREA="${LAMBDA_OUTER_PROJECTED_AREA:-0.0}"
OUTER_SELECTION_PRECISION_WEIGHT="${OUTER_SELECTION_PRECISION_WEIGHT:-1.5}"
OUTER_SELECTION_RECALL_WEIGHT="${OUTER_SELECTION_RECALL_WEIGHT:-0.5}"
OUTER_SELECTION_IOU_WEIGHT="${OUTER_SELECTION_IOU_WEIGHT:-0.5}"
INNER_SELECTION_RECALL_WEIGHT="${INNER_SELECTION_RECALL_WEIGHT:-0.5}"
HARD_RGB_SELECTION_WEIGHT="${HARD_RGB_SELECTION_WEIGHT:-1.0}"
OUTER_PROJECTION_FP_SELECTION_WEIGHT="${OUTER_PROJECTION_FP_SELECTION_WEIGHT:-0.0}"
OUTER_PROJECTION_AREA_SELECTION_WEIGHT="${OUTER_PROJECTION_AREA_SELECTION_WEIGHT:-0.0}"
RENDER_SOFTMAX_TEMPERATURE="${RENDER_SOFTMAX_TEMPERATURE:-1.0}"
UV_CLASSIFICATION="${UV_CLASSIFICATION:-true}"

route_prior_args=()
if [[ "$ROUTE_ROLE_SPATIAL_PRIOR" == "true" ]]; then
  route_prior_args=(--route_role_spatial_prior)
else
  route_prior_args=(--no_route_role_spatial_prior)
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
outer_silhouette_args=()
if [[ "$OUTER_SILHOUETTE_CONSISTENCY" == "true" ]]; then
  outer_silhouette_args=(--outer_silhouette_consistency)
else
  outer_silhouette_args=(--no_outer_silhouette_consistency)
fi
cross_view_outer_args=()
if [[ "$GEOMETRY_CROSS_VIEW_OUTER_CONSISTENCY" == "true" ]]; then
  cross_view_outer_args=(--geometry_cross_view_outer_consistency)
else
  cross_view_outer_args=(--no_geometry_cross_view_outer_consistency)
fi
outer_occupancy_head_args=()
if [[ "$PREDICT_OUTER_UV_OCCUPANCY" == "true" ]]; then
  outer_occupancy_head_args=(--predict_outer_uv_occupancy)
else
  outer_occupancy_head_args=(--no_predict_outer_uv_occupancy)
fi
outer_occupancy_routing_args=()
if [[ "$OUTER_UV_OCCUPANCY_ROUTING" == "true" ]]; then
  outer_occupancy_routing_args=(--outer_uv_occupancy_routing)
else
  outer_occupancy_routing_args=(--no_outer_uv_occupancy_routing)
fi
outer_component_routing_args=()
if [[ "$OUTER_UV_COMPONENT_ROUTING" == "true" ]]; then
  outer_component_routing_args=(--outer_uv_component_routing)
else
  outer_component_routing_args=(--no_outer_uv_component_routing)
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
reproducibility_args=()
if [[ "$REPRODUCIBLE" == "true" ]]; then
  reproducibility_args=(--reproducible)
else
  reproducibility_args=(--no_reproducible)
fi
if [[ "$STRICT_DETERMINISM" == "true" ]]; then
  reproducibility_args+=(--strict_determinism)
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
  if [[ "$SIGLIP_CACHE_SPATIAL" == "true" ]]; then
    cache_args+=(--spatial)
  fi
  if [[ "$CACHE_SIGLIP_FEATURES" == "true" ]]; then
    python cache_semantic_features.py \
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
    if [[ "$SIGLIP_CACHE_SPATIAL" == "true" ]]; then
      semantic_args+=(--siglip_cache_require_spatial)
    fi
  fi
  semantic_args+=(--siglip_model "$SIGLIP_MODEL")
elif [[ "$SEMANTIC_BACKBONE" == "tipsv2" ]]; then
  semantic_args+=(--tipsv2_model "$TIPSV2_MODEL")
  if [[ "$TIPSV2_LOCAL_FILES_ONLY" == "true" ]]; then
    semantic_args+=(--tipsv2_local_files_only)
  fi
fi
if [[ "$SEMANTIC_BACKBONE" != "none" ]]; then
  semantic_args+=(
    --semantic_channels "$SEMANTIC_CHANNELS"
    --semantic_attention_heads "$SEMANTIC_ATTENTION_HEADS"
    --semantic_layers "$SEMANTIC_LAYERS"
    --semantic_dropout "$SEMANTIC_DROPOUT"
    --semantic_spatial_channels "$SEMANTIC_SPATIAL_CHANNELS"
    --semantic_runtime_batch_size "$SEMANTIC_RUNTIME_BATCH_SIZE"
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
  --route_prior_height "$ROUTE_PRIOR_HEIGHT" \
  --route_prior_width "$ROUTE_PRIOR_WIDTH" \
  --route_prior_logit_cap "$ROUTE_PRIOR_LOGIT_CAP" \
  --route_prior_dropout "$ROUTE_PRIOR_DROPOUT" \
  --outer_uv_feature_channels "$OUTER_UV_FEATURE_CHANNELS" \
  --outer_uv_topology_channels "$OUTER_UV_TOPOLOGY_CHANNELS" \
  --outer_uv_topology_layers "$OUTER_UV_TOPOLOGY_LAYERS" \
  --outer_uv_topology_dropout "$OUTER_UV_TOPOLOGY_DROPOUT" \
  --outer_uv_route_evidence_dropout "$OUTER_UV_ROUTE_EVIDENCE_DROPOUT" \
  "${semantic_args[@]}" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
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
  --lambda_primary_route_swap "$LAMBDA_PRIMARY_ROUTE_SWAP" \
  --lambda_route_texel_consistency "$LAMBDA_ROUTE_TEXEL_CONSISTENCY" \
  --lambda_route_texel_supervision "$LAMBDA_ROUTE_TEXEL_SUPERVISION" \
  --lambda_cross_view_outer_visibility "$LAMBDA_CROSS_VIEW_OUTER_VISIBILITY" \
  --cross_view_outer_consistency_loss_weight "$CROSS_VIEW_OUTER_CONSISTENCY_LOSS_WEIGHT" \
  --outer_visibility_hard_negative_fraction "$OUTER_VISIBILITY_HARD_NEGATIVE_FRACTION" \
  --outer_visibility_hard_negative_weight "$OUTER_VISIBILITY_HARD_NEGATIVE_WEIGHT" \
  --route_texel_center_power "$ROUTE_TEXEL_CENTER_POWER" \
  --lambda_route_prior_regularization "$LAMBDA_ROUTE_PRIOR_REGULARIZATION" \
  --lambda_semantic_presence "$LAMBDA_SEMANTIC_PRESENCE" \
  --lambda_semantic_coverage "$LAMBDA_SEMANTIC_COVERAGE" \
  --lambda_outer_uv_occupancy "$LAMBDA_OUTER_UV_OCCUPANCY" \
  --outer_uv_occupancy_dice_weight "$OUTER_UV_OCCUPANCY_DICE_WEIGHT" \
  --outer_uv_occupancy_positive_balance "$OUTER_UV_OCCUPANCY_POSITIVE_BALANCE" \
  --outer_hard_positive_fraction "$OUTER_HARD_POSITIVE_FRACTION" \
  --outer_hard_positive_weight "$OUTER_HARD_POSITIVE_WEIGHT" \
  --outer_hard_negative_fraction "$OUTER_HARD_NEGATIVE_FRACTION" \
  --outer_hard_negative_weight "$OUTER_HARD_NEGATIVE_WEIGHT" \
  --lambda_outer_component_recall "$LAMBDA_OUTER_COMPONENT_RECALL" \
  --lambda_outer_component_false_positive "$LAMBDA_OUTER_COMPONENT_FALSE_POSITIVE" \
  --lambda_outer_topology "$LAMBDA_OUTER_TOPOLOGY" \
  --lambda_outer_negative_topology "$LAMBDA_OUTER_NEGATIVE_TOPOLOGY" \
  --lambda_route_occupancy_agreement "$LAMBDA_ROUTE_OCCUPANCY_AGREEMENT" \
  --outer_occupancy_agreement_warmup_fraction "$OUTER_OCCUPANCY_AGREEMENT_WARMUP_FRACTION" \
  --outer_occupancy_agreement_confidence_threshold "$OUTER_OCCUPANCY_AGREEMENT_CONFIDENCE_THRESHOLD" \
  --outer_false_positive_gamma "$OUTER_FALSE_POSITIVE_GAMMA" \
  --outer_false_negative_gamma "$OUTER_FALSE_NEGATIVE_GAMMA" \
  --primary_route_swap_gamma "$PRIMARY_ROUTE_SWAP_GAMMA" \
  --route_prior_tv_weight "$ROUTE_PRIOR_TV_WEIGHT" \
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
  --lambda_outer_projection_false_positive "$LAMBDA_OUTER_PROJECTION_FALSE_POSITIVE" \
  --lambda_outer_projection_false_negative "$LAMBDA_OUTER_PROJECTION_FALSE_NEGATIVE" \
  --lambda_outer_projection_dice "$LAMBDA_OUTER_PROJECTION_DICE" \
  --lambda_outer_projected_area "$LAMBDA_OUTER_PROJECTED_AREA" \
  --outer_selection_precision_weight "$OUTER_SELECTION_PRECISION_WEIGHT" \
  --outer_selection_recall_weight "$OUTER_SELECTION_RECALL_WEIGHT" \
  --outer_selection_iou_weight "$OUTER_SELECTION_IOU_WEIGHT" \
  --inner_selection_recall_weight "$INNER_SELECTION_RECALL_WEIGHT" \
  --hard_rgb_selection_weight "$HARD_RGB_SELECTION_WEIGHT" \
  --outer_projection_fp_selection_weight "$OUTER_PROJECTION_FP_SELECTION_WEIGHT" \
  --outer_projection_area_selection_weight "$OUTER_PROJECTION_AREA_SELECTION_WEIGHT" \
  --render_softmax_temperature "$RENDER_SOFTMAX_TEMPERATURE" \
  --affine_refine_translation_px "$AFFINE_REFINE_TRANSLATION_PX" \
  --affine_refine_scale "$AFFINE_REFINE_SCALE" \
  --route_confidence_threshold "$ROUTE_CONFIDENCE_THRESHOLD" \
  --route_margin_threshold "$ROUTE_MARGIN_THRESHOLD" \
  --background_color_tolerance "$BACKGROUND_COLOR_TOLERANCE" \
  --color_background_tolerance "$COLOR_BACKGROUND_TOLERANCE" \
  --color_foreground_inset "$COLOR_FOREGROUND_INSET" \
  --outer_route_confidence_threshold "$OUTER_ROUTE_CONFIDENCE_THRESHOLD" \
  --outer_route_margin_threshold "$OUTER_ROUTE_MARGIN_THRESHOLD" \
  --outer_uv_min_coverage "$OUTER_UV_MIN_COVERAGE" \
  --outer_uv_min_source_pixels "$OUTER_UV_MIN_SOURCE_PIXELS" \
  --outer_silhouette_min_coverage "$OUTER_SILHOUETTE_MIN_COVERAGE" \
  --outer_silhouette_dilation "$OUTER_SILHOUETTE_DILATION" \
  --outer_silhouette_min_pixels "$OUTER_SILHOUETTE_MIN_PIXELS" \
  --outer_rescue_confidence_threshold "$OUTER_RESCUE_CONFIDENCE_THRESHOLD" \
  --outer_rescue_margin_threshold "$OUTER_RESCUE_MARGIN_THRESHOLD" \
  --outer_rescue_min_coverage "$OUTER_RESCUE_MIN_COVERAGE" \
  --geometry_route_texel_consensus_weight "$GEOMETRY_ROUTE_TEXEL_CONSENSUS_WEIGHT" \
  --geometry_route_preserve_outer_confidence "$GEOMETRY_ROUTE_PRESERVE_OUTER_CONFIDENCE" \
  --geometry_route_preserve_outer_margin "$GEOMETRY_ROUTE_PRESERVE_OUTER_MARGIN" \
  --geometry_route_consensus_outer_confidence "$GEOMETRY_ROUTE_CONSENSUS_OUTER_CONFIDENCE" \
  --geometry_route_consensus_outer_margin "$GEOMETRY_ROUTE_CONSENSUS_OUTER_MARGIN" \
  --geometry_cross_view_outer_weight "$GEOMETRY_CROSS_VIEW_OUTER_WEIGHT" \
  --geometry_cross_view_outer_positive_confidence "$GEOMETRY_CROSS_VIEW_OUTER_POSITIVE_CONFIDENCE" \
  --geometry_cross_view_outer_positive_margin "$GEOMETRY_CROSS_VIEW_OUTER_POSITIVE_MARGIN" \
  --geometry_cross_view_outer_negative_confidence "$GEOMETRY_CROSS_VIEW_OUTER_NEGATIVE_CONFIDENCE" \
  --geometry_cross_view_outer_negative_margin "$GEOMETRY_CROSS_VIEW_OUTER_NEGATIVE_MARGIN" \
  --geometry_cross_view_outer_background_max_coverage "$GEOMETRY_CROSS_VIEW_OUTER_BACKGROUND_MAX_COVERAGE" \
  --geometry_cross_view_outer_min_views "$GEOMETRY_CROSS_VIEW_OUTER_MIN_VIEWS" \
  --outer_uv_occupancy_blend_weight "$OUTER_UV_OCCUPANCY_BLEND_WEIGHT" \
  --outer_uv_occupancy_gate_threshold "$OUTER_UV_OCCUPANCY_GATE_THRESHOLD" \
  --outer_uv_occupancy_rescue_threshold "$OUTER_UV_OCCUPANCY_RESCUE_THRESHOLD" \
  --outer_uv_occupancy_rescue_route_threshold "$OUTER_UV_OCCUPANCY_RESCUE_ROUTE_THRESHOLD" \
  --outer_uv_component_seed_threshold "$OUTER_UV_COMPONENT_SEED_THRESHOLD" \
  --outer_uv_component_grow_threshold "$OUTER_UV_COMPONENT_GROW_THRESHOLD" \
  --outer_uv_component_min_size "$OUTER_UV_COMPONENT_MIN_SIZE" \
  --splat_color_aggregation "$SPLAT_COLOR_AGGREGATION" \
  "${route_prior_args[@]}" \
  "${background_args[@]}" \
  "${semantic_gate_args[@]}" \
  "${affine_refine_args[@]}" \
  "${fallback_args[@]}" \
  "${routing_consensus_args[@]}" \
  "${outer_silhouette_args[@]}" \
  "${cross_view_outer_args[@]}" \
  "${outer_occupancy_head_args[@]}" \
  "${outer_occupancy_routing_args[@]}" \
  "${outer_component_routing_args[@]}" \
  "${outer_rescue_args[@]}" \
  "${uv_class_args[@]}" \
  "${cudnn_args[@]}" \
  "${reproducibility_args[@]}" \
  "${resume_args[@]}"
