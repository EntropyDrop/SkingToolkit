#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Limit PyTorch CPU parallelism; very high thread counts can cause lock contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

find_latest_checkpoint() {
  local root="$1"
  local prefix="$2"
  local checkpoint_name="$3"
  local best_v=-1
  local best_checkpoint=""
  local dir base suffix v

  shopt -s nullglob
  for dir in "$root"/"${prefix}"*; do
    [[ -d "$dir" ]] || continue
    [[ -f "$dir/$checkpoint_name" ]] || continue

    base="$(basename "$dir")"
    suffix="${base#"$prefix"}"
    [[ "$suffix" =~ ^[0-9]+$ ]] || continue

    v=$((10#$suffix))
    if (( v > best_v )); then
      best_v="$v"
      best_checkpoint="$dir/$checkpoint_name"
    fi
  done
  shopt -u nullglob

  printf '%s\n' "$best_checkpoint"
}

COMPLETION_MODEL="${COMPLETION_MODEL:-topology_maskgit}"
MODEL="${MODEL:-$COMPLETION_MODEL}"
if [[ -z "${RUN_NAME:-}" ]]; then
  v=1
  while [[ -d "runs/semantic_uv_reconstruction_${MODEL}_v${v}" ]]; do
    ((v++))
  done
  RUN_NAME="semantic_uv_reconstruction_${MODEL}_v${v}"
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
    if [[ ! -d "$requested" ]]; then
      echo "MAPPINGS_DIR does not exist: $requested" >&2
      return 1
    fi
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
  if [[ -n "$discovered" ]]; then
    MAPPINGS_DIR="$discovered"
    return 0
  fi

  echo "Could not find $name from $(pwd)." >&2
  echo "Set MAPPINGS_DIR=/absolute/path/to/$name." >&2
  return 1
}

resolve_mappings_dir

IFS=',' read -r -a configured_views <<< "$VIEWS"
missing_mapping_files=()
for view in "${configured_views[@]}"; do
  view="${view//[[:space:]]/}"
  if [[ -n "$view" && ! -f "$MAPPINGS_DIR/${view}_mapping.pt" ]]; then
    missing_mapping_files+=("${view}_mapping.pt")
  fi
done
if (( ${#missing_mapping_files[@]} > 0 )); then
  echo "MAPPINGS_DIR=$MAPPINGS_DIR is missing required files:" >&2
  printf '  %s\n' "${missing_mapping_files[@]}" >&2
  exit 1
fi

MAX_SAMPLES="${MAX_SAMPLES:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
RESUME="${RESUME:-}"
RESUME_LR="${RESUME_LR:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-true}"
BEST_METRIC="${BEST_METRIC:-loss_recon_total}"
SCHEDULER="${SCHEDULER:-cosine}"
MIN_LR="${MIN_LR:-1e-5}"
LOG_EVERY="${LOG_EVERY:-50}"
PRESERVE_KNOWN="${PRESERVE_KNOWN:-true}"

# --- Topology-aware masked generator ---
TOPOLOGY_CHANNELS="${TOPOLOGY_CHANNELS:-128}"
TOPOLOGY_LAYERS="${TOPOLOGY_LAYERS:-4}"
TOPOLOGY_ATTENTION_HEADS="${TOPOLOGY_ATTENTION_HEADS:-4}"
TOPOLOGY_DROPOUT="${TOPOLOGY_DROPOUT:-0.05}"
TOPOLOGY_HARD_LOCK_THRESHOLD="${TOPOLOGY_HARD_LOCK_THRESHOLD:-0.0}"
TOPOLOGY_DROP_KNOWN_MIN="${TOPOLOGY_DROP_KNOWN_MIN:-0.1}"
TOPOLOGY_DROP_KNOWN_MAX="${TOPOLOGY_DROP_KNOWN_MAX:-0.5}"
TOPOLOGY_TEACHER_REVEAL_UNKNOWN="${TOPOLOGY_TEACHER_REVEAL_UNKNOWN:-0.1}"
LAMBDA_RGB_TOKEN="${LAMBDA_RGB_TOKEN:-1.0}"
LAMBDA_RGB_DISTRIBUTION="${LAMBDA_RGB_DISTRIBUTION:-2.0}"
LAMBDA_ALPHA_TOKEN="${LAMBDA_ALPHA_TOKEN:-0.5}"
PREVIEW_GENERATION_STEPS="${PREVIEW_GENERATION_STEPS:-4}"
PREVIEW_GENERATION_TEMPERATURE="${PREVIEW_GENERATION_TEMPERATURE:-0.0}"
PREVIEW_RGB_DECODE="${PREVIEW_RGB_DECODE:-mean}"
PREVIEW_PALETTE_SNAP="${PREVIEW_PALETTE_SNAP:-true}"
PREVIEW_PALETTE_MIN_CONFIDENCE="${PREVIEW_PALETTE_MIN_CONFIDENCE:-0.75}"

# --- Dense parser conditioning ---
PARSER_RUNS_DIR="${PARSER_RUNS_DIR:-../dense_uv_parser/runs}"
PARSER_RUN_PREFIX="${PARSER_RUN_PREFIX:-dense_uv_parser_v}"
PARSER_CHECKPOINT_NAME="${PARSER_CHECKPOINT_NAME:-best.pt}"
PARSER_CHECKPOINT="${PARSER_CHECKPOINT:-}"
PARSER_SPLAT_FG_THRESHOLD="${PARSER_SPLAT_FG_THRESHOLD:-0.5}"
PARSER_SEMANTIC_GATE="${PARSER_SEMANTIC_GATE:-true}"
PARSER_AFFINE_REFINE="${PARSER_AFFINE_REFINE:-false}"
PARSER_AFFINE_REFINE_TRANSLATION_PX="${PARSER_AFFINE_REFINE_TRANSLATION_PX:-0.0}"
PARSER_AFFINE_REFINE_SCALE="${PARSER_AFFINE_REFINE_SCALE:-0.0}"
PARSER_ROUTE_CONFIDENCE_THRESHOLD="${PARSER_ROUTE_CONFIDENCE_THRESHOLD:-0.0}"
PARSER_ROUTE_MARGIN_THRESHOLD="${PARSER_ROUTE_MARGIN_THRESHOLD:-0.0}"
PARSER_OUTER_ROUTE_CONFIDENCE_THRESHOLD="${PARSER_OUTER_ROUTE_CONFIDENCE_THRESHOLD:-0.80}"
PARSER_OUTER_ROUTE_MARGIN_THRESHOLD="${PARSER_OUTER_ROUTE_MARGIN_THRESHOLD:-0.55}"
PARSER_OUTER_UV_MIN_COVERAGE="${PARSER_OUTER_UV_MIN_COVERAGE:-0.25}"
PARSER_OUTER_GEOMETRY_RESCUE="${PARSER_OUTER_GEOMETRY_RESCUE:-true}"
PARSER_OUTER_RESCUE_CONFIDENCE_THRESHOLD="${PARSER_OUTER_RESCUE_CONFIDENCE_THRESHOLD:-0.60}"
PARSER_OUTER_RESCUE_MARGIN_THRESHOLD="${PARSER_OUTER_RESCUE_MARGIN_THRESHOLD:-0.25}"
PARSER_OUTER_RESCUE_MIN_COVERAGE="${PARSER_OUTER_RESCUE_MIN_COVERAGE:-0.10}"
PARSER_GEOMETRY_ROUTE_TEXEL_CONSENSUS="${PARSER_GEOMETRY_ROUTE_TEXEL_CONSENSUS:-true}"
PARSER_SPLAT_COLOR_AGGREGATION="${PARSER_SPLAT_COLOR_AGGREGATION:-grid_mode}"
PARSER_ALLOW_SEMANTIC_FALLBACK="${PARSER_ALLOW_SEMANTIC_FALLBACK:-false}"
PARSER_BACKGROUND_AUGMENT="${PARSER_BACKGROUND_AUGMENT:-true}"
PARSER_BACKGROUND_AUGMENT_PROB="${PARSER_BACKGROUND_AUGMENT_PROB:-0.9}"

# --- Geometry augmentation (disabled for fixed-view pixel alignment) ---
AUGMENT="${AUGMENT:-false}"
AUGMENT_VALIDATION="${AUGMENT_VALIDATION:-false}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.0}"
SCALE_RANGE="${SCALE_RANGE:-0.0}"
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
preserve_known_args=()
if [[ "$PRESERVE_KNOWN" == "true" ]]; then
  preserve_known_args=(--preserve_known)
else
  preserve_known_args=(--no_preserve_known)
fi
preview_palette_args=()
if [[ "$PREVIEW_PALETTE_SNAP" == "true" ]]; then
  preview_palette_args=(--preview_palette_snap)
else
  preview_palette_args=(--no_preview_palette_snap)
fi
if [[ -z "$PARSER_CHECKPOINT" ]]; then
  PARSER_CHECKPOINT="$(find_latest_checkpoint "$PARSER_RUNS_DIR" "$PARSER_RUN_PREFIX" "$PARSER_CHECKPOINT_NAME")"
fi
if [[ -z "$PARSER_CHECKPOINT" ]]; then
  echo "No parser checkpoint found under ${PARSER_RUNS_DIR}/${PARSER_RUN_PREFIX}*/${PARSER_CHECKPOINT_NAME}." >&2
  echo "Train one in ../dense_uv_parser first or set PARSER_CHECKPOINT=/path/to/best.pt." >&2
  exit 1
fi
conditioning_args=(
  --parser_checkpoint "$PARSER_CHECKPOINT"
  --parser_splat_fg_threshold "$PARSER_SPLAT_FG_THRESHOLD"
  --parser_affine_refine_translation_px "$PARSER_AFFINE_REFINE_TRANSLATION_PX"
  --parser_affine_refine_scale "$PARSER_AFFINE_REFINE_SCALE"
  --parser_route_confidence_threshold "$PARSER_ROUTE_CONFIDENCE_THRESHOLD"
  --parser_route_margin_threshold "$PARSER_ROUTE_MARGIN_THRESHOLD"
  --parser_outer_route_confidence_threshold "$PARSER_OUTER_ROUTE_CONFIDENCE_THRESHOLD"
  --parser_outer_route_margin_threshold "$PARSER_OUTER_ROUTE_MARGIN_THRESHOLD"
  --parser_outer_uv_min_coverage "$PARSER_OUTER_UV_MIN_COVERAGE"
  --parser_outer_rescue_confidence_threshold "$PARSER_OUTER_RESCUE_CONFIDENCE_THRESHOLD"
  --parser_outer_rescue_margin_threshold "$PARSER_OUTER_RESCUE_MARGIN_THRESHOLD"
  --parser_outer_rescue_min_coverage "$PARSER_OUTER_RESCUE_MIN_COVERAGE"
  --parser_splat_color_aggregation "$PARSER_SPLAT_COLOR_AGGREGATION"
)
if [[ "$PARSER_ALLOW_SEMANTIC_FALLBACK" == "true" ]]; then
  conditioning_args+=(--parser_allow_semantic_fallback)
fi
if [[ "$PARSER_GEOMETRY_ROUTE_TEXEL_CONSENSUS" == "true" ]]; then
  conditioning_args+=(--parser_geometry_route_texel_consensus)
else
  conditioning_args+=(--no_parser_geometry_route_texel_consensus)
fi
if [[ "$PARSER_OUTER_GEOMETRY_RESCUE" == "true" ]]; then
  conditioning_args+=(--parser_outer_geometry_rescue)
else
  conditioning_args+=(--no_parser_outer_geometry_rescue)
fi
if [[ "$PARSER_SEMANTIC_GATE" == "true" ]]; then
  conditioning_args+=(--parser_semantic_gate)
else
  conditioning_args+=(--no_parser_semantic_gate)
fi
if [[ "$PARSER_AFFINE_REFINE" == "true" ]]; then
  conditioning_args+=(--parser_affine_refine)
else
  conditioning_args+=(--no_parser_affine_refine)
fi
parser_background_args=()
if [[ "$PARSER_BACKGROUND_AUGMENT" == "true" ]]; then
  parser_background_args=(
    --parser_background_augment
    --parser_background_augment_prob "$PARSER_BACKGROUND_AUGMENT_PROB"
  )
else
  parser_background_args=(--no_parser_background_augment)
fi

augment_args=()
if [[ "$AUGMENT" == "true" ]]; then
  augment_args=(
    --augment
    --translation_scale "$TRANSLATION_SCALE"
    --scale_range "$SCALE_RANGE"
    --perspective_scale "$PERSPECTIVE_SCALE"
  )
else
  augment_args=(--no_augment)
fi
if [[ "$AUGMENT_VALIDATION" == "true" ]]; then
  augment_args+=(--augment_validation)
else
  augment_args+=(--no_augment_validation)
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
  --completion_model "$COMPLETION_MODEL" \
  --topology_channels "$TOPOLOGY_CHANNELS" \
  --topology_layers "$TOPOLOGY_LAYERS" \
  --topology_attention_heads "$TOPOLOGY_ATTENTION_HEADS" \
  --topology_dropout "$TOPOLOGY_DROPOUT" \
  --topology_hard_lock_threshold "$TOPOLOGY_HARD_LOCK_THRESHOLD" \
  --topology_drop_known_min "$TOPOLOGY_DROP_KNOWN_MIN" \
  --topology_drop_known_max "$TOPOLOGY_DROP_KNOWN_MAX" \
  --topology_teacher_reveal_unknown "$TOPOLOGY_TEACHER_REVEAL_UNKNOWN" \
  --lambda_rgb_token "$LAMBDA_RGB_TOKEN" \
  --lambda_rgb_distribution "$LAMBDA_RGB_DISTRIBUTION" \
  --lambda_alpha_token "$LAMBDA_ALPHA_TOKEN" \
  --preview_generation_steps "$PREVIEW_GENERATION_STEPS" \
  --preview_generation_temperature "$PREVIEW_GENERATION_TEMPERATURE" \
  --preview_rgb_decode "$PREVIEW_RGB_DECODE" \
  --preview_palette_min_confidence "$PREVIEW_PALETTE_MIN_CONFIDENCE" \
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
  ${conditioning_args[@]+"${conditioning_args[@]}"} \
  ${preserve_known_args[@]+"${preserve_known_args[@]}"} \
  ${preview_palette_args[@]+"${preview_palette_args[@]}"} \
  ${parser_background_args[@]+"${parser_background_args[@]}"} \
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
