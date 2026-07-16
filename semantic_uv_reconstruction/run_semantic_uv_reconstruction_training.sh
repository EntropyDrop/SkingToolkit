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

  # Support both layouts commonly used by this project:
  #   llms/{SkingToolkit,differentiable_minecraft_renderer}
  #   llms/SkingToolkit/{semantic_uv_reconstruction,differentiable_minecraft_renderer}
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
  echo "Locate it with: find ~/llms -type d -name '$name'" >&2
  echo "Then run: MAPPINGS_DIR=/absolute/path/to/$name ./run_semantic_uv_reconstruction_training.sh" >&2
  echo "If mappings have not been generated, run generate_mappings.py in differentiable_minecraft_renderer first." >&2
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
  echo "Generate them with: python generate_mappings.py --views '$VIEWS' --sizes '$MAPPINGS_SIZE'" >&2
  exit 1
fi

MAX_SAMPLES="${MAX_SAMPLES:-}"
SEMANTIC_LABELS_DIR="${SEMANTIC_LABELS_DIR:-}"
SEMANTIC_CLASSES="${SEMANTIC_CLASSES:-13}"
SEMANTIC_BACKBONE="${SEMANTIC_BACKBONE:-siglip2}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip2-base-patch16-224}"
SIGLIP_LOCAL_FILES_ONLY="${SIGLIP_LOCAL_FILES_ONLY:-false}"
CACHE_SIGLIP_GLOBALS="${CACHE_SIGLIP_GLOBALS:-true}"
SIGLIP_CACHE_DIR="${SIGLIP_CACHE_DIR:-cache/siglip2_base_patch16_224_${MAPPINGS_SIZE}}"
SIGLIP_CACHE_BATCH_SIZE="${SIGLIP_CACHE_BATCH_SIZE:-32}"
USE_SIGLIP_PATCH_TOKENS="${USE_SIGLIP_PATCH_TOKENS:-false}"

BASE_CHANNELS="${BASE_CHANNELS:-32}"
TOKEN_CHANNELS="${TOKEN_CHANNELS:-128}"
QUERY_SIZE="${QUERY_SIZE:-32}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
ATTENTION_LAYERS="${ATTENTION_LAYERS:-2}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
MEMORY_LATENTS="${MEMORY_LATENTS:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-4}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEVICE="${DEVICE:-auto}"
SIGLIP_RENDER_EVERY="${SIGLIP_RENDER_EVERY:-4}"
SIGLIP_RENDER_WARMUP_EPOCHS="${SIGLIP_RENDER_WARMUP_EPOCHS:-2}"
RGB_WARMUP_EPOCHS="${RGB_WARMUP_EPOCHS:-2}"
RGB_WARMUP_MULTIPLIER="${RGB_WARMUP_MULTIPLIER:-2.0}"
LOG_EVERY="${LOG_EVERY:-50}"
FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-true}"
TORCH_COMPILE="${TORCH_COMPILE:-true}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune-no-cudagraphs}"

LAMBDA_UV_RGB="${LAMBDA_UV_RGB:-2.0}"
LAMBDA_UV_EDGE="${LAMBDA_UV_EDGE:-1.0}"
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
if [[ "$USE_SIGLIP_PATCH_TOKENS" == "true" ]]; then
  optional_args+=(--use_siglip_patch_tokens)
fi
if [[ "$FUSED_OPTIMIZER" != "true" ]]; then
  optional_args+=(--no_fused_optimizer)
fi
if [[ "$TORCH_COMPILE" == "true" ]]; then
  optional_args+=(--compile --compile_mode "$COMPILE_MODE")
else
  optional_args+=(--no-compile)
fi

if [[ "$SEMANTIC_BACKBONE" == "siglip2" && "$CACHE_SIGLIP_GLOBALS" == "true" ]]; then
  if [[ "$USE_SIGLIP_PATCH_TOKENS" == "true" ]]; then
    echo "USE_SIGLIP_PATCH_TOKENS=true is incompatible with the global cache." >&2
    exit 1
  fi
  cache_args=()
  if [[ "$SIGLIP_LOCAL_FILES_ONLY" == "true" ]]; then
    cache_args+=(--siglip_local_files_only)
  fi
  if [[ -n "$MAX_SAMPLES" ]]; then
    cache_args+=(--max_samples "$MAX_SAMPLES")
  fi
  python cache_siglip_globals.py \
    --data_dir "$DATA_DIR" \
    --cache_dir "$SIGLIP_CACHE_DIR" \
    --mappings_dir "$MAPPINGS_DIR" \
    --views "$VIEWS" \
    --siglip_model "$SIGLIP_MODEL" \
    --batch_size "$SIGLIP_CACHE_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH_FACTOR" \
    --mixed_precision "$MIXED_PRECISION" \
    --device "$DEVICE" \
    "${cache_args[@]}"
  optional_args+=(--siglip_cache_dir "$SIGLIP_CACHE_DIR")
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
  --memory_latents "$MEMORY_LATENTS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --min_lr_ratio "$MIN_LR_RATIO" \
  --mixed_precision "$MIXED_PRECISION" \
  --device "$DEVICE" \
  --lambda_uv_rgb "$LAMBDA_UV_RGB" \
  --lambda_uv_edge "$LAMBDA_UV_EDGE" \
  --lambda_outer_alpha "$LAMBDA_OUTER_ALPHA" \
  --lambda_outer_dice "$LAMBDA_OUTER_DICE" \
  --lambda_semantic_uv "$LAMBDA_SEMANTIC_UV" \
  --lambda_semantic_presence "$LAMBDA_SEMANTIC_PRESENCE" \
  --lambda_semantic_coverage "$LAMBDA_SEMANTIC_COVERAGE" \
  --lambda_semantic_color "$LAMBDA_SEMANTIC_COLOR" \
  --lambda_render_rgb "$LAMBDA_RENDER_RGB" \
  --lambda_render_alpha "$LAMBDA_RENDER_ALPHA" \
  --lambda_siglip_render "$LAMBDA_SIGLIP_RENDER" \
  --siglip_render_every "$SIGLIP_RENDER_EVERY" \
  --siglip_render_warmup_epochs "$SIGLIP_RENDER_WARMUP_EPOCHS" \
  --rgb_warmup_epochs "$RGB_WARMUP_EPOCHS" \
  --rgb_warmup_multiplier "$RGB_WARMUP_MULTIPLIER" \
  --log_every "$LOG_EVERY" \
  "${optional_args[@]}"
