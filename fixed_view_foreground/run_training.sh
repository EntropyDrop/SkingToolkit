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
  for dir in runs/fixed_view_foreground_v*; do
    [[ -d "$dir" && -f "$dir/latest.pt" ]] || continue
    base="$(basename "$dir")"
    suffix="${base#fixed_view_foreground_v}"
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
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

if [[ -n "$RESUME" && -z "${RUN_NAME:-}" ]]; then
  OUTPUT_DIR="$(dirname "$RESUME")"
  RUN_NAME="$(basename "$OUTPUT_DIR")"
elif [[ -z "${RUN_NAME:-}" ]]; then
  version=1
  while [[ -d "runs/fixed_view_foreground_v${version}" ]]; do
    ((version++))
  done
  RUN_NAME="fixed_view_foreground_v${version}"
  OUTPUT_DIR="runs/$RUN_NAME"
else
  OUTPUT_DIR="runs/$RUN_NAME"
fi

DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_DIR="${MAPPINGS_DIR:-}"
MAPPINGS_SIZE="${MAPPINGS_SIZE:-256x512}"
VIEWS="${VIEWS:-walk_front_both_layer_ortho,walk_back_both_layer_ortho}"
MAX_SAMPLES="${MAX_SAMPLES:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$MAPPINGS_DIR" ]]; then
  mapping_name="mappings_${MAPPINGS_SIZE}"
  for candidate in \
    "../differentiable_minecraft_renderer/$mapping_name" \
    "../../differentiable_minecraft_renderer/$mapping_name" \
    "../../github/differentiable_minecraft_renderer/$mapping_name"; do
    if [[ -d "$candidate" ]]; then
      MAPPINGS_DIR="$candidate"
      break
    fi
  done
fi
if [[ -z "$MAPPINGS_DIR" || ! -d "$MAPPINGS_DIR" ]]; then
  echo "Could not find mappings_${MAPPINGS_SIZE}; set MAPPINGS_DIR explicitly." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

args=(
  train.py
  --data_dir "$DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --views "$VIEWS"
  --max_samples "$MAX_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --mixed_precision "$MIXED_PRECISION"
  --base_channels "$BASE_CHANNELS"
)
args+=(--mappings_dir "$MAPPINGS_DIR")
if [[ -n "$RESUME" ]]; then
  args+=(--resume "$RESUME")
fi

echo "Training fixed-view foreground model: $RUN_NAME"
echo "Log: $OUTPUT_DIR/train.log"
"$PYTHON_BIN" "${args[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
