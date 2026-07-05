#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Limit PyTorch CPU parallelism; very high thread counts can cause lock contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

if [[ -z "${RUN_NAME:-}" ]]; then
  v=1
  while [[ -d "runs/foreground_alpha_v${v}" ]]; do
    ((v++))
  done
  RUN_NAME="foreground_alpha_v${v}"
fi

DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
BACKGROUND_MODE="${BACKGROUND_MODE:-random}"
HARD_BG_PROB="${HARD_BG_PROB:-0.3}"
LAMBDA_HOLE="${LAMBDA_HOLE:-1.0}"
LAMBDA_BG="${LAMBDA_BG:-2.0}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"
RESUME="${RESUME:-}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args=(--resume "$RESUME")
fi

python train.py \
  --data_dir "$DATA_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "runs/$RUN_NAME" \
  --views walk_front_both_layer_ortho,walk_back_both_layer_ortho \
  --mappings_dir "$MAPPINGS_DIR" \
  --background_mode "$BACKGROUND_MODE" \
  --hard_bg_prob "$HARD_BG_PROB" \
  --lambda_hole "$LAMBDA_HOLE" \
  --lambda_bg "$LAMBDA_BG" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --val_split "$VAL_SPLIT" \
  --save_every 1 \
  --preview_every 1 \
  --mixed_precision "$MIXED_PRECISION" \
  ${resume_args[@]+"${resume_args[@]}"}