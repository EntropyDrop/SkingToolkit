#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Limit PyTorch CPU parallelism; very high thread counts can cause lock contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

RUN_NAME="${RUN_NAME:-inverse_uv_test18}"
DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
RESUME="${RESUME:-}"
PERSPECTIVE_SCALE="${PERSPECTIVE_SCALE:-0.01}"
DISTORTION_SCALE="${DISTORTION_SCALE:-0.005}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.02}"
LAMBDA_SSIM="${LAMBDA_SSIM:-0.2}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"

resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args=(--resume "$RESUME" --resume_lr "$LR")
fi

mkdir -p "runs/$RUN_NAME"

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
  --augment \
  --perspective_scale "$PERSPECTIVE_SCALE" \
  --distortion_scale "$DISTORTION_SCALE" \
  --translation_scale "$TRANSLATION_SCALE" \
  --lambda_ssim "$LAMBDA_SSIM" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  ${resume_args[@]+"${resume_args[@]}"} 2>&1 | tee "runs/$RUN_NAME/train.log"
