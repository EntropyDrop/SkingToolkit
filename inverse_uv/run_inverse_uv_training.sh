#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Limit PyTorch CPU parallelism; very high thread counts can cause lock contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

MODEL="${MODEL:-full}"
if [[ -z "${RUN_NAME:-}" ]]; then
  v=1
  while [[ -d "runs/inverse_uv_${MODEL}_v${v}" ]]; do
    ((v++))
  done
  RUN_NAME="inverse_uv_${MODEL}_v${v}"
fi
DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-}"
RESUME="${RESUME:-}"
LAMBDA_ALPHA="${LAMBDA_ALPHA:-0.5}"
LAMBDA_RENDER="${LAMBDA_RENDER:-0.1}"
LAMBDA_SSIM="${LAMBDA_SSIM:-0.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.25}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SUPERVISE_COVERED_INNER="${SUPERVISE_COVERED_INNER:-true}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

resume_args=()
if [[ -n "$RESUME" ]]; then
  # If resuming, use the provided LR if explicitly set, otherwise let python train.py handle it
  if [[ -n "$LR" ]]; then
    resume_args=(--resume "$RESUME" --resume_lr "$LR")
  else
    resume_args=(--resume "$RESUME")
  fi
fi

extra_args=()
if [[ -n "$LR" ]]; then
  extra_args+=(--lr "$LR")
fi
if [[ -n "$LAMBDA_ALPHA" ]]; then
  extra_args+=(--lambda_alpha "$LAMBDA_ALPHA")
fi
if [[ -n "$LAMBDA_RENDER" ]]; then
  extra_args+=(--lambda_render "$LAMBDA_RENDER")
fi
if [[ -n "$LAMBDA_SSIM" ]]; then
  extra_args+=(--lambda_ssim "$LAMBDA_SSIM")
fi
if [[ -n "$LAMBDA_EDGE" ]]; then
  extra_args+=(--lambda_edge "$LAMBDA_EDGE")
fi
if [[ -n "$WARMUP_EPOCHS" ]]; then
  extra_args+=(--warmup_epochs "$WARMUP_EPOCHS")
fi
if [[ "$SUPERVISE_COVERED_INNER" == "true" ]]; then
  extra_args+=(--supervise_covered_inner)
fi

python train.py \
  --data_dir "$DATA_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "runs/$RUN_NAME" \
  --views walk_front_both_layer_ortho,walk_back_both_layer_ortho \
  --model "$MODEL" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --val_split 0.1 \
  --mappings_dir "$MAPPINGS_DIR" \
  --save_every 1 \
  --preview_every 1 \
  --mixed_precision "$MIXED_PRECISION" \
  ${extra_args[@]+"${extra_args[@]}"} \
  ${resume_args[@]+"${resume_args[@]}"}

