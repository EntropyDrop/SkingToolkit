#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Limit PyTorch CPU parallelism; very high thread counts can cause lock contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

RUN_NAME="${RUN_NAME:-inverse_uv_light_v1}"
DATA_DIR="${DATA_DIR:-../skins}"
MAPPINGS_DIR="${MAPPINGS_DIR:-../../github/differentiable_minecraft_renderer/mappings}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
MODEL="${MODEL:-light}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
RESUME="${RESUME:-}"
# Augmentation scales operate on 512×1024 rendered views, not 64×64 UV.
# 0.01 perspective ≈ 5-10px corner shift; 0.005 distortion ≈ 2.5-5px elastic;
# 0.02 translation ≈ 10-20px offset — already meaningful at this resolution.
PERSPECTIVE_SCALE="${PERSPECTIVE_SCALE:-0.01}"
DISTORTION_SCALE="${DISTORTION_SCALE:-0.005}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-0.02}"
LAMBDA_SSIM="${LAMBDA_SSIM:-0.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"

resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args=(--resume "$RESUME" --resume_lr "$LR")
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
  --lambda_edge "$LAMBDA_EDGE" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  ${resume_args[@]+"${resume_args[@]}"}
