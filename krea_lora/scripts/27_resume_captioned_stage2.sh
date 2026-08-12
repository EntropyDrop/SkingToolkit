#!/usr/bin/env bash
set -euo pipefail

KREA_LORA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${KREA_LORA_DIR}"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_LORA_DIR}/configs/ddj_captioned.json}"
source "$(dirname "$0")/_env.sh"

KREA_RESUME_LORA="${KREA_RESUME_LORA:-runs/ddj_captioned_noise_white_lora/checkpoint-1000}"
KREA_STAGE2_OUTPUT="${KREA_STAGE2_OUTPUT:-runs/ddj_captioned_noise_white_lora_stage2}"
KREA_STAGE2_STEPS="${KREA_STAGE2_STEPS:-2000}"
KREA_STAGE2_LR="${KREA_STAGE2_LR:-0.00004}"

accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_captioned_lora.py \
  --config "${KREA_CONFIG}" \
  --resume-lora "${KREA_RESUME_LORA}" \
  --output-dir "${KREA_STAGE2_OUTPUT}" \
  --max-train-steps "${KREA_STAGE2_STEPS}" \
  --learning-rate "${KREA_STAGE2_LR}" \
  --lr-warmup-steps 0
