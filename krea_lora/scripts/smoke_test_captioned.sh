#!/usr/bin/env bash
set -euo pipefail
KREA_DDJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$(dirname "$0")/_env.sh"

SMOKE_ROOT="${KREA_LORA_DIR}/.smoke_captioned"
SMOKE_CONFIG="${SMOKE_ROOT}/smoke.json"
SMOKE_PAIRED_CONFIG="${SMOKE_ROOT}/paired.json"
rm -rf -- "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}"
python make_captioned_smoke_config.py \
  --source configs/ddj_captioned.json \
  --paired-source configs/ddj_conditional.json \
  --output "${SMOKE_CONFIG}" \
  --paired-output "${SMOKE_PAIRED_CONFIG}" \
  --root "${SMOKE_ROOT}"
python prepare_paired_dataset.py --config "${SMOKE_PAIRED_CONFIG}" --max-images 2 --force
python caption_ddj_sources.py --config "${SMOKE_CONFIG}" --limit 2 --force
python prepare_captioned_dataset.py --config "${SMOKE_CONFIG}" --force
python cache_captioned_prompts.py --config "${SMOKE_CONFIG}" --force
python cache_latents.py --config "${SMOKE_CONFIG}" --force
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_captioned_lora.py --config "${SMOKE_CONFIG}" --max-train-steps 1
test -f "${SMOKE_ROOT}/run/checkpoint-0/tests/00_img3_generated.png"
test -f "${SMOKE_ROOT}/run/checkpoint-1/pytorch_lora_weights.safetensors"
test -f "${SMOKE_ROOT}/run/final/pytorch_lora_weights.safetensors"
echo "Qwen-captioned smoke test passed: ${SMOKE_ROOT}"
