#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"

SMOKE_ROOT="${KREA_LORA_DIR}/.smoke"
SMOKE_CONFIG="${SMOKE_ROOT}/smoke.json"
mkdir -p "${SMOKE_ROOT}"
python make_smoke_config.py --source "${KREA_CONFIG}" --output "${SMOKE_CONFIG}" --root "${SMOKE_ROOT}"

python prepare_dataset.py --config "${SMOKE_CONFIG}" --max-images 8 --force
python cache_prompt.py --config "${SMOKE_CONFIG}" --force
python cache_latents.py --config "${SMOKE_CONFIG}" --limit 8 --force
python validate_setup.py --config "${SMOKE_CONFIG}" --require-dataset --require-cache
accelerate launch --num_processes 1 --mixed_precision bf16 train_lora.py --config "${SMOKE_CONFIG}" --max-train-steps 1
test -f "${SMOKE_ROOT}/run/final/pytorch_lora_weights.safetensors"
echo "Smoke test passed: ${SMOKE_ROOT}/run/final/pytorch_lora_weights.safetensors"
