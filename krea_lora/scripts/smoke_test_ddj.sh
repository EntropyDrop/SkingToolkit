#!/usr/bin/env bash
set -euo pipefail
KREA_DDJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_DDJ_DIR}/configs/ddj_conditional.json}"
source "$(dirname "$0")/_env.sh"

SMOKE_ROOT="${KREA_LORA_DIR}/.smoke_ddj"
SMOKE_CONFIG="${SMOKE_ROOT}/smoke.json"
mkdir -p "${SMOKE_ROOT}"
python make_smoke_config.py --source "${KREA_CONFIG}" --output "${SMOKE_CONFIG}" --root "${SMOKE_ROOT}"
python prepare_paired_dataset.py --config "${SMOKE_CONFIG}" --max-images 8 --force
python cache_prompt.py --config "${SMOKE_CONFIG}" --force
python cache_paired_latents.py --config "${SMOKE_CONFIG}" --limit 8 --force
python validate_paired_setup.py --config "${SMOKE_CONFIG}" --require-cache
accelerate launch --num_processes 1 --mixed_precision bf16 train_conditional_lora.py --config "${SMOKE_CONFIG}" --max-train-steps 1
test -f "${SMOKE_ROOT}/run/final/pytorch_lora_weights.safetensors"
SMOKE_SOURCE="$(python -c 'import json,sys; print(json.loads(open(sys.argv[1], encoding="utf-8").readline())["source_image"])' "${SMOKE_ROOT}/data/metadata.jsonl")"
python generate_conditional.py --config "${SMOKE_CONFIG}" --source "${SMOKE_SOURCE}"
test -f "${SMOKE_ROOT}/generated.png"
echo "Paired smoke test passed: ${SMOKE_ROOT}"
