#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"

KREA_DATASET_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["data"]["dataset_dir"])' "${KREA_CONFIG}")"
if [[ ! -f "${KREA_DATASET_DIR}/metadata.jsonl" ]]; then
  python prepare_dataset.py --config "${KREA_CONFIG}"
fi
if [[ ! -f "${KREA_DATASET_DIR}/prompt_cache.safetensors" ]]; then
  python cache_prompt.py --config "${KREA_CONFIG}"
fi
python cache_latents.py --config "${KREA_CONFIG}"
bash scripts/04_train.sh "$@"

