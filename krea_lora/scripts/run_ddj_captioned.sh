#!/usr/bin/env bash
set -euo pipefail
KREA_DDJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_DDJ_DIR}/configs/ddj_captioned.json}"
source "$(dirname "$0")/_env.sh"

KREA_DATASET_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["data"]["dataset_dir"])' "${KREA_CONFIG}")"
bash scripts/21_caption_ddj_sources.sh
if [[ ! -f "${KREA_DATASET_DIR}/metadata.jsonl" ]]; then
  python prepare_captioned_dataset.py --config "${KREA_CONFIG}"
fi
python cache_captioned_prompts.py --config "${KREA_CONFIG}"
python cache_latents.py --config "${KREA_CONFIG}"
bash scripts/25_train_captioned_lora.sh "$@"
