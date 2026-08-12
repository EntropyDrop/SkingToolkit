#!/usr/bin/env bash
set -euo pipefail
KREA_DDJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_DDJ_DIR}/configs/ddj_captioned.json}"
source "$(dirname "$0")/_env.sh"

bash scripts/21_caption_ddj_sources.sh
# Always rebuild metadata after captioning completes. A pilot run may leave a
# valid but intentionally partial metadata.jsonl behind; reusing it here would
# silently train the production LoRA on only the pilot subset.
python prepare_captioned_dataset.py --config "${KREA_CONFIG}" --force
python cache_captioned_prompts.py --config "${KREA_CONFIG}"
python cache_latents.py --config "${KREA_CONFIG}"
bash scripts/25_train_captioned_lora.sh "$@"
