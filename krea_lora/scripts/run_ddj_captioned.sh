#!/usr/bin/env bash
set -euo pipefail
KREA_DDJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_DDJ_DIR}/configs/ddj_captioned.json}"
source "$(dirname "$0")/_env.sh"

# Render the paired metadata first because a new view-specific dataset has no
# source manifest until this step. Captions are view-independent and can then
# be copied from the previous caption file by caption_ddj_sources.py.
python prepare_paired_dataset.py --config configs/ddj_conditional.json --force
bash scripts/21_caption_ddj_sources.sh
# Always rebuild metadata after captioning completes. A pilot run may leave a
# valid but intentionally partial metadata.jsonl behind; reusing it here would
# silently train the production LoRA on only the pilot subset.
python prepare_captioned_dataset.py --config "${KREA_CONFIG}" --force
python cache_captioned_prompts.py --config "${KREA_CONFIG}"
python cache_latents.py --config "${KREA_CONFIG}" --force
bash scripts/25_train_captioned_lora.sh "$@"
