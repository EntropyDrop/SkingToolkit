#!/usr/bin/env bash
set -euo pipefail

KREA_LORA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${KREA_LORA_DIR}"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_LORA_DIR}/configs/ddj_captioned.json}"
source "$(dirname "$0")/_env.sh"

readarray -t DETAIL_VALUES < <(
  python - "${KREA_CONFIG}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    detail = json.load(handle)["detail_finetune"]
for key in (
    "resume_lora",
    "output_dir",
    "max_train_steps",
    "learning_rate",
    "lr_warmup_steps",
    "timestep_min_fraction",
    "timestep_max_fraction",
):
    print(detail[key])
PY
)

KREA_DETAIL_RESUME="${KREA_DETAIL_RESUME:-${DETAIL_VALUES[0]}}"
KREA_DETAIL_OUTPUT="${KREA_DETAIL_OUTPUT:-${DETAIL_VALUES[1]}}"
KREA_DETAIL_STEPS="${KREA_DETAIL_STEPS:-${DETAIL_VALUES[2]}}"
KREA_DETAIL_LR="${KREA_DETAIL_LR:-${DETAIL_VALUES[3]}}"
KREA_DETAIL_WARMUP="${KREA_DETAIL_WARMUP:-${DETAIL_VALUES[4]}}"
KREA_DETAIL_TIMESTEP_MIN="${KREA_DETAIL_TIMESTEP_MIN:-${DETAIL_VALUES[5]}}"
KREA_DETAIL_TIMESTEP_MAX="${KREA_DETAIL_TIMESTEP_MAX:-${DETAIL_VALUES[6]}}"

accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_captioned_lora.py \
  --config "${KREA_CONFIG}" \
  --resume-lora "${KREA_DETAIL_RESUME}" \
  --output-dir "${KREA_DETAIL_OUTPUT}" \
  --max-train-steps "${KREA_DETAIL_STEPS}" \
  --learning-rate "${KREA_DETAIL_LR}" \
  --lr-warmup-steps "${KREA_DETAIL_WARMUP}" \
  --timestep-min-fraction "${KREA_DETAIL_TIMESTEP_MIN}" \
  --timestep-max-fraction "${KREA_DETAIL_TIMESTEP_MAX}"
