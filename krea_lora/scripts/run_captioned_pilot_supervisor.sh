#!/usr/bin/env bash
set -euo pipefail

KREA_LORA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${KREA_LORA_DIR}"

TARGET_CAPTIONS="${KREA_PILOT_CAPTIONS:-128}"
CAPTION_FILE="data/ddj_captioned_front_left_back_left_white_512/qwen_captions.jsonl"
PIPELINE_PID_FILE="runs/ddj_captioned_pipeline.pid"
PILOT_OUTPUT="${KREA_PILOT_OUTPUT:-runs/ddj_captioned_front_left_back_left_pilot_lora}"

if [[ ! -f "${PIPELINE_PID_FILE}" ]]; then
  echo "Missing pipeline PID file: ${PIPELINE_PID_FILE}"
  exit 2
fi
PIPELINE_PID="$(<"${PIPELINE_PID_FILE}")"
echo "waiting for ${TARGET_CAPTIONS} captions; pipeline=${PIPELINE_PID}"

while kill -0 "${PIPELINE_PID}" 2>/dev/null; do
  count=0
  [[ -f "${CAPTION_FILE}" ]] && count="$(wc -l < "${CAPTION_FILE}")"
  echo "$(date -Is) captions=${count}/${TARGET_CAPTIONS}"
  if (( count >= TARGET_CAPTIONS )); then
    break
  fi
  sleep 30
done

count=0
[[ -f "${CAPTION_FILE}" ]] && count="$(wc -l < "${CAPTION_FILE}")"
if (( count < TARGET_CAPTIONS )); then
  echo "caption pipeline stopped early at ${count}"
  exit 3
fi

caption_pid="$(pgrep -P "${PIPELINE_PID}" -f caption_ddj_sources.py | head -n 1 || true)"
if [[ -z "${caption_pid}" ]]; then
  caption_shell_pid="$(pgrep -P "${PIPELINE_PID}" | head -n 1 || true)"
  if [[ -n "${caption_shell_pid}" ]]; then
    caption_pid="$(pgrep -P "${caption_shell_pid}" -f caption_ddj_sources.py | head -n 1 || true)"
  fi
fi
if [[ -n "${caption_pid}" ]]; then
  echo "stopping caption process ${caption_pid} after ${count} durable captions"
  kill -TERM "${caption_pid}"
fi

for _ in $(seq 1 60); do
  if ! kill -0 "${PIPELINE_PID}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "${PIPELINE_PID}" 2>/dev/null; then
  echo "stopping remaining pipeline shell ${PIPELINE_PID}"
  kill -TERM "${PIPELINE_PID}"
fi

source /home/ds/miniconda3/etc/profile.d/conda.sh
conda activate krea-train
python prepare_captioned_dataset.py --config configs/ddj_captioned.json --force --allow-partial
python cache_captioned_prompts.py --config configs/ddj_captioned.json
python cache_latents.py --config configs/ddj_captioned.json
accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_captioned_lora.py \
  --config configs/ddj_captioned.json \
  --max-train-steps 250 \
  --output-dir "${PILOT_OUTPUT}"
echo "PILOT_COMPLETE ${PILOT_OUTPUT}"
