#!/usr/bin/env bash
set -euo pipefail
KREA_DDJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_DDJ_DIR}/configs/ddj_captioned.json}"
source "$(dirname "$0")/_env.sh"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
KREA_FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
KREA_CAPTION_MIN_FREE_MB="${KREA_CAPTION_MIN_FREE_MB:-40000}"
if (( KREA_FREE_MB < KREA_CAPTION_MIN_FREE_MB )); then
  echo "Refusing Qwen caption launch: ${KREA_FREE_MB} MiB free, ${KREA_CAPTION_MIN_FREE_MB} MiB required."
  exit 2
fi
python caption_ddj_sources.py --config "${KREA_CONFIG}" "$@"
