#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
KREA_FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
KREA_MIN_FREE_MB="${KREA_MIN_FREE_MB:-50000}"
if (( KREA_FREE_MB < KREA_MIN_FREE_MB )); then
  echo "Refusing production launch: ${KREA_FREE_MB} MiB free, ${KREA_MIN_FREE_MB} MiB required."
  echo "Let other GPU jobs finish, or explicitly lower KREA_MIN_FREE_MB after enabling training.layerwise_casting."
  exit 2
fi
accelerate launch --num_processes 1 --mixed_precision bf16 train_lora.py --config "${KREA_CONFIG}" "$@"

