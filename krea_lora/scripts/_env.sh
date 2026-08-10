#!/usr/bin/env bash
set -euo pipefail

KREA_LORA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KREA_CONDA_ROOT="${KREA_CONDA_ROOT:-/home/ds/miniconda3}"
KREA_TRAIN_ENV="${KREA_TRAIN_ENV:-krea-train}"
KREA_CONFIG="${KREA_CONFIG:-${KREA_LORA_DIR}/configs/mc_preview.json}"

source "${KREA_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${KREA_TRAIN_ENV}"
cd "${KREA_LORA_DIR}"

