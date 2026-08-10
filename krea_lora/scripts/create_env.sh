#!/usr/bin/env bash
set -euo pipefail

KREA_CONDA_ROOT="${KREA_CONDA_ROOT:-/home/ds/miniconda3}"
KREA_SOURCE_ENV="${KREA_SOURCE_ENV:-krea}"
KREA_TRAIN_ENV="${KREA_TRAIN_ENV:-krea-train}"
KREA_LORA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${KREA_CONDA_ROOT}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${KREA_TRAIN_ENV}"; then
  echo "Conda environment already exists: ${KREA_TRAIN_ENV}"
else
  conda create -y -n "${KREA_TRAIN_ENV}" --clone "${KREA_SOURCE_ENV}"
fi

conda activate "${KREA_TRAIN_ENV}"
python -c 'import accelerate, diffusers, peft, safetensors, torch, transformers; print({"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__, "accelerate": accelerate.__version__, "peft": peft.__version__})'
python "${KREA_LORA_DIR}/validate_setup.py" --config "${KREA_LORA_DIR}/configs/mc_preview.json"

