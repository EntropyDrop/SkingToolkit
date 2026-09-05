#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-/home/ds/miniconda3/envs/sking-v61-worker/bin/python}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/run_local.py" semantic_generalization "$@"
