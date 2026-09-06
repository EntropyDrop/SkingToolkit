#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_DIR="${1:-dense_uv_parser/runs/v103_final_uv_retrain_20260907}"
PYTHON=/home/ds/miniconda3/envs/sking-v61-worker/bin/python
"$PYTHON" dense_uv_parser/run_local.py prepare_final_head_uv \
  --output-dir "$RUN_DIR/cache" --train-count 128 --validation-count 32
exec "$PYTHON" dense_uv_parser/run_local.py train_final_head_uv \
  --cache "$RUN_DIR/cache" --output-dir "$RUN_DIR/training" \
  --steps 6000 --batch-size 6 --first-eval 200 --eval-every 1000
