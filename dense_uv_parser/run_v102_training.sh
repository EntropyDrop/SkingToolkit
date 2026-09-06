#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/ds/miniconda3/envs/sking-v61-worker/bin/python dense_uv_parser/run_local.py train_v102 \
  --output-dir "${1:-dense_uv_parser/runs/v102_multiview_20260906}" \
  --steps 4800 --batch-size 6 --lr 0.0002 --first-eval 200 --eval-every 600 --validation-count 96
