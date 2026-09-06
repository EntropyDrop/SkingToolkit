#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/ds/miniconda3/envs/sking-v61-worker/bin/python dense_uv_parser/run_local.py train_v103 \
  --output-dir "${1:-dense_uv_parser/runs/v103_uv_structured_20260906}" \
  --steps 6000 --batch-size 4 --lr 0.00005 --first-eval 200 --eval-every 1000 --validation-count 128
