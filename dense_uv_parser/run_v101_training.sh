#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
PYTHON_BIN="${PYTHON_BIN:-/home/ds/miniconda3/envs/sking-v61-worker/bin/python}"
exec "$PYTHON_BIN" dense_uv_parser/run_local.py train_accessories \
  --output-dir "${OUTPUT_DIR:-dense_uv_parser/runs/dense_uv_parser_v101}" \
  --resume "${RESUME:-dense_uv_parser/runs/v101_seed/seed.pt}" \
  --steps "${STEPS:-20000}" --eval-every "${EVAL_EVERY:-1000}" \
  --batch-size 8 --val-samples 64 --clean-samples 64 \
  --lr 0.00005 --replay-weight .25 --require-approved-start "$@"
