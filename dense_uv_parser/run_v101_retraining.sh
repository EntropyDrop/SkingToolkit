#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
PYTHON_BIN="${PYTHON_BIN:-/home/ds/miniconda3/envs/sking-v61-worker/bin/python}"
exec "$PYTHON_BIN" dense_uv_parser/run_local.py train_accessories \
  --output-dir "${OUTPUT_DIR:-dense_uv_parser/runs/dense_uv_parser_v101_retrain_20260906}" \
  --resume "${RESUME:-dense_uv_parser/runs/v101_components_pilot/latest.pt}" \
  --pipeline dense_uv_parser/v101_foreground_pipeline.json --hat-components \
  --steps "${STEPS:-8000}" --eval-every "${EVAL_EVERY:-1000}" \
  --batch-size 8 --val-samples 64 --clean-samples 64 --lr .00002 --replay-weight 0 "$@"
