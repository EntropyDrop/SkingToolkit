#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/home/ds/miniconda3/envs/sking-v61-worker/bin/python}"
VERSION="${VERSION:-v101}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
exec "$PYTHON_BIN" run_local.py batch_accessories --checkpoint "${CHECKPOINT:-runs/dense_uv_parser_${VERSION}/best.pt}" --output-dir "${OUTPUT_DIR:-output_history/${VERSION}}" "$@"
