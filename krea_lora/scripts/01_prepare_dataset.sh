#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
python prepare_dataset.py --config "${KREA_CONFIG}" "$@"

