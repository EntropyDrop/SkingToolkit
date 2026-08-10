#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
python validate_lora.py --config "${KREA_CONFIG}" "$@"

