#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
python cache_prompt.py --config "${KREA_CONFIG}" "$@"

