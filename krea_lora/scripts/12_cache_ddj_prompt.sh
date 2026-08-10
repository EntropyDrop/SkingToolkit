#!/usr/bin/env bash
set -euo pipefail
KREA_DDJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export KREA_CONFIG="${KREA_CONFIG:-${KREA_DDJ_DIR}/configs/ddj_conditional.json}"
source "$(dirname "$0")/_env.sh"
python cache_prompt.py --config "${KREA_CONFIG}" "$@"
