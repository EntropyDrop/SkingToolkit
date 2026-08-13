#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="${PROJECT_DIR}/runs/captioned_web/server.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "Krea MC Web is not running"
  exit 0
fi
PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill -TERM "${PID}"
  echo "Stopped Krea MC Web PID ${PID}"
else
  echo "Removed stale PID file for ${PID}"
fi
rm -f "${PID_FILE}"
