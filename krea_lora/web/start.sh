#!/usr/bin/env bash
set -euo pipefail

WEB_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${WEB_DIR}/.." && pwd)"
PYTHON_BIN="${KREA_PYTHON:-/home/ds/miniconda3/envs/krea-train/bin/python}"
HOST="${KREA_WEB_HOST:-0.0.0.0}"
PORT="${KREA_WEB_PORT:-7862}"
PID_FILE="${PROJECT_DIR}/runs/captioned_web/server.pid"
LOG_FILE="${PROJECT_DIR}/runs/captioned_web/server.log"

mkdir -p "$(dirname "${PID_FILE}")"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Krea MC Web is already running: PID $(cat "${PID_FILE}")"
  exit 0
fi
cd "${WEB_DIR}"
nohup "${PYTHON_BIN}" -m uvicorn app:app --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"
echo "Krea MC Web started: http://192.168.0.111:${PORT}/ (PID $(cat "${PID_FILE}"))"
echo "Log: ${LOG_FILE}"
