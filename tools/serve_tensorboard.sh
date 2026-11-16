#!/usr/bin/env bash
# Serve TensorBoard for FluxVLA work_dirs on a shared-filesystem node.
#
# The DLC launcher logs metrics as JSONL; this script mirrors them into
# TensorBoard event files (tools/metrics_jsonl_to_tensorboard.py) and serves the
# work_dirs tree over HTTP on 127.0.0.1, so it can be reached through an SSH
# tunnel.
#
# Usage:
#   bash tools/serve_tensorboard.sh [WORK_DIRS_ROOT] [PORT]
#   bash tools/serve_tensorboard.sh stop
#
# Defaults: root = /mnt/data/cpfs/users/neri/work_dirs, port = 6006.
set -uo pipefail

ROOT="${1:-/mnt/data/cpfs/users/neri/work_dirs}"
PORT="${2:-6006}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${TB_LOG_DIR:-/tmp}"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "metrics_jsonl_to_tensorboar[d]" && echo "stopped metrics mirror"
  pkill -f "tensorboard.main --logdir.*--port " && echo "stopped tensorboard"
  exit 0
fi

python3 -c "import tensorboard" >/dev/null 2>&1 || {
  echo "tensorboard is not installed: python3 -m pip install --break-system-packages tensorboard" >&2
  exit 1
}

pkill -f "metrics_jsonl_to_tensorboar[d]" 2>/dev/null
pkill -f "tensorboard.main --logdir.*--port " 2>/dev/null
sleep 1

setsid nohup python3 "${HERE}/metrics_jsonl_to_tensorboard.py" \
  --root "${ROOT}" --watch 30 > "${LOG_DIR}/tb_mirror.log" 2>&1 < /dev/null &
echo "metrics mirror started, log: @${LOG_DIR}/tb_mirror.log"

sleep 5
setsid nohup python3 -m tensorboard.main \
  --logdir "${ROOT}" \
  --host 127.0.0.1 --port "${PORT}" \
  --reload_interval 15 --load_fast=false \
  > "${LOG_DIR}/tensorboard.log" 2>&1 < /dev/null &
sleep 8

if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/"; then
  echo "TensorBoard: http://127.0.0.1:${PORT}/"
  echo "tunnel:      ssh -N -L ${PORT}:127.0.0.1:${PORT} <this-host>"
else
  echo "tensorboard did not answer on port ${PORT}; see @${LOG_DIR}/tensorboard.log" >&2
  exit 1
fi
