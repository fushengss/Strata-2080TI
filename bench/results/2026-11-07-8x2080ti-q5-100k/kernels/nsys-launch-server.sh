#!/bin/sh
# Prepared but not launched by the research agent. Requires a maintenance window.
# Usage: sh nsys-launch-server.sh CONFIG SESSION PORT > SERVER_LOG 2>&1
# Wrap the Python server, preserving the engine's private stdin/stdout protocol.
set -eu
CONFIG=$1
SESSION=$2
PORT=${3:-8091}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /srv/Strata-0157
exec /usr/local/bin/nsys launch --session-new="$SESSION" \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node --cuda-event-trace=false \
  --osrt-threshold=10000 --show-output=true --wait=all \
  /srv/Strata/.venv/bin/python /srv/Strata-0157/serve/server.py \
  --engine strata --config "$CONFIG" --port "$PORT"
