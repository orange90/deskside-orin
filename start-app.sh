#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD/runtime/dla/usr/lib/aarch64-linux-gnu/nvidia:$PWD/runtime/dla/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export ORIN_COSMOS_FRAMES="${ORIN_COSMOS_FRAMES:-3}"
export ORIN_WHISPER_MODEL="${ORIN_WHISPER_MODEL:-ggml-small.bin}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
ulimit -c 0
exec .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8765
