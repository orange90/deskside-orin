#!/usr/bin/env bash
set -euo pipefail
cd /home/orange/Documents/orin-desk-agent
root="$PWD/runtime/yolo26-eval"
source cosmos-env.sh
export LD_LIBRARY_PATH="$PWD/runtime/dla/usr/lib/aarch64-linux-gnu/nvidia:$PWD/runtime/dla/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH"
export YOLO_CONFIG_DIR="$root/config" YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
ulimit -c 0
"$root/venv/bin/python" "$root/evaluate.py" "$@"
