#!/usr/bin/env bash
set -euo pipefail
cd /home/orange/Documents/orin-desk-agent
source cosmos-env.sh
export LD_LIBRARY_PATH="$PWD/runtime/dla/usr/lib/aarch64-linux-gnu/nvidia:$PWD/runtime/dla/usr/local/cuda-12.6/targets/aarch64-linux/lib:$LD_LIBRARY_PATH"
root="$PWD/runtime/yolo26-eval"
export YOLO_CONFIG_DIR="$root/config" YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
"$root/venv/bin/python" "$root/build-fast.py" > "$root/fast-build.log" 2>&1
cd "$root"
for model in yolo26s-pose yolo26m-pose; do
 bash run.sh --model "$model" --backend engine > "$model-engine.log" 2>&1
 bash run.sh --model "$model" --backend engine --review > "$model-engine-review.log" 2>&1
done
echo COMPLETE > fast-batch.status
