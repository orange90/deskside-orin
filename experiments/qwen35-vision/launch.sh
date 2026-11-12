#!/usr/bin/env bash
set -euo pipefail
cd /home/orange/Documents/orin-desk-agent
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
ulimit -c 0
exec runtime/llama-server --model models/qwen.gguf \
  --mmproj runtime/qwen35-vision-eval/mmproj-F16.gguf \
  --alias qwen35-vision-eval --ctx-size 4096 --n-gpu-layers 99 \
  --parallel 1 --host 127.0.0.1 --port 8083 --jinja \
  --image-max-tokens 1024 --no-cache-prompt --cache-ram 0 --threads 4
