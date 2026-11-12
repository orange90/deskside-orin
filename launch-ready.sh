#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
target="$1"
case "$target" in
  vision)
    until [[ -s models/yolo26m-pose.engine ]]; do sleep 10; done
    exec bash start-app.sh
    ;;
  qwen)
    until [[ -x runtime/llama-server && -s models/qwen.gguf ]]; do sleep 10; done
    exec runtime/llama-server --model models/qwen.gguf --alias qwen --ctx-size 4096 \
      --n-gpu-layers 99 --parallel 1 --host 127.0.0.1 --port 8081 --jinja
    ;;
  cosmos)
    until [[ -x .cosmos/bin/uvicorn && -s runtime/cosmos-cuda/gpu-ready.json \
      && -s models/cosmos/vision_encoder/model.safetensors \
      && -s models/cosmos/transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
      && -s models/cosmos/transformer/diffusion_pytorch_model-00002-of-00002.safetensors ]]; do sleep 10; done
    if [[ -f .cosmos-vllm && -f runtime/vllm-cosmos-build/orin-triton-installed ]]; then
      exec bash serve-vllm-cosmos.sh
    fi
    source ./cosmos-env.sh
    exec .cosmos/bin/uvicorn cosmos_server:app --host 127.0.0.1 --port 8082
    ;;
esac
