#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./cosmos-env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_HOST_IP=127.0.0.1
export PATH="/usr/local/cuda/bin:$PATH"
export CPATH="/usr/local/cuda/include:${CPATH:-}"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PYTHONPATH="$PWD/runtime/vllm-cosmos-src"
export VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
exec .vllm-runtime/bin/python -m vllm.entrypoints.cli.main serve "$PWD/models/cosmos" \
  --served-model-name cosmos --host 127.0.0.1 --port 8082 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":{"count":4,"width":640,"height":640},"video":0}' --enforce-eager \
  --gpu-memory-utilization .30 --kv-cache-memory-bytes 1073741824 --mm-processor-cache-gb 0 --skip-mm-profiling \
  --attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA --allowed-media-domains no-network.invalid
