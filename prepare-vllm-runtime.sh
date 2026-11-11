#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./cosmos-env.sh
deps="$PWD/runtime/vllm-vendor"
wheel=triton-3.5.1-cp310-cp310-linux_aarch64.whl
if [[ ! -f "$deps/$wheel" ]]; then
  wget -nv -T 35 -O "$deps/$wheel.part" "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/b0f/ad648021f2fd6/$wheel"
  printf '%s  %s\n' b0fad648021f2fd655ee4019f885bf6fb204f7f2efc060f8ddb6a3ca811ba99f "$deps/$wheel.part" | sha256sum -c
  mv "$deps/$wheel.part" "$deps/$wheel"
fi
if [[ ! -x .vllm-runtime/bin/python ]]; then
  runtime/uv venv --python /usr/bin/python3 .vllm-runtime
fi
runtime/uv pip install --python .vllm-runtime/bin/python -r runtime/vllm-cosmos-src/requirements/common.txt \
  runtime/cosmos-deps/torch-2.11.0-cp310-cp310-linux_aarch64.whl \
  runtime/cosmos-deps/torchvision-0.26.0-cp310-cp310-linux_aarch64.whl "$deps/$wheel" \
  transformers==5.17.0 numba==0.65.0 "setuptools>=77.0.3,<81" setuptools-scm setuptools-rust wheel nvidia-ml-py
.vllm-runtime/bin/python -c 'import torch,triton; assert torch.cuda.is_available(); print(torch.__version__,triton.__version__,torch.cuda.get_device_name(0))'
