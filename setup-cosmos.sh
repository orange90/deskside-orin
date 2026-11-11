#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runtime/cosmos-deps
wheels=runtime/cosmos-deps
base=https://pypi.jetson-ai-lab.io/jp6/cu126/+f
torch=torch-2.11.0-cp310-cp310-linux_aarch64.whl
vision=torchvision-0.26.0-cp310-cp310-linux_aarch64.whl
if ! test -s "$wheels/$torch"; then
  wget -nv -T 45 -O "$wheels/$torch.part" "$base/46b/b8b13f844b211/$torch"
  echo "46bb8b13f844b2111f6f41abc703d01075a7185e0c587cb002e957ede2b8c8a3  $wheels/$torch.part" | sha256sum -c
  mv "$wheels/$torch.part" "$wheels/$torch"
fi
if ! test -s "$wheels/$vision"; then
  wget -nv -T 45 -O "$wheels/$vision.part" "$base/d11/6f08d3d62417d/$vision"
  echo "d116f08d3d62417dc36cbf41923e491b8b49b77399575f0878ae391e6e77a159  $wheels/$vision.part" | sha256sum -c
  mv "$wheels/$vision.part" "$wheels/$vision"
fi
if [[ ! -x .cosmos/bin/python ]]; then
  runtime/uv venv .cosmos --python /usr/bin/python3
fi
if [[ -s requirements-cosmos.txt ]]; then
  dependencies=(-r requirements-cosmos.txt)
else
  dependencies=(transformers==5.17.0 accelerate pillow fastapi uvicorn)
fi
UV_NO_CACHE=1 runtime/uv pip install --python .cosmos/bin/python "$wheels/$torch" "$wheels/$vision" "${dependencies[@]}"
runtime/uv pip freeze --python .cosmos/bin/python > runtime/cosmos-python-lock.txt
