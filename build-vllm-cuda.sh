#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./cosmos-env.sh
tools="$PWD/runtime/vllm-build-tools"
export PATH="$tools/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.7
# Retain the existing pressure-test baseline without large compiler activity.
if [[ -n "${1:-}" ]]; then
  while [[ ! -s "$1" ]]; do
    sleep 20
  done
  python3 - "$1" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    report = json.load(f)
assert report["elapsed_seconds"] >= 7200
PY
fi
test -s runtime/vllm-cosmos-build/build.ninja
# The model registry imports a FlashAttention extension even when using Triton.
# Build FA2 for Ampere and omit FA3 for Hopper.
build_dir=runtime/vllm-cosmos-build
targets=(cumem_allocator _C_stable_libtorch _moe_C_stable_libtorch _vllm_fa2_C)
rm -f "$build_dir/orin-triton-installed"
cmake --build "$build_dir" --parallel 2 --target "${targets[@]}"
for component in "${targets[@]}" triton_kernels; do
  cmake --install "$build_dir" --component "$component"
done
touch "$build_dir/orin-triton-installed"
