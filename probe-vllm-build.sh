#!/usr/bin/env bash
# Configure, but do not build, the first upstream Cosmos3 Edge vLLM commit.
set -euo pipefail
cd "$(dirname "$0")"
source ./cosmos-env.sh
export PATH="/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.7
export CUDA_HOME=/usr/local/cuda
tools="$PWD/runtime/vllm-build-tools"
if [[ ! -x "$tools/bin/python" ]]; then
  runtime/uv venv --python /usr/bin/python3 "$tools"
fi
runtime/uv pip install --python "$tools/bin/python" "cmake>=3.26.1,<4.1" ninja
export PATH="$tools/bin:$PATH"
vendor="$PWD/runtime/vllm-vendor"
export VLLM_CUTLASS_SRC_DIR="$vendor/cutlass"
export TRITON_KERNELS_SRC_DIR="$vendor/triton/python/triton_kernels/triton_kernels"
export DEEPGEMM_SRC_DIR="$vendor/deepgemm"
export FMHA_SM100_SRC_DIR="$vendor/msa"
export FLASH_MLA_SRC_DIR="$vendor/flashmla"
export QUTLASS_SRC_DIR="$vendor/qutlass"
export VLLM_FLASH_ATTN_SRC_DIR="$vendor/flashattn"
if [[ ! -f "$VLLM_FLASH_ATTN_SRC_DIR/csrc/cutlass/include/cutlass/cutlass.h" ]]; then
  test -s "$vendor/flash-cutlass/.revision"
  if [[ -d "$VLLM_FLASH_ATTN_SRC_DIR/csrc/cutlass" ]]; then
    rmdir "$VLLM_FLASH_ATTN_SRC_DIR/csrc/cutlass"
  fi
  ln -s "$vendor/flash-cutlass" "$VLLM_FLASH_ATTN_SRC_DIR/csrc/cutlass"
fi
cmake -G Ninja -S runtime/vllm-cosmos-src -B runtime/vllm-cosmos-build \
  -DVLLM_PYTHON_EXECUTABLE="$PWD/.cosmos/bin/python" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_nvrtc_LIBRARY="$PWD/runtime/cosmos-cuda/usr/local/cuda-12.6/targets/aarch64-linux/lib/libnvrtc.so.12" \
  -DCUDA_NVRTC_LIB="$PWD/runtime/cosmos-cuda/usr/local/cuda-12.6/targets/aarch64-linux/lib/libnvrtc.so.12" \
  -DCMAKE_INSTALL_PREFIX="$PWD/runtime/vllm-cosmos-src" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
