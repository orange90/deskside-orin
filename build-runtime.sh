#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/usr/local/cuda/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=4
for engine in whisper.cpp llama.cpp; do
  source="runtime/$engine-src"
  test -s "$source/.revision"
  cat "$source/.revision"
  args=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=OFF)
  if [[ "$engine" == llama.cpp ]]; then
    args+=(-DLLAMA_CURL=OFF)
    target=llama-server
  else
    target=whisper-cli
  fi
  cmake -S "$source" -B "$source/build" "${args[@]}"
  cmake --build "$source/build" --target "$target"
  ln -sfn "$engine-src/build/bin/$target" "runtime/$target"
done
