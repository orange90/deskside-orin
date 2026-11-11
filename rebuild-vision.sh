#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runtime/dla logs
(
  cd runtime/dla
  apt-get download nvidia-l4t-dla-compiler=36.4.4-20250616085344 libcudla-12-6=12.6.68-1
  dpkg-deb -x nvidia-l4t-dla-compiler_36.4.4-20250616085344_arm64.deb .
  dpkg-deb -x libcudla-12-6_12.6.68-1_arm64.deb .
)
export LD_LIBRARY_PATH="$PWD/runtime/dla/usr/lib/aarch64-linux-gnu/nvidia:$PWD/runtime/dla/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
.venv/bin/python export-detector.py
trt=/usr/src/tensorrt/bin/trtexec
"$trt" --onnx=models/rtmdet-raw.onnx --saveEngine=models/rtmdet.plan \
  --fp16 --builderOptimizationLevel=2 --skipInference 2>&1 | tee logs/04-rtmdet-trt.log
"$trt" --onnx=models/rtmpose.onnx --saveEngine=models/rtmpose.plan \
  --shapes=input:1x3x256x192 --fp16 --skipInference 2>&1 | tee logs/04-rtmpose-trt.log
