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
# Export is isolated from the lightweight production environment.
if [[ ! -s models/yolo26m-pose.onnx ]]; then
  export_python="${ORIN_EXPORT_PYTHON:-runtime/yolo26-eval/venv/bin/python}"
  if [[ ! -x "$export_python" ]]; then
    echo 'Set ORIN_EXPORT_PYTHON to a Python environment with ultralytics==8.4.210 and torch; see docs/YOLO26m上线.md' >&2
    exit 1
  fi
  YOLO_OFFLINE=true YOLO_AUTOINSTALL=false "$export_python" - <<'PYEXPORT'
from ultralytics import YOLO, settings
settings.update({'sync': False})
YOLO('models/yolo26m-pose.pt').export(format='onnx', device='cpu', imgsz=640, batch=1,
    half=False, simplify=False, opset=17, dynamic=False)
PYEXPORT
fi
.venv/bin/python build-yolo-engine.py 2>&1 | tee logs/yolo26m-build.log
