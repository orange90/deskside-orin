#!/usr/bin/env bash
set -euo pipefail
cd /home/orange/Documents/orin-desk-agent
export LD_LIBRARY_PATH="$PWD/runtime/dla/usr/lib/aarch64-linux-gnu/nvidia:$PWD/runtime/dla/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
python="$PWD/.venv/bin/python"
cd runtime/yolo26-switch
"$python" -m unittest discover -s tests -p 'test_*.py' -v
"$python" validate.py
echo PASS > validated
