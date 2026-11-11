#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" != --inside ]]; then
  exec unshare -Urn bash "$0" --inside
fi
ip link set lo up
test "$(ip -o link show | wc -l)" -eq 1
test -z "$(ip route show)"
test -z "$(ip -6 route show default)"
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
runtime/whisper.cpp-src/build/bin/whisper-server \
  --host 127.0.0.1 --port 8083 -t 4 -l zh -bs 5 -bo 5 \
  --prompt '普通话简体中文。今天坐了多久、站立目标、久坐时长、添加待办、设置提醒。' \
  --vad -vm models/ggml-silero-v6.2.0.bin -vp 120 \
  -m runtime/whisper-turbo-benchmark/ggml-large-v3-turbo-q5_0.bin >logs/89-whisper-server-runtime.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
timeout 25 bash -c 'until wget -q --timeout=1 --tries=1 -O /dev/null http://127.0.0.1:8083/health; do sleep 1; done'
echo "RESIDENT_TURBO_LOOPBACK_READY"
.venv/bin/python runtime/whisper-resident/check.py
