#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" != --inside ]]; then
  exec unshare -Urn bash "$0" --inside
fi
ip link set lo up
test "$(ip -o link show | wc -l)" -eq 1
bash launch-ready.sh qwen >logs/69-voice-latency-qwen.log 2>&1 &
qwen_pid=$!
trap 'kill "$qwen_pid" 2>/dev/null || true; wait "$qwen_pid" 2>/dev/null || true' EXIT
timeout 55 bash -c 'until wget -q --timeout=2 --tries=1 -O /dev/null http://127.0.0.1:8081/health; do sleep 1; done'
.venv/bin/python runtime/voice-answer-latency/check.py
echo ANSWER_BENCHMARK_COMPLETE
