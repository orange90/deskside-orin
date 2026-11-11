#!/usr/bin/env bash
# Run independent model processes inside a loopback-only network namespace.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" != --inside ]]; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  exec unshare -Urn bash "$0" --inside "${1:-}"
fi
ip link set lo up
test "$(ip -o link show | wc -l)" -eq 1
test -z "$(ip route show)"
test -z "$(ip -6 route show default)"
ip -brief addr show
mkdir -p logs
source ./cosmos-env.sh
cosmos_pid=
qwen_pid=
cleanup() {
  for pid in "$qwen_pid" "$cosmos_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
.cosmos/bin/uvicorn cosmos_server:app --host 127.0.0.1 --port 8082 >logs/21-cosmos-offline-server.log 2>&1 &
cosmos_pid=$!
timeout 60 bash -c 'until wget -q --timeout=2 --tries=1 -O /dev/null http://127.0.0.1:8082/health; do sleep 2; done'
.cosmos/bin/python cosmos-smoke.py
.venv/bin/python voice-smoke.py
if [[ "${2:-}" == --with-qwen ]]; then
  test -x runtime/llama-server
  runtime/llama-server --model models/qwen.gguf --alias qwen --ctx-size 4096 \
    --n-gpu-layers 99 --parallel 1 --host 127.0.0.1 --port 8081 --jinja >logs/21-qwen-offline-server.log 2>&1 &
  qwen_pid=$!
  timeout 60 bash -c 'until wget -q --timeout=2 --tries=1 -O /dev/null http://127.0.0.1:8081/health; do sleep 2; done'
  .venv/bin/python qwen-smoke.py
  .venv/bin/python offline-dashboard-smoke.py
fi
