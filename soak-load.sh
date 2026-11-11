#!/usr/bin/env bash
# This run uses synthetic voice and image fixtures and the live camera health endpoint.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
until wget -q --timeout=2 --tries=1 -O /dev/null http://127.0.0.1:8081/health &&
      wget -q --timeout=2 --tries=1 -O /dev/null http://127.0.0.1:8082/health; do
  sleep 10
done
run_id="$(date +%Y%m%d-%H%M%S)"
prefix="logs/soak-$run_id"
.venv/bin/python measure.py --require-models --seconds 7200 --interval 5 \
  --output "$prefix-metrics.jsonl" > "$prefix-summary.json" 2> "$prefix-measure.err" &
monitor_pid=$!
trap 'if kill -0 "$monitor_pid" 2>/dev/null; then kill "$monitor_pid" 2>/dev/null || true; fi' EXIT
round=0
while kill -0 "$monitor_pid" 2>/dev/null; do
  round=$((round+1))
  for job in cosmos voice qwen; do
    printf '\n%s round=%d test=%s\n' "$(date -Iseconds)" "$round" "$job" >> "$prefix-load.log"
    if [[ "$job" == cosmos ]]; then
      command=(.cosmos/bin/python cosmos-smoke.py)
    elif [[ "$job" == voice ]]; then
      command=(.venv/bin/python voice-smoke.py)
    else
      command=(.venv/bin/python qwen-smoke.py)
    fi
    if timeout 120 "${command[@]}" >> "$prefix-load.log" 2>&1; then
      printf 'exit=0\n' >> "$prefix-load.log"
    else
      printf 'exit=%d\n' "$?" >> "$prefix-load.log"
    fi
  done
  for second in $(seq 1 30); do
    kill -0 "$monitor_pid" 2>/dev/null || break
    sleep 10
  done
done
wait "$monitor_pid"
cat "$prefix-summary.json"
