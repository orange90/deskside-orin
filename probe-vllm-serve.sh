#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" != isolated ]]; then
  while tmux list-windows -t '=orin-agent' -F '#{window_name}' | grep -Fxq vllm-build; do
    sleep 20
  done
  test -f runtime/vllm-cosmos-build/orin-triton-installed || {
    echo "Native installation did not complete. Inspect logs/29-vllm-build.log." >&2
    exit 1
  }
  exec unshare -Urn bash "$0" isolated
fi
ip link set lo up
.venv/bin/python - <<'PY'
import subprocess
interfaces = subprocess.check_output(["ip", "-br", "link"], text=True)
assert [line.split()[0] for line in interfaces.splitlines()] == ["lo"]
print("Network namespace contains only loopback.")
PY
bash serve-vllm-cosmos.sh &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
for ((i=0; i<240; i++)); do
  kill -0 "$pid"
  if wget --timeout=1 --tries=1 -qO /dev/null http://127.0.0.1:8082/health; then
    .venv/bin/python - <<'PYTHON'
import json
from runpy import run_path
import time
import assistant

image = run_path("cosmos-smoke.py")["image"]
for expected, frames in [
    ("standing", [image(174)]),
    ("sitting", [image(153, "sitting")]),
    ("away", [image(0, "away")]),
    ("standing", [image(x) for x in (174, 137, 93)]),
]:
    start = time.monotonic()
    observed, _ = assistant.describe_desk_frames(frames, timeout=60)
    print(json.dumps({"expected": expected, "observed": observed, "frames": len(frames),
                      "seconds": round(time.monotonic() - start, 3)}), flush=True)
    assert observed == expected
PYTHON
    exit 0
  fi
  sleep 3
done
echo "vLLM did not become healthy in 12 minutes." >&2
exit 1
