#!/usr/bin/env bash
set -euo pipefail
base=https://pypi.jetson-ai-lab.io/jp6/cu126/+f
relay() {
  local path="$1" hash="$2" name="${1##*/}"
  local remote="Documents/orin-desk-agent/runtime/cosmos-deps"
  curl -fLsS --connect-timeout 15 "$base/$path" |
    ssh -o BatchMode=yes orin "cd '$remote' && cat > '$name.relay'"
  ssh -o BatchMode=yes orin "cd '$remote' && printf '%s  %s\n' '$hash' '$name.relay' | sha256sum -c - && mv '$name.relay' '$name'"
}
relay 46b/b8b13f844b211/torch-2.11.0-cp310-cp310-linux_aarch64.whl 46bb8b13f844b2111f6f41abc703d01075a7185e0c587cb002e957ede2b8c8a3 &
torch_pid=$!
relay d11/6f08d3d62417d/torchvision-0.26.0-cp310-cp310-linux_aarch64.whl d116f08d3d62417dc36cbf41923e491b8b49b77399575f0878ae391e6e77a159 &
vision_pid=$!
status=0
wait "$torch_pid" || status=1
wait "$vision_pid" || status=1
exit "$status"
