#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source_root="$(realpath ../..)"
tree="$(mktemp -d "$PWD/release.XXXXXX")"
unzip -q release.zip -d "$tree"
cd "$tree"
for dir in .venv .cosmos models runtime; do
  ln -s "$source_root/$dir" "$dir"
done
mkdir -p logs "$PWD/tmux"
user_name="$(id -un)"
env -i HOME="$HOME" LOGNAME="$user_name" USER="$user_name" PATH=/usr/bin:/bin TERM=xterm-256color SHELL=/bin/sh \
  TMUX_TMPDIR="$PWD/tmux" ORIN_CAMERA=1234 /usr/bin/unshare -Urn /bin/bash -s <<'INSIDE'
set -euo pipefail
ip link set lo up
test "$(ip -o link show | wc -l)" -eq 1
if tmux has-session -t '=orin-agent' 2>/dev/null; then
  echo "The isolated tmux server was not empty." >&2
  exit 1
fi
cleanup() {
  socket="$(tmux display-message -p '#{socket_path}' 2>/dev/null || true)"
  case "$socket" in "$TMUX_TMPDIR"/*) tmux kill-server;; esac
}
trap cleanup EXIT
/usr/bin/flock -n -o "$PWD/.watch.lock" /bin/bash "$PWD/stack.sh" watch
test "$(tmux show-options -gqv default-shell)" = /bin/sh
echo "FRESH_TMUX_DEFAULT_SHELL=/bin/sh"
for path in 8765/api/state 8081/health 8082/health; do
  timeout 100 bash -c 'until wget -q --timeout=2 --tries=1 -O /dev/null "http://127.0.0.1:$1"; do sleep 2; done' _ "$path"
done
bash stack.sh status
.venv/bin/python - <<'PY'
import json
from urllib.request import urlopen
with urlopen("http://127.0.0.1:8765/api/state",timeout=4) as response:
    result=json.load(response)
assert not result["camera"]["ready"]
assert isinstance(result["settings"],dict)
print("Dashboard and empty test database healthy; real camera reserved for production.")
PY
old="$(tmux display-message -p -t '=orin-agent:=qwen' '#{pane_pid}')"
tmux send-keys -t '=orin-agent:=qwen' C-c
timeout 30 bash -c 'while tmux list-windows -t "=orin-agent" -F "#{window_name}" | grep -Fxq qwen; do sleep 1; done'
/usr/bin/flock -n -o "$PWD/.watch.lock" /bin/bash "$PWD/stack.sh" watch
timeout 90 bash -c 'until wget -q --timeout=2 --tries=1 -O /dev/null http://127.0.0.1:8081/health; do sleep 2; done'
new="$(tmux display-message -p -t '=orin-agent:=qwen' '#{pane_pid}')"
test "$old" != "$new"
echo "COLD_TMUX_BOOT_AND_QWEN_RECOVERY_OK"
INSIDE
