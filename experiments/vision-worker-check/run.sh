#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" != --inside ]]; then
  exec unshare -Urn bash "$0" --inside
fi
ip link set lo up
test "$(ip -o link show | wc -l)" -eq 1
root="$PWD"
fixture="$root/runtime/vision-worker-check"
case_dir="$(mktemp -d /dev/shm/orin-vision-worker.XXXXXX)"
export ORIN_TEST_SOCKET="$case_dir/tmux.sock"
cleanup() {
  if [[ "$1" -ne 0 ]]; then tail -n 60 "$case_dir/logs/07-vision.log" 2>/dev/null || true; fi
  /usr/bin/tmux -S "$ORIN_TEST_SOCKET" kill-server 2>/dev/null || true
  rm -rf "$case_dir"
}
trap 'cleanup "$?"' EXIT
unset TMUX TMUX_PANE
cp stack.sh start-app.sh assistant.py store.py state.py trt_model.py vision.py "$case_dir"
cp "$fixture/main.py" "$case_dir/candidate_main.py"
cp "$fixture/entry.py" "$case_dir/main.py"
for item in .venv runtime models; do ln -s "$root/$item" "$case_dir/$item"; done
mkdir "$case_dir/bin"
cat > "$case_dir/bin/tmux" <<'MOCK'
#!/bin/sh
exec /usr/bin/tmux -S "$ORIN_TEST_SOCKET" "$@"
MOCK
chmod 700 "$case_dir/bin/tmux"
export PATH="$case_dir/bin:$PATH"
cat > "$case_dir/launch-ready.sh" <<'MOCK'
if [[ "$1" == vision ]]; then
  exec bash start-app.sh
fi
exec sleep 90
MOCK
cd "$case_dir"
flock -n -o .watch.lock bash stack.sh watch
timeout 15 bash -c 'until test -e first-attempt && ! tmux list-windows -t =orin-agent -F "#{window_name}" | grep -Fxq vision; do sleep .25; done'
grep -F 'Vision loop failed; stopping the app for supervised restart' logs/07-vision.log
grep -F 'injected camera worker failure' logs/07-vision.log
echo 'INITIAL_VISION_WINDOW_EXITED'
flock -n -o .watch.lock bash stack.sh watch
"$root/.venv/bin/python" - <<'PY'
import json
import time
from urllib.request import urlopen

end = time.monotonic() + 40
while time.monotonic() < end:
    try:
        with urlopen("http://127.0.0.1:8765/api/state", timeout=2) as reply:
            camera = json.load(reply)["camera"]
    except OSError:
        pass
    else:
        if camera.get("ready") and not camera.get("vision_paused") and camera.get("frame_count", 0) >= 8:
            print(f"HTTP_CAMERA_RESTORED frames={camera['frame_count']} fps={camera['fps']}")
            break
    time.sleep(.25)
else:
    raise SystemExit("camera did not recover")
PY
