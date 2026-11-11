#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == --inside ]]; then
  ip link set lo up
  test "$(ip -o link show | wc -l)" -eq 1
  test -z "$(ip route show)"
  test -z "$(ip -6 route show default)"
  export PYTHONPATH="$PWD"
  if [[ "${ORIN_CAPTURE_CANDIDATE:-0}" == 1 ]]; then
    export PYTHONPATH="$PWD/runtime/voice-capture-audit/candidate:$PYTHONPATH"
  fi
  .venv/bin/python runtime/voice-capture-audit/check.py
  exit
fi

test_dir="$(mktemp -d /tmp/orin-voice-suite.XXXXXX)"
export PULSE_SERVER="unix:$test_dir/native"
export ORIN_AUDIO_SOURCE="orin_fixture.monitor"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
dbus-run-session -- bash -c 'printf "%s\n" "$$" > "$1"; shift; exec pulseaudio "$@"' _ "$test_dir/pulse.pid" \
  -n --daemonize=no --use-pid-file=no --exit-idle-time=-1 --disable-shm=yes \
  --log-target=stderr --load="module-native-protocol-unix socket=$test_dir/native auth-anonymous=1" \
  --load="module-null-sink sink_name=orin_fixture rate=48000" >"$test_dir/pulse.log" 2>&1 &
pulse_pid=$!
cleanup() {
  if test -s "$test_dir/pulse.pid"; then kill "$(cat "$test_dir/pulse.pid")" 2>/dev/null || true; fi
  wait "$pulse_pid" 2>/dev/null || true
  if [[ "$1" -ne 0 ]]; then tail -n 5 "$test_dir"/*.log; fi
  rm -rf -- "$test_dir"
}
trap 'cleanup "$?"' EXIT
timeout 10 bash -c 'until test -S "$1"; do sleep .2; done' _ "$test_dir/native"
test "$(pactl list short sources | cut -f2)" = "$ORIN_AUDIO_SOURCE"
test "$(pactl list short sinks | cut -f2)" = "orin_fixture"
echo "PRIVATE_PULSE: virtual input only. NETWORK: loopback only."
unshare -Urn bash "$0" --inside "$test_dir"
