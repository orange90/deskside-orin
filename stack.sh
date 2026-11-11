#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
session=orin-agent
services=(vision qwen cosmos)
has_window() {
  tmux list-windows -t "=$session" -F '#{window_name}' 2>/dev/null | grep -Fxq "$1"
}
case "${1:-status}" in
  start|watch)
    if [[ "$1" == watch && -f .autostart-paused ]]; then exit 0; fi
    if [[ "$1" == start ]]; then rm -f .autostart-paused; fi
    mkdir -p logs
    if ! tmux has-session -t "=$session" 2>/dev/null; then
      tmux new-session -d -s "$session" -n console -c "$PWD"
    fi
    for service in "${services[@]}"; do
      if ! has_window "$service"; then
        tmux new-window -d -t "=$session" -n "$service" -c "$PWD" \
          /bin/bash -o pipefail -c 'bash launch-ready.sh "$1" 2>&1 | tee -a "logs/07-$1.log"' _ "$service"
      fi
    done
    ;;
  stop)
    touch .autostart-paused
    for service in "${services[@]}"; do
      if has_window "$service"; then
        tmux send-keys -t "=$session:=$service" C-c
      fi
    done
    ;;
  status)
    for item in "vision:8765:api/state" "qwen:8081:health" "cosmos:8082:health"; do
      IFS=: read -r service port route <<< "$item"
      if wget --timeout=2 --tries=1 -q -O /dev/null "http://127.0.0.1:$port/$route"; then
        printf '%-8s ready (127.0.0.1:%s)\n' "$service" "$port"
      elif has_window "$service"; then
        printf '%-8s pending (tmux window present)\n' "$service"
      else
        printf '%-8s stopped\n' "$service"
      fi
    done
    ;;
  *) printf 'Usage: %s {start|stop|status|watch}\n' "$0" >&2; exit 2 ;;
esac
