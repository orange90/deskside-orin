#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root="$PWD"
tag='# orin-desk-agent-supervisor'
job="/usr/bin/flock -n -o '$root/.watch.lock' /bin/bash '$root/stack.sh' watch >>'$root/logs/autostart.log' 2>&1"
# Cron treats percent signs specially and the path is already wrapped in single quotes.
case "$root" in *"'"*|*%*|*$'\n'*) echo "Unsupported path for the cron job." >&2; exit 2;; esac
mkdir -p logs
file=$(mktemp)
trap 'rm -f "$file"' EXIT
crontab -l 2>/dev/null | awk -v tag="$tag" 'index($0, tag) == 0' >"$file" || true
printf '@reboot %s %s\n* * * * * %s %s\n' "$job" "$tag" "$job" "$tag" >>"$file"
crontab "$file"
bash stack.sh start
crontab -l | awk -v tag="$tag" 'index($0, tag) > 0'
