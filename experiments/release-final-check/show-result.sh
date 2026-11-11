#!/usr/bin/env bash
set -euo pipefail
log=outputs/orin-desk-agent/evidence/42-release-final-check.txt
echo "Orin release ZIP and offline check"
echo
printf 'Archive SHA-256  '
awk 'NR == 1 {print $1}' "$log"
printf 'Validated weight files  '
grep -c '^models/.*: OK$' "$log"
echo
grep -E '^rtmdet |^rtmpose |^lo |^Due reminder|^ARCHIVE_RUNTIME_OK|^FRESH_TMUX_DEFAULT_SHELL|^vision   ready|^qwen     ready|^cosmos   ready|^COLD_TMUX_BOOT|^RELEASE_ACCEPTANCE_SMOKE_OK' "$log"
