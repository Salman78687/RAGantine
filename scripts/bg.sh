#!/usr/bin/env bash
# Run a long ragantine command as a detached, memory-capped background job.
#
#   scripts/bg.sh NAME ragantine-args...      e.g.  scripts/bg.sh questions bench questions --config configs/v01-free.yaml --yes
#   scripts/bg.sh status NAME                  is it still running?
#   scripts/bg.sh stop NAME
#
# The job runs as a systemd user service, outside your editor and terminal, so closing VS Code or
# a Claude session does not stop it, and an out-of-memory condition can only stop the job itself
# (MemoryMax), never your editor. Output goes to data/logs/NAME.log.
set -euo pipefail
cd "$(dirname "$0")/.."
MEM_MAX="${RAGANTINE_MEM_MAX:-5G}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

case "${1:-}" in
  status) systemctl --user status "ragantine-$2" --no-pager --lines=0 2>/dev/null | sed -n 1,4p || true
          echo "--- tail data/logs/$2.log"; tail -n 5 "data/logs/$2.log" 2>/dev/null || true; exit 0 ;;
  stop)   systemctl --user stop "ragantine-$2"; exit 0 ;;
  ""|-h|--help) sed -n 2,11p "$0"; exit 0 ;;
esac

name="$1"; shift
mkdir -p data/logs
log="data/logs/$name.log"
systemd-run --user --unit="ragantine-$name" --collect --same-dir \
  -p MemoryMax="$MEM_MAX" -p MemorySwapMax=1G -p CPUWeight=40 -p Nice=10 \
  --setenv=PYTHONUNBUFFERED=1 --setenv=RAGANTINE_TORCH_THREADS="${RAGANTINE_TORCH_THREADS:-4}" \
  --setenv=MALLOC_ARENA_MAX=2 \
  ${RAGANTINE_MODELS_CONFIG:+--setenv=RAGANTINE_MODELS_CONFIG="$RAGANTINE_MODELS_CONFIG"} \
  /bin/bash -c "'$UV' run ragantine $(printf '%q ' "$@") > '$log' 2>&1; echo \"exit=\$?\" >> '$log'"
echo "started ragantine-$name (memory cap $MEM_MAX) -> $log"
echo "check: scripts/bg.sh status $name"
