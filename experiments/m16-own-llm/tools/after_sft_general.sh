#!/bin/bash
# A3: wait for a3-sft-general to finish (state "done", never on a manual pause), then start
# a3-sft-tasks. Logs to ~/yts-experiments/m16-llm/logs/after_sft_general.log.
M16=$(cd "$(dirname "$0")/.." && pwd)
P=~/projects/new_youtube_suite/experiments/.venv/bin/python
S=~/yts-experiments/m16-llm/runs/a3-sft-general/status.json
log() { echo "$(date '+%F %T') $*"; }
log "waiting for a3-sft-general"
until grep -q '"state": "done"' "$S" 2>/dev/null && ! pgrep -f "llm.train _run" >/dev/null; do sleep 60; done
log "a3-sft-general done; starting a3-sft-tasks"
cd "$M16" && "$P" -m llm.train start configs/a3/sft_tasks.yaml
log "started (exit $?)"
