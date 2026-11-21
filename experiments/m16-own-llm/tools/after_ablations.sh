#!/bin/bash
# Runs the GPU work that must not overlap the A1 ablation queue, once every ablation is done:
# corpus processing (Whisper + Qwen teacher) of the audio still waiting.
# It waits for ALL configs/ablations/*.yaml runs to be "done" and no trainer to be alive, so a
# manual pause (gaming, using the app) never triggers it. It does not start stage A: that needs
# the ablation results reviewed first (notebook 02).
# (Until 2026-11-15 it also tokenized the extra Spanish text for stage B; stage B left the scope,
# 0025 amendment, and that text was deleted.)
#   setsid nohup tools/after_ablations.sh >> ~/yts-experiments/m16-llm/logs/after_ablations.log 2>&1 < /dev/null &
P=~/projects/new_youtube_suite/experiments/.venv/bin/python
L=~/yts-experiments/m16-llm/logs
R=~/yts-experiments/m16-llm/runs
cd "$(dirname "$0")/.."

all_done() {
  for c in configs/ablations/*.yaml; do
    s="$R/ablation-$(basename "$c" .yaml)/status.json"
    [ -f "$s" ] && grep -q '"state": "done"' "$s" || return 1
  done
  ! pgrep -f 'llm.train' >/dev/null
}

echo "$(date '+%F %H:%M') waiting for the ablation queue to finish"
until all_done; do sleep 300; done
echo "$(date '+%F %H:%M') all ablations done"

echo "$(date '+%F %H:%M') corpus process: start"
"$P" -u -m llm.corpus process >> "$L/corpus_process.log" 2>&1
echo "$(date '+%F %H:%M') corpus process: exit code $?"
