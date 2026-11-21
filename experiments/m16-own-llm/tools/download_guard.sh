#!/bin/bash
# Corpus audio download over IPv4 with a single connection, stopped automatically if IPv4 fails for
# 3 consecutive minutes (the ISP CGNAT problem of 2026-11-14), so the PC never loses internet for long.
#   setsid nohup tools/download_guard.sh > ~/yts-experiments/m16-llm/logs/download_guard.log 2>&1 &
P=~/projects/new_youtube_suite/experiments/.venv/bin/python
L=~/yts-experiments/m16-llm/logs
cd "$(dirname "$0")/.."
"$P" -u -m llm.corpus download --ip 4 --fragments 1 >> "$L/corpus_download.log" 2>&1 &
DL=$!
fails=0
while kill -0 "$DL" 2>/dev/null; do
  sleep 60
  if curl -4 -s -o /dev/null --max-time 10 https://www.google.com; then fails=0; else fails=$((fails + 1)); fi
  if [ "$fails" -ge 3 ]; then
    echo "$(date +%H:%M) IPv4 down for 3 min: stopping the download"
    pkill -P "$DL"
    kill "$DL"
    exit 1
  fi
done
echo "$(date +%H:%M) download finished"
