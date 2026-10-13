#!/usr/bin/env bash
# Overnight trace for 06_titles_llm (owner's request, 2026-09-28): progress every
# 10 min to progress.log; on exit, the final status and a copy of the nbconvert log.
R=$(cd "$(dirname "$0")" && pwd)
L=$R/nbconvert.log   # run 2+: log straight into the run dir (run 1 log: nbconvert.run1.log)
P=$R/progress.log
while pgrep -f "nbconvert.*06_titles_llm" >/dev/null; do
  line="$(date '+%F %T')"
  for f in "$R"/gen_*.jsonl "$R"/test_*.jsonl; do [ -f "$f" ] && line="$line | $(basename "$f" .jsonl) $(wc -l < "$f")"; done
  line="$line | ggufs $(ls /mnt/d/yts-experiments/m5-chapters/llm/*.gguf 2>/dev/null | wc -l)/6 | mem avail $(free -g | awk '/Mem/{print $7}')G | load $(cut -d' ' -f1 /proc/loadavg)"
  echo "$line" >> "$P"
  sleep 600
done
echo "$(date '+%F %T') | notebook process gone; nbconvert log tail:" >> "$P"
tail -5 "$L" >> "$P"
