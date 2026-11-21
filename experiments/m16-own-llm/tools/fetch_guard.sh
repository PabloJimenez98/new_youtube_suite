#!/bin/bash
# Spanish HF text download (FineWeb2-HQ, for stage B) over IPv4 with one plain-HTTP connection.
# Like download_guard.sh: if IPv4 fails for 3 consecutive minutes (the ISP CGNAT problem of
# 2026-11-14) it stops the download so the PC gets its internet back; unlike it, it then waits for
# IPv4 to recover and relaunches (fetch is resumable: finished files are in the manifest), up to
# 10 times. Never touches the shards, so a running training is unaffected.
#   setsid nohup tools/fetch_guard.sh >> ~/yts-experiments/m16-llm/logs/fetch_guard.log 2>&1 < /dev/null &
P=~/projects/new_youtube_suite/experiments/.venv/bin/python
L=~/yts-experiments/m16-llm/logs
cd "$(dirname "$0")/.."
v4() { curl -4 -s -o /dev/null --max-time 10 https://huggingface.co; }

for attempt in $(seq 1 10); do
  until v4; do sleep 60; done
  echo "$(date '+%F %H:%M') attempt $attempt: starting fetch"
  HF_HUB_DISABLE_XET=1 "$P" -u -m llm.prepare_data fetch --source es --workers 1 --ipv4 >> "$L/fetch.log" 2>&1 &
  FP=$!
  fails=0
  while kill -0 "$FP" 2>/dev/null; do
    sleep 60
    if v4; then fails=0; else fails=$((fails + 1)); fi
    if [ "$fails" -ge 3 ]; then
      echo "$(date '+%F %H:%M') IPv4 down for 3 min: stopping the fetch"
      kill "$FP"; wait "$FP" 2>/dev/null
      continue 2
    fi
  done
  wait "$FP"; rc=$?
  if [ "$rc" -eq 0 ]; then echo "$(date '+%F %H:%M') fetch finished"; exit 0; fi
  echo "$(date '+%F %H:%M') fetch exited with code $rc; retrying in 5 min"
  sleep 300
done
echo "$(date '+%F %H:%M') giving up after 10 attempts"
exit 1
