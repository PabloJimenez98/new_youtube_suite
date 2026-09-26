#!/usr/bin/env bash
# Start the whole local dev stack in one terminal:
#   Postgres + Redis (+ pgAdmin) via docker compose, pending Alembic
#   migrations, the FastAPI API, the arq worker (with the CUDA libs on
#   LD_LIBRARY_PATH) and the Angular dev server.
#
# Output from each process is prefixed with [api] / [worker] / [web].
# Ctrl+C stops the API, worker and frontend; the docker containers keep
# running (stop them with `docker compose stop`).
#
# Usage: scripts/dev.sh
# Env overrides pass through, e.g. `WHISPER_DEVICE=cpu scripts/dev.sh`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
VENV="$BACKEND/.venv"

log() { printf '\033[1;35m[dev]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- Preflight -------------------------------------------------------

command -v docker >/dev/null || die "docker not found"
command -v ffmpeg >/dev/null || die "ffmpeg not found (sudo apt install ffmpeg) -- transcription needs it"
[[ -x "$VENV/bin/python" ]] || die "backend venv missing: see README 'Getting started'"
[[ -f "$BACKEND/.env" ]] || die "backend/.env missing: cp backend/.env.example backend/.env"

for port in 8000 4200; do
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    die "port $port is already in use -- is the stack already running?"
  fi
done

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  log "installing frontend dependencies"
  (cd "$FRONTEND" && npm install)
fi

# ---- Services --------------------------------------------------------

log "starting docker services"
(cd "$ROOT" && docker compose up -d)

log "waiting for db and redis to be healthy"
for _ in $(seq 1 60); do
  healthy=$(cd "$ROOT" && docker compose ps --format '{{.Service}} {{.Health}}' \
    | grep -cE '^(db|redis) healthy$' || true)
  [[ "$healthy" == 2 ]] && break
  sleep 1
done
[[ "$healthy" == 2 ]] || die "db/redis not healthy after 60s (check: docker compose ps)"

log "applying migrations"
(cd "$BACKEND" && "$VENV/bin/alembic" upgrade head)

# ---- App processes ---------------------------------------------------

# CUDA 12 runtime from the nvidia-* pip wheels, for the worker's GPU path
# (docs/decisions/0007-transcription-pipeline.md).
SITE_PACKAGES="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cleanup() {
  trap - INT TERM EXIT
  log "stopping api, worker and web (docker services keep running)"
  kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

run() {
  local name=$1 dir=$2; shift 2
  (cd "$dir" && exec "$@") 2>&1 | sed -u "s/^/[$name] /" &
}

run api    "$BACKEND"  "$VENV/bin/uvicorn" app.main:app --reload
run worker "$BACKEND"  "$VENV/bin/arq" app.worker.settings.WorkerSettings
run web    "$FRONTEND" npx ng serve

log "API http://127.0.0.1:8000 | app http://localhost:4200 | pgAdmin http://localhost:5050"
log "Ctrl+C to stop"
wait
