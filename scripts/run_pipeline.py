"""Run every tool on every episode in a folder, through the real API (M9 test harness).

Not a product feature: it is how the M9 season run drives the app
(docs/decisions/0014). For each video file in FOLDER it uploads the file
(unless an upload with the same filename exists), then runs the tool
chain in order, waiting for each job:

    transcribe -> diarize -> chapters -> relevance -> shownotes -> shorts

- Resumable: a tool that already has a succeeded job is skipped
  (`--rerun` runs the chosen tools again anyway).
- `--parallel N` episodes are in flight at once, so one episode's GPU
  steps (transcribe, diarize) overlap another's CPU/LLM steps. The
  worker's GPU and LLM locks keep each resource to one job at a time.
- A failed tool stops that episode's chain (later tools depend on it);
  the other episodes go on.
- One CSV row per job: episode, tool, job id, status, seconds, error.

Run from backend/ with its venv (it mints a token for --user-id):

    cd backend && .venv/bin/python ../scripts/run_pipeline.py /mnt/d/podcasts/X --csv run.csv
"""

import argparse
import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.auth.security import create_access_token  # noqa: E402

TOOLS = ("transcribe", "diarize", "chapters", "relevance", "shownotes", "shorts")
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}
TERMINAL = {"succeeded", "failed"}


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def seconds(job: dict) -> float | None:
    if job.get("started_at") and job.get("completed_at"):
        a = datetime.fromisoformat(job["started_at"])
        b = datetime.fromisoformat(job["completed_at"])
        return round((b - a).total_seconds(), 1)
    return None


class Runner:
    def __init__(self, api: str, token: str, csv_path: Path, tools: list[str], shorts_params: dict, poll_s: float,
                 rerun: bool = False):
        self.client = httpx.AsyncClient(base_url=api, headers={"Authorization": f"Bearer {token}"},
                                        timeout=httpx.Timeout(60, write=600))
        self.tools, self.shorts_params, self.poll_s, self.rerun = tools, shorts_params, poll_s, rerun
        self.csv_path = csv_path
        new = not csv_path.exists()
        self.csv_file = csv_path.open("a", newline="")
        self.csv = csv.writer(self.csv_file)
        if new:
            self.csv.writerow(["finished_at", "episode", "upload_id", "tool", "job_id", "status", "seconds", "error"])

    async def upload_id_for(self, path: Path) -> int:
        uploads = (await self.client.get("/uploads")).raise_for_status().json()
        existing = [u for u in uploads if u["filename"] == path.name]
        if existing:
            return existing[0]["id"]
        log(f"uploading {path.name}")
        with path.open("rb") as f:
            r = await self.client.post("/uploads", files={"file": (path.name, f, "video/mp4")})
        return r.raise_for_status().json()["id"]

    async def jobs(self, upload_id: int) -> list[dict]:
        return (await self.client.get(f"/uploads/{upload_id}/jobs")).raise_for_status().json()

    async def run_tool(self, episode: str, upload_id: int, tool: str) -> bool:
        jobs = [j for j in await self.jobs(upload_id) if j["tool_type"] == tool]
        if not self.rerun and any(j["status"] == "succeeded" for j in jobs):
            log(f"[{episode}] {tool}: already done")
            return True
        active = [j for j in jobs if j["status"] in ("queued", "running")]
        if active:
            job = active[0]
        else:
            body = {"tool_type": tool, **({"params": self.shorts_params} if tool == "shorts" else {})}
            job = (await self.client.post(f"/uploads/{upload_id}/jobs", json=body)).raise_for_status().json()
        log(f"[{episode}] {tool}: job {job['id']} started")
        while job["status"] not in TERMINAL:
            await asyncio.sleep(self.poll_s)
            job = next(j for j in await self.jobs(upload_id) if j["id"] == job["id"])
        secs = seconds(job)
        log(f"[{episode}] {tool}: {job['status']} in {secs}s" + (f" -- {job['error_message']}" if job["error_message"] else ""))
        self.csv.writerow([datetime.now().isoformat(timespec="seconds"), episode, upload_id, tool, job["id"],
                           job["status"], secs, job["error_message"] or ""])
        self.csv_file.flush()
        return job["status"] == "succeeded"

    async def episode(self, path: Path) -> None:
        name = path.stem
        try:
            upload_id = await self.upload_id_for(path)
            for tool in self.tools:
                if not await self.run_tool(name, upload_id, tool):
                    log(f"[{name}] chain stopped after {tool} failed")
                    return
            log(f"[{name}] all tools done")
        except httpx.HTTPError as e:
            log(f"[{name}] API error: {e!r}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--user-id", default="1")
    ap.add_argument("--csv", type=Path, default=Path("pipeline_run.csv"))
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--aspect-ratio", default="9:16")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes (by name)")
    ap.add_argument("--rerun", action="store_true", help="run the tools even if they already succeeded")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if args.limit:
        files = files[: args.limit]
    tools = [t for t in args.tools.split(",") if t]
    unknown = set(tools) - set(TOOLS)
    if unknown:
        sys.exit(f"unknown tools: {sorted(unknown)}")
    runner = Runner(args.api, create_access_token(args.user_id), args.csv, tools,
                    {"aspect_ratio": args.aspect_ratio, "captions": not args.no_captions}, args.poll, args.rerun)
    log(f"{len(files)} episodes, {args.parallel} in flight, tools: {', '.join(tools)}")

    queue: asyncio.Queue[Path] = asyncio.Queue()
    for f in files:
        queue.put_nowait(f)

    async def worker() -> None:
        while not queue.empty():
            await runner.episode(queue.get_nowait())

    t0 = time.time()
    await asyncio.gather(*(worker() for _ in range(max(1, args.parallel))))
    log(f"done in {(time.time() - t0) / 3600:.2f} h; results in {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())
