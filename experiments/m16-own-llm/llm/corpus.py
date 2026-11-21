"""Task corpus (0025 §6): podcast audio from the 22 M10 seed channels -> transcripts -> Qwen teacher.

    python -m llm.corpus select     # list each channel's recent long-form videos, pick up to the hour quota
    python -m llm.corpus download   # audio only, resumable (yt-dlp download archive)

Transcription and the teacher outputs are separate steps (notebook 03). Everything
lives under $M16_CORPUS (default ~/yts-data/podcasts/m16-corpus), never in the repo.

Quotas: 25 h per Spanish channel, 12 h per English channel (~400 h in total, about
70/30 like pretraining). Sastre y Maldonado is skipped: T1-T2 are already local
(~/yts-data/podcasts/Sastre_y_Maldonado_T*) and hold the 0024 test episodes.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SEED = REPO / "backend" / "app" / "market" / "seed_channels.json"
CORPUS = Path(os.environ.get("M16_CORPUS", "~/yts-data/podcasts/m16-corpus")).expanduser()
YTDLP = Path(os.environ.get("YTDLP", "~/.venvs/yt-dlp/bin/yt-dlp")).expanduser()
QUOTA_H = {"es": 25.0, "en": 12.0}
MIN_S = 15 * 60
SKIP = {"@SastreyMaldonado"}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def channels() -> list[dict]:
    return [c for c in json.loads(SEED.read_text())["channels"] if c.get("ref") and c["ref"] not in SKIP]


def find_node() -> str | None:
    """`node` on PATH, else the newest nvm install (nvm is only loaded in interactive shells)."""
    node = shutil.which("node")
    if node:
        return node
    nvm = sorted(Path("~/.nvm/versions/node").expanduser().glob("v*/bin/node"))
    return str(nvm[-1]) if nvm else None


def ytdlp_base() -> list[str]:
    node = find_node()
    cmd = [str(YTDLP)]
    if node:
        cmd += ["--js-runtimes", f"node:{node}"]
    return cmd


def cmd_select(a):
    sel_p = CORPUS / "selection.json"
    sel = json.loads(sel_p.read_text()) if sel_p.exists() else {}
    for c in channels():
        key = slug(c["name"])
        if key in sel and not a.refresh:
            continue
        url = f"https://www.youtube.com/{c['ref']}/videos"
        out = subprocess.run(ytdlp_base() + ["--flat-playlist", "-J", "--playlist-end", "400", url],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(f"{c['name']}: listing failed: {out.stderr.strip()[-300:]}", flush=True)
            continue
        entries = json.loads(out.stdout).get("entries") or []
        picked, hours = [], 0.0
        for e in entries:  # newest first
            d = e.get("duration") or 0
            if d < MIN_S or e.get("live_status") in ("is_live", "is_upcoming"):
                continue
            picked.append({"id": e["id"], "title": e.get("title"), "duration": d})
            hours += d / 3600
            if hours >= QUOTA_H[c["lang"]]:
                break
        sel[key] = {"name": c["name"], "ref": c["ref"], "lang": c["lang"], "hours": round(hours, 1),
                    "videos": picked}
        CORPUS.mkdir(parents=True, exist_ok=True)
        tmp = sel_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(sel, indent=1, ensure_ascii=False))
        tmp.replace(sel_p)
        print(f"{c['name']} ({c['lang']}): {len(picked)} videos, {hours:.1f} h", flush=True)
    total = {lang: sum(v["hours"] for v in sel.values() if v["lang"] == lang) for lang in ("es", "en")}
    print(f"selection: es {total['es']:.0f} h, en {total['en']:.0f} h -> {sel_p}")


def cmd_download(a):
    sel = json.loads((CORPUS / "selection.json").read_text())
    for key, ch in sel.items():
        d = CORPUS / key
        d.mkdir(parents=True, exist_ok=True)
        ids = [v["id"] for v in ch["videos"]]
        print(f"{ch['name']}: {len(ids)} videos", flush=True)
        cmd = ytdlp_base() + [
            "-f", "bestaudio[ext=webm]/bestaudio",  # Opus when available; Whisper decodes it via ffmpeg
            "--download-archive", str(CORPUS / "archive.txt"),
            "--no-overwrites", "--continue", "--ignore-errors", "--no-progress",
            # IPv6 avoids the IPv4 CGNAT entirely, but WSL's mirrored IPv6 proved flaky (2026-11-14);
            # --ip 4 with one fragment is the low-load fallback.
            "--force-ipv6" if a.ip == 6 else "--force-ipv4",
            # Concurrent fragments: one stream is throttled to ~0.6 MB/s. Many parallel connections filled
            # the ISP's IPv4 CGNAT table on 2026-11-14 (IPv4 died for the whole PC), so few.
            "-N", str(a.fragments),
            "--sleep-requests", "1", "--sleep-interval", "2", "--max-sleep-interval", "6",
            "-o", str(d / "%(id)s.%(ext)s"), "--",
        ] + ids
        rc = subprocess.run(cmd).returncode
        if rc not in (0, 1):  # 1 = some videos failed (--ignore-errors); the archive keeps the rest
            sys.exit(f"yt-dlp failed for {ch['name']} (exit {rc})")


# ---------------------------------------------------------------------------- split + process

# Whole channels held out for the final test (0025 §6): one Spanish, one English.
TEST_CHANNELS = {"poco-se-habla", "lenny-s-podcast"}
SYM_LOCAL = [Path("~/yts-data/podcasts/Sastre_y_Maldonado_T1").expanduser(),
             Path("~/yts-data/podcasts/Sastre_y_Maldonado_T2").expanduser()]
SYM_SPLIT = Path("~/yts-data/podcasts/sym_split_0024.json").expanduser()  # 0024 split as YouTube ids
AUDIO_EXT = (".webm", ".m4a", ".opus", ".mp4", ".mkv", ".mp3")


def episodes() -> list[dict]:
    """Every corpus episode: selected downloads plus the local Sastre y Maldonado T1-T2 files."""
    out = []
    sel = json.loads((CORPUS / "selection.json").read_text())
    for key, ch in sel.items():
        for v in ch["videos"]:
            out.append({"id": v["id"], "channel": key, "lang": ch["lang"], "title": v["title"],
                        "duration": v["duration"], "dir": str(CORPUS / key), "owned": True})
    for d in SYM_LOCAL:
        for f in sorted(d.glob("*.webm")):
            m = re.search(r"\[([\w-]{11})\]\.\w+$", f.name)
            if m:
                out.append({"id": m.group(1), "channel": "sastre-y-maldonado", "lang": "es", "title": f.stem,
                            "duration": None, "dir": str(d), "file": str(f), "owned": False})
    return out


def make_split(eps: list[dict]) -> dict:
    """Frozen once written (0025 §6): test = 2 channels + the 0024 test episodes; val = 5 % (+ 0024 val)."""
    p = CORPUS / "split.json"
    if p.exists():
        return json.loads(p.read_text())
    import hashlib

    sym = json.loads(SYM_SPLIT.read_text())["split_0024"]
    split = {}
    for e in eps:
        if e["channel"] in TEST_CHANNELS or e["id"] in sym["test"]:
            split[e["id"]] = "test"
        elif e["id"] in sym["val"] or int(hashlib.sha1(e["id"].encode()).hexdigest(), 16) % 100 < 5:
            split[e["id"]] = "val"
        else:
            split[e["id"]] = "train"
    p.write_text(json.dumps(split, indent=1))
    return split


def audio_file(e: dict) -> Path | None:
    if e.get("file"):
        return Path(e["file"])
    for ext in AUDIO_EXT:
        f = Path(e["dir"]) / f"{e['id']}{ext}"
        if f.exists():
            return f
    return None


class Recorder:
    """Records every LLM call the app's tools make: the exact prompts become task-SFT data (0025 §9)."""

    def __init__(self):
        self.task = None
        self.calls: list[dict] = []
        self._patched_titles = False

    def wrap_chat(self, chat):
        if getattr(chat, "_m16_recorded", False):
            return
        orig = chat.complete

        def complete(messages, *, max_tokens, temperature=0.0, seed=0, json_schema=None):
            t0 = time.perf_counter()
            out = orig(messages, max_tokens=max_tokens, temperature=temperature, seed=seed, json_schema=json_schema)
            self.calls.append({"task": self.task, "messages": messages, "output": out, "max_tokens": max_tokens,
                               "temperature": temperature, "seed": seed, "json_schema": json_schema,
                               "seconds": round(time.perf_counter() - t0, 2)})
            return out

        chat.complete = complete
        chat._m16_recorded = True

    def patch_titles(self):
        if self._patched_titles:
            return
        from app.chapters import llm_titles

        orig = llm_titles.LlamaTitler.title
        rec = self

        def title(self_, text, lang, previous=None, lid=None):
            res = orig(self_, text, lang, previous, lid)
            msgs = llm_titles.build_messages(llm_titles.sample_text(text, self_.max_words), lang, previous)
            rec.calls.append({"task": "title", "messages": msgs, "output": res.title, "raw": res.raw,
                              "attempts": res.attempts, "max_tokens": self_.max_tokens, "seconds": round(res.seconds, 2)})
            return res

        llm_titles.LlamaTitler.title = title
        self._patched_titles = True


def _backend():
    """Import the app from the WSL clone (its .env and models/ resolve relative to backend/)."""
    backend = Path(os.environ.get("M16_BACKEND", "~/projects/new_youtube_suite/backend")).expanduser()
    os.chdir(backend)
    sys.path.insert(0, str(backend))


def process_episode(e: dict, rec: Recorder, out: Path) -> dict:
    from app.chapters.pipeline import generate_chapters
    from app.chapters.units import Atom
    from app.llm import runtime
    from app.relevance import pipeline as relevance
    from app.relevance.windows import Word
    from app.shownotes import pipeline as shownotes
    from app.shownotes.chunking import ChapterMark, Seg
    from app.transcription.whisper import transcribe

    meta = {"id": e["id"], "channel": e["channel"], "lang_hint": e["lang"], "title": e["title"], "steps": {}}
    tp = out / "transcript.json"
    t0 = time.perf_counter()
    if tp.exists():
        segs = json.loads(tp.read_text())
    else:
        segs = [{"start": s.start, "end": s.end, "text": s.text,
                 "words": [{"w": w.word, "s": w.start, "e": w.end} for w in s.words]}
                for s in transcribe(audio_file(e))]
        tmp = tp.with_suffix(".tmp")
        tmp.write_text(json.dumps(segs, ensure_ascii=False))
        tmp.replace(tp)
    meta["steps"]["transcribe_s"] = round(time.perf_counter() - t0, 1)
    duration = max((s["end"] for s in segs), default=0.0)
    meta["duration"] = duration
    rec.calls = []
    rec.patch_titles()

    rec.task = "title"
    t0 = time.perf_counter()
    atoms = [Atom(w["s"], w["e"], w["w"]) for s in segs for w in s["words"]] or [
        Atom(s["start"], s["end"], s["text"]) for s in segs]
    version, chapters = generate_chapters(atoms, duration)
    meta["steps"]["chapters_s"] = round(time.perf_counter() - t0, 1)
    meta["chapters_version"] = version
    chap = [{"start": c.start, "title": c.title} for c in chapters]

    t0 = time.perf_counter()
    gen = shownotes.get_generator()
    rec.wrap_chat(gen.llm)
    rec.task = "shownotes"
    model_used, notes = shownotes.generate_show_notes([Seg(s["start"], s["end"], s["text"]) for s in segs],
                                                      [ChapterMark(c["start"], c["title"]) for c in chap])
    meta["steps"]["shownotes_s"] = round(time.perf_counter() - t0, 1)
    d = notes.draft
    sn = {"model": model_used, "language": notes.language, "title_suggestions": d.title_suggestions,
          "description": d.description, "tags": d.tags, "summary": d.summary}

    t0 = time.perf_counter()
    scorer = relevance.get_scorer()
    rec.wrap_chat(scorer.llm)
    rec.task = "relevance"
    words = [Word(w["s"], w["w"]) for s in segs for w in s["words"]] or [Word(s["start"], s["text"]) for s in segs]
    label, _desc, res = relevance.score_relevance(words, duration, [(c["start"], c["title"]) for c in chap])
    meta["steps"]["relevance_s"] = round(time.perf_counter() - t0, 1)
    rel = {"version": label, "language": res.language if hasattr(res, "language") else None,
           "windows": [{"i": s.window.index, "start": s.window.start, "end": s.window.end, "text": s.window.text,
                        "heuristic": s.heuristic, "llm": s.llm, "score": s.score} for s in res.scores]}
    runtime.release_all()

    for name, obj in (("chapters.json", chap), ("shownotes.json", sn), ("relevance.json", rel)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False))
    with open(out / "calls.jsonl", "w", encoding="utf-8") as f:
        for c in rec.calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta["llm_calls"] = {t: sum(1 for c in rec.calls if c["task"] == t) for t in ("title", "shownotes", "relevance")}
    return meta


def cmd_process(a):
    import signal

    eps = episodes()
    split = make_split(eps)
    _backend()
    rec = Recorder()
    stop = {"flag": False}

    def handler(signum, _f):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        print("Stopping after the current episode (signal again to abort it).", flush=True)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    root = CORPUS / "processed"
    while True:
        pending = [e for e in eps if not (root / e["id"] / "meta.json").exists()]
        ready = [e for e in pending if audio_file(e) or (root / e["id"] / "transcript.json").exists()]
        if not pending:
            print("All episodes processed.", flush=True)
            return
        if not ready:
            if not a.wait:
                print(f"{len(pending)} episodes still have no audio; re-run when the download finishes.")
                return
            time.sleep(60)
            continue
        for e in ready:
            if stop["flag"] or (CORPUS / "PAUSE").exists():
                print("Stopped (resumable: python -m llm.corpus process).", flush=True)
                return
            out = root / e["id"]
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            try:
                meta = process_episode(e, rec, out)
            except Exception as ex:  # one bad episode must not stop the corpus; it is retried next run
                print(f"{e['id']} ({e['channel']}): FAILED {type(ex).__name__}: {ex}", flush=True)
                (out / "error.txt").write_text(f"{type(ex).__name__}: {ex}")
                from app.llm import runtime

                runtime.release_all()
                continue
            meta["split"] = split.get(e["id"], "train")
            meta["seconds"] = round(time.perf_counter() - t0, 1)
            (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            (out / "error.txt").unlink(missing_ok=True)
            f = audio_file(e)
            if e["owned"] and f and not a.keep_audio:  # transcribed: the audio is no longer needed
                f.unlink()
            done = sum(1 for x in eps if (root / x["id"] / "meta.json").exists())
            print(f"{e['id']} ({e['channel']}, {meta['duration'] / 60:.0f} min): {meta['seconds']:.0f}s "
                  f"{meta['steps']} calls {meta['llm_calls']} [{done}/{len(eps)}]", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m llm.corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(fn=cmd_select)
    d = sub.add_parser("download")
    d.add_argument("--ip", type=int, choices=(4, 6), default=6)
    d.add_argument("--fragments", type=int, default=2)
    d.set_defaults(fn=cmd_download)
    p = sub.add_parser("process", help="transcribe + Qwen teacher per episode (GPU); resumable")
    p.add_argument("--wait", action="store_true", help="keep polling for audio still being downloaded")
    p.add_argument("--keep-audio", action="store_true")
    p.set_defaults(fn=cmd_process)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
