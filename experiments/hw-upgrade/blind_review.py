"""Blind A/B sheet for the owner: two chapter bundles that differ only in
their LLM (e.g. Qwen3.5-4B vs 9B), run through the real app pipelines
(chapter titles, show notes, relevance top-5) on episodes already
transcribed in the dev DB (docs/decisions/0016, checkpoints C3-C5).

    cd backend && <venv>/python ../experiments/hw-upgrade/blind_review.py \
        BUNDLE_X BUNDLE_Y OUT_DIR UPLOAD_ID [UPLOAD_ID ...]

Writes OUT_DIR/blind_sheet.md (options shuffled per episode and per
section) and OUT_DIR/blind_key.json (which option is which bundle, plus
timings). Read the sheet before opening the key.
"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.chapters import pipeline as chapters_pipeline
from app.config import get_settings
from app.database import async_session_factory
from app.llm import runtime
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.relevance import pipeline as relevance_pipeline
from app.shownotes import pipeline as shownotes_pipeline
from app.shownotes.chunking import ChapterMark, Seg
from app.worker.handlers.chapters import atoms_from_segments
from app.worker.handlers.relevance import words_from_segments

TOP_K, MIN_GAP_S = 5, 60.0


async def load(upload_id: int):
    async with async_session_factory() as db:
        upload = await db.get(Upload, upload_id)
        segments = list((await db.execute(
            select(TranscriptSegment).where(TranscriptSegment.upload_id == upload_id)
            .order_by(TranscriptSegment.segment_index).options(selectinload(TranscriptSegment.words))
        )).scalars())
    duration = upload.duration_seconds or max(s.end_seconds for s in segments)
    return upload.filename, segments, duration


def top_windows(result, k=TOP_K, gap=MIN_GAP_S):
    picked = []
    for s in sorted(result.scores, key=lambda s: s.score, reverse=True):
        if all(abs(s.window.start - p.window.start) >= gap for p in picked):
            picked.append(s)
        if len(picked) == k:
            break
    return sorted(picked, key=lambda s: s.window.start)


def run_bundle(bundle: Path, segments, duration):
    settings = get_settings()
    settings.chapters_model_dir = bundle
    runtime.release_all()
    out, t = {}, time.time()
    version, chapters = chapters_pipeline.generate_chapters(atoms_from_segments(segments), duration)
    out["chapters"] = [(c.start, c.title) for c in chapters]
    out["t_chapters"] = time.time() - t
    t = time.time()
    model, notes = shownotes_pipeline.generate_show_notes(
        [Seg(s.start_seconds, s.end_seconds, s.text) for s in segments],
        [ChapterMark(s, title) for s, title in out["chapters"]])
    out["notes"] = notes.draft
    out["t_notes"] = time.time() - t
    t = time.time()
    label, _desc, result = relevance_pipeline.score_relevance(words_from_segments(segments), duration, out["chapters"])
    out["top"] = top_windows(result)
    out["t_relevance"] = time.time() - t
    out["labels"] = {"chapters": version, "shownotes": model, "relevance": label}
    runtime.release_all()
    return out


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def main(bundle_x: str, bundle_y: str, out_dir: str, *upload_ids: str) -> None:
    rng = random.Random(20261107)
    bundles = {"X": Path(bundle_x), "Y": Path(bundle_y)}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sheet = ["# Blind review: chapter titles, show notes, relevance top-5", "",
             "For each episode and section, pick the better option (1 or 2) and, if you like, say why.",
             "Options are shuffled independently per episode and per section. The key is in blind_key.json.", ""]
    key: dict = {"bundles": {k: str(v) for k, v in bundles.items()}, "episodes": {}}
    async def load_all():  # one event loop: the engine's pool is bound to it
        return [await load(uid) for uid in map(int, upload_ids)]

    for name, segments, duration in asyncio.run(load_all()):
        res = {k: run_bundle(b, segments, duration) for k, b in bundles.items()}
        ep_key = key["episodes"][name] = {"timings": {k: {t: round(v[t], 1) for t in ("t_chapters", "t_notes", "t_relevance")}
                                                       for k, v in res.items()},
                                           "labels": {k: v["labels"] for k, v in res.items()}}
        sheet += [f"## {name}", ""]
        for section in ("chapters", "notes", "top"):
            order = ["X", "Y"]; rng.shuffle(order)
            ep_key[section] = {"1": order[0], "2": order[1]}
            if section == "chapters":
                sheet += ["### Chapter titles", ""]
                for i, k in enumerate(order, 1):
                    sheet += [f"**Option {i}**", ""] + [f"- {mmss(s)} {t}" for s, t in res[k]["chapters"]] + [""]
            elif section == "notes":
                sheet += ["### Show notes", ""]
                for i, k in enumerate(order, 1):
                    d = res[k]["notes"]
                    sheet += [f"**Option {i}**", "", "Titles: " + " / ".join(d.title_suggestions), "",
                              "Summary: " + d.summary, "", "Description:", "", "```", d.description, "```", "",
                              "Tags: " + ", ".join(d.tags), ""]
            else:
                sheet += ["### Best moments for shorts (relevance top-5)", ""]
                for i, k in enumerate(order, 1):
                    sheet += [f"**Option {i}**", ""] + [
                        f"- {mmss(s.window.start)}-{mmss(s.window.end)} (score {s.score:.2f}): {s.window.text[:300]}"
                        for s in res[k]["top"]] + [""]
        print(name, json.dumps(ep_key["timings"]), flush=True)
    (out / "blind_sheet.md").write_text("\n".join(sheet), encoding="utf-8")
    (out / "blind_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out / "blind_sheet.md")


if __name__ == "__main__":
    main(*sys.argv[1:])
