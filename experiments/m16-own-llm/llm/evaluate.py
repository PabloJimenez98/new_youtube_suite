"""A3 functional gates (0025 §10) on the frozen test split, through the app's own pipelines.

    python -m llm.evaluate run --model own --label LABEL [--own-dir DIR] [--tasks titles,shownotes,relevance]
    python -m llm.evaluate run --model qwen --label qwen --tasks relevance      # the Qwen reference
    python -m llm.evaluate gates --label LABEL [--ref qwen]                      # -> eval/LABEL/gates.json

Inputs: the processed corpus episodes whose meta.json has split "test" (49 episodes, two
held-out channels + the 0024 test episodes); their Whisper transcripts and Qwen outputs were
written by `llm.corpus process`. Nothing here trains on them.

Per episode (eval/LABEL/<id>.json):
- titles: the chapter tool with the chosen LLM (same boundary model, so the same chapters as
  Qwen's run). Per chapter: the LLM title or None when the app fell back to mT5, attempts, LID.
- shownotes: the show-notes tool with the chosen LLM over those chapters: valid draft or the
  error after the 3 reduce attempts, how many reduce attempts it took, LID of the description.
- relevance: the zero-shot scorer's two passes kept separately (the app averages them), with
  Qwen's chapter titles as context for both models so only the scorer differs.

Gates (0025 §10, "selectable"):
- titles: <= 10 % of chapters fall back to mT5; >= 95 % of titles in the episode's language.
- show notes: >= 95 % of episodes get a valid draft within the 3 attempts; >= 95 % right language.
- relevance: 100 % valid batches; mean per-episode Spearman(own, Qwen) of the window scores
  >= 0.8 x Qwen's own pass-to-pass Spearman (pass 0 vs pass 1), both measured here.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from llm import HOME
from llm.corpus import CORPUS, _backend

EVAL = HOME / "eval"
TASKS = ("titles", "shownotes", "relevance")


def test_episodes(split: str = "test") -> list[Path]:
    out = []
    for meta_p in sorted((CORPUS / "processed").glob("*/meta.json")):
        meta = json.loads(meta_p.read_text())
        if meta.get("split") == split and (meta_p.parent / "transcript.json").exists():
            out.append(meta_p.parent)
    return out


def _lid():
    from app.chapters.text import LanguageId
    from app.config import get_settings

    bundle = Path(get_settings().chapters_model_dir)
    titles = json.loads((bundle / "manifest.json").read_text())["titles"]
    return LanguageId(bundle / titles["lid"]["file"])


def run_titles(atoms, duration, llm: str, lid, lang: str) -> dict:
    from app.chapters import llm_titles
    from app.chapters.pipeline import get_chapterer

    results = []
    orig = llm_titles.LlamaTitler.title

    def title(self_, text, lang_, previous=None, lid=None):
        r = orig(self_, text, lang_, previous, lid)
        results.append({"title": r.title, "attempts": r.attempts, "raw": r.raw})
        return r

    llm_titles.LlamaTitler.title = title
    try:
        t0 = time.perf_counter()
        chapters = get_chapterer().run(atoms, duration, llm)
        secs = time.perf_counter() - t0
    finally:
        llm_titles.LlamaTitler.title = orig
    out = []
    for i, c in enumerate(chapters):
        r = results[i] if i < len(results) else {"title": None, "attempts": 0, "raw": []}
        pred, conf = lid(c.title)
        out.append({"start": c.start, "title": c.title, "llm_title": r["title"], "fallback": r["title"] is None,
                    "attempts": r["attempts"], "raw": r["raw"], "lid": pred, "lid_conf": round(conf, 3),
                    "lang_ok": pred == lang})
    return {"seconds": round(secs, 1), "chapters": out}


def run_shownotes(segs, chapters, llm: str, lid, lang: str) -> dict:
    from app.shownotes import pipeline as shownotes
    from app.shownotes.chunking import ChapterMark, Seg

    gen = shownotes.get_generator(llm)
    reduce_calls = {"n": 0}
    orig = gen.llm.complete

    def complete(messages, **kw):
        if kw.get("json_schema") is not None:
            reduce_calls["n"] += 1
        return orig(messages, **kw)

    gen.llm.complete = complete
    t0 = time.perf_counter()
    try:
        _, notes = shownotes.generate_show_notes([Seg(s["start"], s["end"], s["text"]) for s in segs],
                                                 [ChapterMark(c["start"], c["title"]) for c in chapters], llm)
        d = notes.draft
        pred, conf = lid(d.summary or d.description)
        res = {"ok": True, "attempts": reduce_calls["n"], "lid": pred, "lid_conf": round(conf, 3),
               "lang_ok": pred == lang,
               "draft": {"title_suggestions": d.title_suggestions, "description": d.description,
                         "tags": d.tags, "summary": d.summary}}
    except shownotes.ShowNotesGenerationError as e:
        res = {"ok": False, "attempts": reduce_calls["n"], "error": str(e)[:2000], "lang_ok": False}
    finally:
        gen.llm.complete = orig
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


def run_relevance(words, duration, chapters, llm: str) -> dict:
    """RelevanceScorer.run, keeping the passes apart (app.relevance.pipeline)."""
    import bisect

    from app.relevance import pipeline as relevance
    from app.relevance.llm_scoring import PASS_OFFSETS, block_scores, window_llm_scores
    from app.relevance.windows import build_blocks, build_windows

    scorer = relevance.get_scorer(llm)
    text = " ".join(w.text for w in words)
    detected, _ = scorer.lid(text[:5000])
    lang = "es" if detected == "es" else "en"
    blocks = build_blocks(words, duration)
    windows = build_windows(blocks)
    starts = [s for s, _ in chapters]

    def chapter_at(t):
        i = bisect.bisect_right(starts, t) - 1
        return chapters[i][1] if i >= 0 else None

    t0 = time.perf_counter()
    passes, batches, failed = [], 0, 0
    for off in PASS_OFFSETS:
        bl, b, f = block_scores(scorer.llm, blocks, lang, chapter_at, off)
        batches, failed = batches + b, failed + f
        passes.append(window_llm_scores(bl, len(windows), off))
    return {"lang": lang, "batches": batches, "failed": failed, "passes": passes,
            "seconds": round(time.perf_counter() - t0, 1)}


def cmd_run(a):
    if a.own_dir:
        os.environ["LLM_OWN_DIR"] = str(Path(a.own_dir).expanduser())
    _backend()
    from app.chapters.units import Atom
    from app.llm import runtime
    from app.relevance.windows import Word

    tasks = a.tasks.split(",")
    out_dir = EVAL / a.label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps({"model": a.model, "own_dir": a.own_dir, "tasks": tasks,
                                                  "started": time.strftime("%F %T")}, indent=1))
    lid = _lid()
    eps = test_episodes()[: a.limit or None]
    for n, d in enumerate(eps, 1):
        res_p = out_dir / f"{d.name}.json"
        res = json.loads(res_p.read_text()) if res_p.exists() else {"id": d.name}
        todo = [t for t in tasks if t not in res]
        if not todo:
            continue
        meta = json.loads((d / "meta.json").read_text())
        segs = json.loads((d / "transcript.json").read_text())
        duration = meta.get("duration") or max((s["end"] for s in segs), default=0.0)
        text = " ".join(s["text"] for s in segs)
        det, _ = lid(text[:5000])
        lang = "es" if det == "es" else "en"
        res.update({"channel": meta["channel"], "lang": lang, "duration": duration})
        qwen_chapters = json.loads((d / "chapters.json").read_text())
        t0 = time.perf_counter()
        if "titles" in todo:
            atoms = [Atom(w["s"], w["e"], w["w"]) for s in segs for w in s["words"]] or [
                Atom(s["start"], s["end"], s["text"]) for s in segs]
            res["titles"] = run_titles(atoms, duration, a.model, lid, lang)
        if "shownotes" in todo:
            chaps = [{"start": c["start"], "title": c["title"]} for c in res["titles"]["chapters"]] \
                if "titles" in res else qwen_chapters
            res["shownotes"] = run_shownotes(segs, chaps, a.model, lid, lang)
        if "relevance" in todo:
            words = [Word(w["s"], w["w"]) for s in segs for w in s["words"]] or [
                Word(s["start"], s["text"]) for s in segs]
            res["relevance"] = run_relevance(words, duration, [(c["start"], c["title"]) for c in qwen_chapters],
                                             a.model)
        tmp = res_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False))
        tmp.replace(res_p)
        print(f"[{n}/{len(eps)}] {d.name} {meta['channel']} {lang} {time.perf_counter() - t0:.0f}s", flush=True)
    runtime.release_all()


def spearman(a: list, b: list) -> float | None:
    from scipy.stats import spearmanr

    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    r = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    return None if r != r else float(r)  # NaN (a constant series) -> None


def mean_passes(passes):
    out = []
    for vals in zip(*passes):
        v = [x for x in vals if x is not None]
        out.append(sum(v) / len(v) if v else None)
    return out


def gates(label: str, ref: str = "qwen") -> dict:
    d, rd = EVAL / label, EVAL / ref
    eps = [json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if p.name not in ("run.json", "gates.json")]
    chs = [c for e in eps if "titles" in e for c in e["titles"]["chapters"]]
    sn = [e["shownotes"] for e in eps if "shownotes" in e]
    rel = [e for e in eps if "relevance" in e]
    out = {"label": label, "episodes": len(eps), "measured": time.strftime("%F %T")}
    if chs:
        fb = sum(c["fallback"] for c in chs) / len(chs)
        lang = sum(c["lang_ok"] for c in chs) / len(chs)
        out["titles"] = {"chapters": len(chs), "fallback_rate": round(fb, 4), "lang_ok_rate": round(lang, 4),
                         "pass": fb <= 0.10 and lang >= 0.95}
    if sn:
        ok = sum(s["ok"] for s in sn) / len(sn)
        lang = sum(s["lang_ok"] for s in sn) / len(sn)
        out["shownotes"] = {"episodes": len(sn), "valid_rate": round(ok, 4), "lang_ok_rate": round(lang, 4),
                            "mean_reduce_attempts": round(statistics.mean(s["attempts"] for s in sn), 2),
                            "pass": ok >= 0.95 and lang >= 0.95}
    if rel:
        b = sum(e["relevance"]["batches"] for e in rel)
        f = sum(e["relevance"]["failed"] for e in rel)
        r_own, r_p2p = [], []
        for e in rel:
            rp = rd / f"{e['id']}.json"
            if not rp.exists():
                continue
            q = json.loads(rp.read_text())["relevance"]
            s = spearman(mean_passes(e["relevance"]["passes"]), mean_passes(q["passes"]))
            p = spearman(q["passes"][0], q["passes"][1])
            if s is not None and p is not None:
                r_own.append(s)
                r_p2p.append(p)
        own_mean = statistics.mean(r_own) if r_own else None
        p2p_mean = statistics.mean(r_p2p) if r_p2p else None
        out["relevance"] = {"batches": b, "failed": f, "valid_rate": round(1 - f / b, 4) if b else None,
                            "episodes_compared": len(r_own),
                            "spearman_vs_qwen": None if own_mean is None else round(own_mean, 4),
                            "qwen_pass_to_pass": None if p2p_mean is None else round(p2p_mean, 4),
                            "pass": bool(b and f == 0 and own_mean is not None and p2p_mean is not None
                                         and own_mean >= 0.8 * p2p_mean)}
    (d / "gates.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m llm.evaluate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=("own", "qwen"), required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--own-dir", help="LLM_OWN_DIR for this run (a directory with manifest.json + GGUF)")
    r.add_argument("--tasks", default=",".join(TASKS))
    r.add_argument("--limit", type=int, default=0)
    g = sub.add_parser("gates")
    g.add_argument("--label", required=True)
    g.add_argument("--ref", default="qwen")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    else:
        print(json.dumps(gates(a.label, a.ref), indent=1))


if __name__ == "__main__":
    sys.exit(main())
