"""Task DPO pairs (0025 §9.3): teacher output (chosen) vs. our SFT model's failing sample (rejected).

    python -m llm.dpo_pairs build --own-dir DIR [--per-task N] [--out FILE] [--tasks T,..] [--repeats K]

--repeats K samples each chosen prompt K times with new seeds (used for shownotes_reduce, which
has only 260 train prompts, 2026-11-18).

For a sample of the task-SFT prompts (train and val splits, never test), our exported SFT model
(llama.cpp, through the app's LlamaChat, with the same max_tokens and JSON schema the app uses)
answers at temperature 0.7. The answer is checked by the §9.4 verifiers; when it fails, the pair
is (chosen = the Qwen teacher's output, rejected = our answer). Answers that pass are not used:
with no judge there is no basis to say the teacher's is better. Verifiers:
- title: the app's postprocess gives a title, <= 8 words, in the episode's language (LID).
- shownotes_map: non-empty, in the episode's language, not degenerate (distinct-3-gram ratio >= 0.5).
- shownotes_reduce: `parse_draft` accepts it and the description is in the episode's language.
- relevance: valid scores (the grammar makes this near certain), not constant, and Spearman with
  the teacher's scores >= 0.3.
The language is the LID of the prompt's transcript text, as the app's tools detect it.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from llm import HOME
from llm.corpus import _backend

SFT = HOME / "data" / "sft"


def distinct3(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i : i + 3]) for i in range(len(w) - 2)]
    return len(set(grams)) / len(grams) if grams else 0.0


def prompt_text(ex: dict) -> str:
    return " ".join(m["content"] for m in ex["messages"] if m["role"] == "user")


def verify(task: str, out: str, teacher: str, lang: str, lid, schema) -> str | None:
    """None when our answer passes, else the reason it fails."""
    from app.chapters.llm_titles import postprocess, wrong_language
    from app.relevance.llm_scoring import ScoresInvalid, parse_scores
    from app.shownotes.validate import DraftInvalid, parse_draft

    if task == "title":
        t = postprocess(out)
        if not t:
            return "no title"
        if len(t.split()) > 8:
            return "too long"
        if wrong_language(t, lang, lid):
            return "wrong language"
        return None
    if task == "shownotes_map":
        if not out.strip():
            return "empty"
        if lid(out)[0] != lang:
            return "wrong language"
        if distinct3(out) < 0.5:
            return "repetitive"
        return None
    if task == "shownotes_reduce":
        try:
            d = parse_draft(out)
        except DraftInvalid as e:
            return f"invalid draft ({str(e)[:60]})"
        if wrong_language(d.description, lang, lid):
            return "wrong language"
        return None
    if task == "relevance":
        from llm.evaluate import spearman

        n = schema["properties"]["scores"]["minItems"]
        try:
            ours = parse_scores(out, n)
            theirs = parse_scores(teacher, n)
        except ScoresInvalid as e:
            return f"invalid ({str(e)[:60]})"
        if len(set(ours)) == 1:
            return "constant"
        r = spearman(ours, theirs)
        if r is None or r < 0.3:
            return f"disagrees with teacher (spearman {r if r is None else round(r, 2)})"
        return None
    raise ValueError(task)


def cmd_build(a):
    os.environ["LLM_OWN_DIR"] = str(Path(a.own_dir).expanduser())
    _backend()
    from app.config import get_settings
    from app.llm import registry, runtime
    from app.llm.chat import LlamaChat
    from llm.evaluate import _lid

    lid = _lid()
    spec = registry.resolve("own")
    chat = LlamaChat(runtime.get_llama(spec.path, spec.n_ctx, get_settings().chapters_llm_threads), spec.n_ctx)
    rng = random.Random(0)
    out_p = Path(a.out).expanduser()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    reasons = Counter()
    t0 = time.perf_counter()
    with open(out_p, "w", encoding="utf-8") as f:
        for split in ("train", "val"):
            exs = [json.loads(l) for l in open(SFT / "tasks" / f"tasks_{split}.jsonl", encoding="utf-8")]
            by_task = {}
            for ex in exs:
                by_task.setdefault(ex["task"], []).append(ex)
            for task, items in sorted(by_task.items()):
                if a.tasks and task not in a.tasks.split(","):
                    continue
                n = a.per_task if split == "train" else max(1, a.per_task // 10)
                chosen = [ex for _ in range(a.repeats) for ex in rng.sample(items, min(n, len(items)))]
                for ex in chosen:
                    prompt = ex["messages"][:-1]
                    teacher = ex["messages"][-1]["content"]
                    det, _ = lid(prompt_text(ex)[:5000])
                    lang = "es" if det == "es" else "en"
                    max_tokens = {"title": 32, "shownotes_map": 256, "shownotes_reduce": 900}.get(
                        task, 8 * ex.get("json_schema", {}).get("properties", {}).get("scores", {}).get("minItems", 20) + 24)
                    try:
                        out = chat.complete(prompt, max_tokens=max_tokens, temperature=0.7, seed=rng.randrange(1 << 30),
                                            json_schema=ex.get("json_schema"))
                    except ValueError:  # prompt + max_tokens over n_ctx
                        stats[f"{split}_{task}_skipped"] += 1
                        continue
                    why = verify(task, out, teacher, lang, lid, ex.get("json_schema"))
                    stats[f"{split}_{task}_sampled"] += 1
                    if why is None:
                        stats[f"{split}_{task}_passed"] += 1
                        continue
                    reasons[f"{task}: {re.sub(r'[0-9.-]+', '#', why)}"] += 1
                    stats[f"{split}_{task}_pairs"] += 1
                    f.write(json.dumps({"messages": prompt, "chosen": teacher, "rejected": out, "task": task,
                                        "reason": why, "split": split, "episode": ex["episode"]},
                                       ensure_ascii=False) + "\n")
                print(f"{split} {task}: {dict((k, v) for k, v in stats.items() if k.startswith(f'{split}_{task}'))}"
                      f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    runtime.release_all()
    summary = {"stats": dict(sorted(stats.items())), "reasons": dict(reasons.most_common()),
               "own_dir": a.own_dir, "seconds": round(time.perf_counter() - t0)}
    out_p.with_suffix(".stats.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m llm.dpo_pairs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--own-dir", required=True)
    b.add_argument("--per-task", type=int, default=1000)
    b.add_argument("--out", default=str(SFT / "dpo_tasks" / "pairs.jsonl"))
    b.add_argument("--tasks", default="", help="only these tasks (comma-separated)")
    b.add_argument("--repeats", type=int, default=1, help="sample each chosen prompt this many times (new seeds)")
    a = ap.parse_args(argv)
    cmd_build(a)


if __name__ == "__main__":
    main()
