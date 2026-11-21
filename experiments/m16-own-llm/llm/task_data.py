"""Task-SFT examples from the recorded Qwen teacher calls (0025 §6, §9).

    python -m llm.task_data build [--out DIR]   # -> tasks_train.jsonl, tasks_val.jsonl, tasks_test.jsonl, stats

Each processed episode (llm.corpus process) has calls.jsonl with every LLM call the app's
tools made: {task, messages, output, json_schema, ...}. One call = one example:
messages + the teacher's output as the assistant turn, i.e. exactly the prompt the app
will send to our model. Only usable outputs are kept:
- title: a non-empty cleaned title (the titler's own post-processing already ran);
- shownotes map (no schema): non-empty notes;
- shownotes reduce / relevance (JSON schema): output that parses, and for relevance has
  exactly the number of integers the schema asks for.
Retries are recorded too, so a failed first attempt is dropped and its successful retry kept.
The split comes from the episode's meta.json (frozen corpus split, 0025 §6).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from llm.corpus import CORPUS


def usable(call: dict) -> bool:
    out = call.get("output")
    if not isinstance(out, str) or not out.strip():
        return False
    schema = call.get("json_schema")
    if schema is None:
        return True
    try:
        obj = json.loads(out)
    except ValueError:
        return False
    if call["task"] == "relevance":
        n = schema["properties"]["scores"]["minItems"]
        scores = obj.get("scores") if isinstance(obj, dict) else None
        return isinstance(scores, list) and len(scores) == n and all(isinstance(s, int) for s in scores)
    return isinstance(obj, dict)


def subtask(call: dict) -> str:
    if call["task"] == "shownotes":
        return "shownotes_reduce" if call.get("json_schema") is not None else "shownotes_map"
    return call["task"]


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"tasks_{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    stats = Counter()
    try:
        for meta_p in sorted((CORPUS / "processed").glob("*/meta.json")):
            meta = json.loads(meta_p.read_text())
            split = meta.get("split", "train")
            calls_p = meta_p.parent / "calls.jsonl"
            if not calls_p.exists():
                continue
            stats[f"episodes_{split}"] += 1
            for line in calls_p.read_text(encoding="utf-8").splitlines():
                call = json.loads(line)
                st = subtask(call)
                if not usable(call):
                    stats[f"dropped_{st}"] += 1
                    continue
                ex = {"messages": call["messages"] + [{"role": "assistant", "content": call["output"].strip()}],
                      "task": st, "episode": meta["id"], "channel": meta["channel"], "split": split}
                if call.get("json_schema"):
                    ex["json_schema"] = call["json_schema"]
                files[split].write(json.dumps(ex, ensure_ascii=False) + "\n")
                stats[f"{split}_{st}"] += 1
    finally:
        for f in files.values():
            f.close()
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
    return dict(sorted(stats.items()))


def main(argv=None):
    from llm import HOME

    ap = argparse.ArgumentParser(prog="python -m llm.task_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(HOME / "data" / "sft" / "tasks"))
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.out)), indent=1))


if __name__ == "__main__":
    main()
