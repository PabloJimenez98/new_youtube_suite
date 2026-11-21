"""General post-training data (0025 §9 steps 1 and 3): open instruction and preference datasets.

    python -m llm.general_sft fetch [NAME ...]   # download the source parquet files (resumable)
    python -m llm.general_sft build [NAME ...]   # -> data/sft/general/<name>.jsonl, deletes the download

The selection, licences and mix are recorded in notebook 05 (05_general_sft_data.ipynb).
Downloads use plain HTTP, one file at a time, over IPv4 (the CGNAT incident, TRAINING.md §9):
run with HF_HUB_DISABLE_XET=1.

Output (one JSON object per line):
    SFT sources: {"messages": [{"role", "content"}, ...], "source", "lang"}   (llm.sft_data input)
    DPO sources: {"messages": [prompt turns], "chosen", "rejected", "source", "lang", "split"}  (llm.dpo input)
Only user/assistant/system turns are kept; examples that don't alternate cleanly or end on a user
turn are dropped. Language: the dataset's own label where it has one, else fastText LID (lid.176).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from llm import HOME

DL = HOME / "data" / "download" / "general"
OUT = HOME / "data" / "sft" / "general"

SOURCES = {
    "smoltalk2-ml": {
        "repo": "HuggingFaceTB/smoltalk2", "kind": "sft", "licence": "Apache-2.0",
        "files": [f"SFT/smoltalk_multilingual_8languages_lang_5_no_think-0000{i}-of-00002.parquet"
                  for i in range(2)],
        "keep_langs": {"es"},
    },
    "aya": {
        "repo": "CohereLabs/aya_dataset", "kind": "sft", "licence": "Apache-2.0",
        "files": ["data/train-00000-of-00001.parquet"],
        "keep_langs": {"es", "en"},
    },
    "openhermes-es": {
        "repo": "Iker/OpenHermes-2.5-Spanish", "kind": "sft",
        "licence": "Apache-2.0 (machine translation of teknium/OpenHermes-2.5, GPT-4 outputs)",
        "files": None,  # listed from the repo: data/train-*.parquet
        "prefix": "data/train-",
        "keep_langs": {"es"},
    },
    "smol-smoltalk": {
        "repo": "HuggingFaceTB/smol-smoltalk", "kind": "sft", "licence": "Apache-2.0",
        "files": [f"data/train-0000{i}-of-00004.parquet" for i in range(4)],
        "keep_langs": {"en"},
    },
    "ultrafeedback": {
        "repo": "HuggingFaceH4/ultrafeedback_binarized", "kind": "dpo", "licence": "MIT",
        "files": ["data/train_prefs-00000-of-00001.parquet", "data/test_prefs-00000-of-00001.parquet"],
        "keep_langs": {"en"},
    },
}

ROLE = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
        "system": "system", "Prompter": "user", "Assistant": "assistant"}
AYA_LANG = {"Spanish": "es", "English": "en"}


def files_of(name: str) -> list[str]:
    s = SOURCES[name]
    if s["files"] is not None:
        return s["files"]
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(s["repo"], path_in_repo="data", repo_type="dataset")
    return sorted(f.path for f in tree if f.path.startswith(s["prefix"]) and f.path.endswith(".parquet"))


def cmd_fetch(a):
    from huggingface_hub import hf_hub_download

    for name in a.names or list(SOURCES):
        if (OUT / f"{name}.jsonl").exists():
            print(f"[{name}] already built; skipping", flush=True)
            continue
        for remote in files_of(name):
            for attempt in range(5):
                try:
                    t0 = time.time()
                    p = Path(hf_hub_download(SOURCES[name]["repo"], remote, repo_type="dataset",
                                             local_dir=DL / name))
                    print(f"[{name}] {remote}: {p.stat().st_size / 1e6:,.0f} MB in {time.time() - t0:.0f}s",
                          flush=True)
                    break
                except Exception as e:  # network hiccups: back off and retry
                    if attempt == 4:
                        raise
                    print(f"[{name}] {remote}: {type(e).__name__} {e}; retrying", flush=True)
                    time.sleep(30 * (attempt + 1))


# ---------------------------------------------------------------------------- build

_lid = None


def lid(text: str) -> str:
    """fastText lid.176 language code of a text (first 1,000 chars, newlines removed)."""
    global _lid
    if _lid is None:
        import fasttext

        _lid = fasttext.load_model(str(lid_path()))
    labels, _ = _lid.predict(text[:1000].replace("\n", " "))
    return labels[0].removeprefix("__label__")


def lid_path() -> Path:
    """The same LID the app uses (the chapter bundle's lid.176.ftz), so 'language' means the same thing."""
    models = Path("~/projects/new_youtube_suite/backend/models").expanduser()
    for p in [HOME / "lid.176.bin", models / "chapters" / "current" / "lid.176.ftz",
              *sorted(models.glob("**/lid.176*"))]:
        if p.exists():
            return p
    sys.exit("fastText lid.176 not found: put it at $M16_HOME/lid.176.bin")


def clean_turns(turns) -> list[dict] | None:
    """Normalise roles; require optional system, then strictly alternating user/assistant ending on assistant."""
    out = []
    for t in turns:
        role = ROLE.get(t.get("role") or t.get("from"))
        content = t.get("content") if "content" in t else t.get("value")
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        out.append({"role": role, "content": content.strip()})
    body = out[1:] if out and out[0]["role"] == "system" else out
    if len(body) < 2 or body[-1]["role"] != "assistant":
        return None
    if any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(body)):
        return None
    return out


def rows(name: str):
    import pyarrow.parquet as pq

    for f in sorted((DL / name).glob("**/*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096):
            for r in batch.to_pylist():
                yield f, r


def convert(name: str, f: Path, r: dict) -> dict | None:
    if name == "aya":
        msgs = clean_turns([{"role": "user", "content": r["inputs"]},
                            {"role": "assistant", "content": r["targets"]}])
        return msgs and {"messages": msgs, "lang": AYA_LANG.get(r.get("language"), "other")}
    if name == "ultrafeedback":
        c, j = r["chosen"], r["rejected"]
        if not (c and j and c[-1]["role"] == "assistant" and j[-1]["role"] == "assistant"):
            return None
        prompt = clean_turns(c[:-1] + [{"role": "assistant", "content": "x"}])
        if prompt is None or c[:-1] != j[:-1] or c[-1]["content"].strip() == j[-1]["content"].strip():
            return None
        return {"messages": prompt[:-1], "chosen": c[-1]["content"].strip(), "rejected": j[-1]["content"].strip(),
                "lang": "en", "split": "val" if "test_prefs" in f.name else "train",
                "score_chosen": r.get("score_chosen"), "score_rejected": r.get("score_rejected")}
    turns = r.get("messages") or r.get("conversations")
    msgs = clean_turns(turns or [])
    if msgs is None:
        return None
    return {"messages": msgs, "lang": lid(" ".join(m["content"] for m in msgs if m["role"] != "system"))}


def cmd_build(a):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.names or list(SOURCES):
        out = OUT / f"{name}.jsonl"
        if out.exists():
            print(f"[{name}] already built", flush=True)
            continue
        if not any((DL / name).glob("**/*.parquet")):
            sys.exit(f"[{name}] nothing downloaded; run fetch first")
        keep = SOURCES[name]["keep_langs"]
        stats = {"rows": 0, "kept": 0, "bad_format": 0, "other_lang": 0}
        tmp = out.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as w:
            for f, r in rows(name):
                stats["rows"] += 1
                ex = convert(name, f, r)
                if not ex:
                    stats["bad_format"] += 1
                elif ex["lang"] not in keep:
                    stats["other_lang"] += 1
                else:
                    ex["source"] = name
                    w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
        tmp.replace(out)
        (OUT / f"{name}.stats.json").write_text(json.dumps(stats, indent=1))
        print(f"[{name}] {stats}", flush=True)
        if not a.keep_download:  # the owner prefers deleting intermediates as soon as they are consumed
            import shutil

            shutil.rmtree(DL / name)
            print(f"[{name}] deleted {DL / name}", flush=True)


def main(argv=None):
    random.seed(0)
    ap = argparse.ArgumentParser(prog="python -m llm.general_sft")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("names", nargs="*", default=[])
    b = sub.add_parser("build")
    b.add_argument("names", nargs="*", default=[])
    b.add_argument("--keep-download", action="store_true", help="don't delete the parquet files afterwards")
    a = ap.parse_args(argv)
    if unknown := set(a.names) - set(SOURCES):
        sys.exit(f"unknown sources {sorted(unknown)}; known: {list(SOURCES)}")
    {"fetch": cmd_fetch, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
