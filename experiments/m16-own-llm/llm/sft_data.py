"""SFT data (0025 §9): chat examples -> packed token rows + an assistant-only loss mask.

Input: JSONL files, one example per line: {"messages": [{"role", "content"}, ...], ...}.
Every assistant turn is a target. Examples are rendered with our ChatML template
(llm.tokenizer.render_chat), exactly as the app prompts the model at inference.

Output directory (read by llm.data.SFTLoader):
    tokens.bin / mask.bin            train rows: uint16 ids, uint8 1 = token is a target
    val_tokens.bin / val_mask.bin    validation rows
    index.json                       {"seq_len", "rows", "val_rows", "examples", "dropped_too_long", ...}

Packing: examples are shuffled (fixed seed) and packed greedily into rows of `seq_len + 1`
tokens, padded with <|pad|> (mask 0). Attention is not masked between examples that
share a row; each example starts after the previous one's <|im_end|>, which the model
learns to treat as a boundary (the same packing choice as pretraining, 0025 §5).

    python -m llm.sft_data build --out DIR --seq-len 4096 [--val-fraction 0.02] FILE.jsonl[:WEIGHT] ...

`:WEIGHT` repeats (or subsamples, if < 1) a file's examples, to set the mix.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from llm.tokenizer import IM_END, IM_START, PAD, load_tokenizer


def chat_text(messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """The rendered conversation (identical to render_chat) and the char spans of assistant targets."""
    text, spans = "", []
    for m in messages:
        text += f"{IM_START}{m['role']}\n"
        start = len(text)
        text += f"{m['content']}{IM_END}"
        if m["role"] == "assistant":
            spans.append((start, len(text)))
        text += "\n"
    return text, spans


def encode_example(tok, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids of the whole rendered conversation -- tokenized in one piece, exactly as the app
    tokenizes its prompts -- and a 0/1 mask: 1 on tokens lying inside an assistant turn's content
    or its <|im_end|>."""
    text, spans = chat_text(messages)
    enc = tok.encode(text)
    mask = [int(b > a and any(s <= a and b <= e for s, e in spans)) for a, b in enc.offsets]
    return enc.ids, mask


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pack(examples: list[tuple[list[int], list[int]]], row_len: int, pad_id: int):
    rows_ids, rows_mask = [], []
    cur_i, cur_m = [], []
    for ids, mask in examples:
        if len(cur_i) + len(ids) > row_len:
            rows_ids.append(cur_i + [pad_id] * (row_len - len(cur_i)))
            rows_mask.append(cur_m + [0] * (row_len - len(cur_m)))
            cur_i, cur_m = [], []
        cur_i += ids
        cur_m += mask
    if cur_i:
        rows_ids.append(cur_i + [pad_id] * (row_len - len(cur_i)))
        rows_mask.append(cur_m + [0] * (row_len - len(cur_m)))
    return np.array(rows_ids, dtype=np.uint16), np.array(rows_mask, dtype=np.uint8)


def build(sources: list[tuple[Path, float]], tokenizer_path: str | Path, out: Path, seq_len: int,
          val_fraction: float = 0.02, seed: int = 0) -> dict:
    tok = load_tokenizer(tokenizer_path)
    rng = random.Random(seed)
    row_len = seq_len + 1  # inputs and shifted targets come from the same row
    train, val, dropped, per_source = [], [], 0, {}
    for path, weight in sources:
        exs = list(read_jsonl(path))
        reps = int(weight) + (1 if rng.random() < weight - int(weight) else 0) if weight >= 1 else 1
        if weight < 1:
            exs = rng.sample(exs, int(len(exs) * weight))
        n = 0
        for ex in exs:
            ids, mask = encode_example(tok, ex["messages"])
            if len(ids) > row_len:
                dropped += 1
                continue
            if not any(mask):
                continue
            to_val = ex.get("split") == "val" or (ex.get("split") is None and rng.random() < val_fraction)
            (val if to_val else train).extend([(ids, mask)] * (1 if to_val else reps))
            n += 1
        per_source[Path(path).name] = {"examples": n, "repeats": reps if weight >= 1 else weight}
    rng.shuffle(train)
    pad = tok.token_to_id(PAD)
    out.mkdir(parents=True, exist_ok=True)
    tr_i, tr_m = pack(train, row_len, pad)
    va_i, va_m = pack(val, row_len, pad)
    tr_i.tofile(out / "tokens.bin")
    tr_m.tofile(out / "mask.bin")
    va_i.tofile(out / "val_tokens.bin")
    va_m.tofile(out / "val_mask.bin")
    idx = {"seq_len": seq_len, "rows": len(tr_i), "val_rows": len(va_i), "examples": len(train),
           "val_examples": len(val), "dropped_too_long": dropped, "target_tokens": int(tr_m.sum()),
           "fill": float((tr_i != pad).mean()) if len(tr_i) else 0.0, "sources": per_source,
           "tokenizer": str(tokenizer_path)}
    (out / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


def main(argv=None):
    from llm import prepare_data

    ap = argparse.ArgumentParser(prog="python -m llm.sft_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--seq-len", type=int, default=4096)
    b.add_argument("--val-fraction", type=float, default=0.02)
    b.add_argument("--tokenizer", default=str(prepare_data.tokenizer_path()))
    b.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    srcs = []
    for f in a.files:
        path, _, w = f.partition(":")
        srcs.append((Path(path).expanduser(), float(w) if w else 1.0))
    print(json.dumps(build(srcs, a.tokenizer, Path(a.out).expanduser(), a.seq_len, a.val_fraction), indent=1))


if __name__ == "__main__":
    main()
