"""Pretraining data (0025 §5): download -> tokenizer sample -> tokenizer -> shards.

    python -m llm.prepare_data fetch [--source es|en]     # resumable; text column only
    python -m llm.prepare_data sample                     # ~2 GB 70/30 text sample for the tokenizer
    python -m llm.prepare_data train-tokenizer
    python -m llm.prepare_data tokenize [--source es|en]  # resumable, one shard per raw file

Layout under $M16_HOME (default ~/yts-experiments/m16-llm):
    data/raw/<source>/<file>.parquet   text-only copies (zstd); originals are deleted after extraction
    data/raw/<source>/manifest.json    per-file docs / UTF-8 bytes
    data/tok_sample.txt                tokenizer training sample
    tokenizer/tokenizer.json
    data/shards/<source>/...           see llm.data

Every step is resumable: finished files are skipped, partial outputs are `*.tmp`.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from llm import HOME

SOURCES = {
    # FineWeb2-HQ: the top-10 % quality subset of FineWeb-2 (ODC-By). Chosen over plain FineWeb-2 in
    # notebook 01 / 0025 §5. Files also carry an embedding column, which is dropped on extraction.
    "es": {"repo": "epfml/FineWeb2-HQ", "prefix": "spa_Latn/", "target_bytes": 95e9},
    # FineWeb-Edu sample-10BT (ODC-By): all 14 files, ~10B GPT-2 tokens.
    "en": {"repo": "HuggingFaceFW/fineweb-edu", "prefix": "sample/10BT/", "target_bytes": 60e9},
}
VAL_DOCS = 5_000  # first docs of each source's first raw file -> val.bin


def raw_dir(src: str) -> Path:
    return HOME / "data" / "raw" / src


def shard_dir(src: str) -> Path:
    return HOME / "data" / "shards" / src


def tokenizer_path() -> Path:
    return HOME / "tokenizer" / "tokenizer.json"


def _manifest(src: str) -> dict:
    p = raw_dir(src) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_manifest(src: str, m: dict):
    p = raw_dir(src) / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1))
    tmp.replace(p)


# ---------------------------------------------------------------------------- fetch

def _remote_files(src: str) -> list[str]:
    from huggingface_hub import HfApi

    s = SOURCES[src]
    files = HfApi().list_repo_tree(s["repo"], path_in_repo=s["prefix"].rstrip("/"), repo_type="dataset",
                                   recursive=True)
    return sorted(f.path for f in files if f.path.endswith(".parquet") and "/test/" not in f.path)


def _fetch_one(src: str, remote: str) -> tuple[str, int, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    s = SOURCES[src]
    out = raw_dir(src) / Path(remote).name
    dl_dir = HOME / "data" / "download" / src
    for attempt in range(5):
        try:
            local = Path(hf_hub_download(s["repo"], remote, repo_type="dataset", local_dir=dl_dir))
            break
        except Exception as e:  # network hiccups: back off and retry
            if attempt == 4:
                raise
            print(f"  {remote}: {type(e).__name__} {e}; retrying", flush=True)
            time.sleep(30 * (attempt + 1))
    pf = pq.ParquetFile(local)
    tmp = out.with_suffix(".parquet.tmp")
    docs = nbytes = 0
    with pq.ParquetWriter(tmp, pa.schema([("text", pa.string())]), compression="zstd") as w:
        for batch in pf.iter_batches(columns=["text"], batch_size=8192):
            col = batch.column(0)
            docs += len(col)
            nbytes += sum(len(t.encode("utf-8")) for t in col.to_pylist())
            w.write_batch(pa.record_batch([col], names=["text"]))
    tmp.replace(out)
    local.unlink()
    return out.name, docs, nbytes


def force_ipv6():
    """Resolve hosts to IPv6 only (falling back to everything if a host has no AAAA record).

    On 2026-11-14 sustained parallel downloads filled the router/ISP's IPv4 NAT table and new IPv4
    connections failed for the whole PC, while IPv6 (no NAT) kept working. Downloading over IPv6 keeps
    the IPv4 table free for normal browsing. Needs WSL's mirrored networking for IPv6 (C:/Users/Pablo/.wslconfig).
    """
    import socket

    orig = socket.getaddrinfo

    def v6_first(host, *args, **kw):
        res = orig(host, *args, **kw)
        v6 = [r for r in res if r[0] == socket.AF_INET6]
        return v6 or res

    socket.getaddrinfo = v6_first


def cmd_fetch(a):
    if a.ipv6:
        force_ipv6()
    for src in [a.source] if a.source else list(SOURCES):
        raw_dir(src).mkdir(parents=True, exist_ok=True)
        m = _manifest(src)
        have = sum(v["bytes"] for v in m.values())
        target = SOURCES[src]["target_bytes"]
        todo = [r for r in _remote_files(src) if Path(r).name not in m]
        print(f"[{src}] have {have / 1e9:.1f} / {target / 1e9:.0f} GB of text; {len(todo)} remote files left",
              flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            pending = set()
            it = iter(todo)
            while True:
                while have < target and len(pending) < a.workers:
                    r = next(it, None)
                    if r is None:
                        break
                    pending.add(ex.submit(_fetch_one, src, r))
                if not pending:
                    break
                done = next(as_completed(pending))
                pending.remove(done)
                name, docs, nbytes = done.result()
                m[name] = {"docs": docs, "bytes": nbytes}
                _save_manifest(src, m)
                have += nbytes
                print(f"[{src}] {name}: {docs:,} docs, {nbytes / 1e9:.2f} GB -> total {have / 1e9:.1f} GB",
                      flush=True)
        print(f"[{src}] fetch complete: {have / 1e9:.1f} GB of text in {len(m)} files", flush=True)


# ---------------------------------------------------------------------------- tokenizer sample

def _iter_texts(src: str, files: list[str] | None = None):
    import pyarrow.parquet as pq

    for name in files or sorted(_manifest(src)):
        for batch in pq.ParquetFile(raw_dir(src) / name).iter_batches(columns=["text"], batch_size=4096):
            yield from batch.column(0).to_pylist()


def cmd_sample(a):
    """~`--gb` of text, 70/30, skipping the val docs, drawn from several files of each source."""
    out = HOME / "data" / "tok_sample.txt"
    rng = random.Random(0)
    shares = {"es": 0.7, "en": 0.3}
    with open(out.with_suffix(".tmp"), "w", encoding="utf-8") as f:
        for src, share in shares.items():
            budget = a.gb * 1e9 * share
            files = sorted(_manifest(src))
            if not files:
                sys.exit(f"No raw files for {src}; run fetch first.")
            picked = files[: max(1, min(len(files), 6))]
            per_file = budget / len(picked)
            for name in picked:
                n = 0
                for i, t in enumerate(_iter_texts(src, [name])):
                    if name == files[0] and i < VAL_DOCS:
                        continue
                    if rng.random() < 0.5:
                        continue
                    f.write(t.replace("\x00", "") + "\x00")  # NUL-separated documents
                    n += len(t.encode("utf-8"))
                    if n >= per_file:
                        break
            print(f"sample: {src} done", flush=True)
    out.with_suffix(".tmp").replace(out)
    print(f"sample: {out} ({out.stat().st_size / 1e9:.2f} GB)")


def cmd_train_tokenizer(a):
    from llm.tokenizer import train_tokenizer

    sample = HOME / "data" / "tok_sample.txt"

    def docs():
        buf = ""
        with open(sample, encoding="utf-8") as f:
            while chunk := f.read(1 << 24):
                buf += chunk
                *done, buf = buf.split("\x00")
                yield from done
        if buf:
            yield buf

    t0 = time.time()
    tok = train_tokenizer(docs(), a.vocab_size, tokenizer_path())
    print(f"tokenizer: {tok.get_vocab_size()} tokens in {time.time() - t0:.0f}s -> {tokenizer_path()}")
    if not a.keep_sample:  # the owner prefers deleting intermediates as soon as they are consumed
        sample.unlink()
        print(f"deleted {sample}")


# ---------------------------------------------------------------------------- tokenize

def _tokenize_file(src: str, name: str, is_first: bool) -> dict:
    import pyarrow.parquet as pq

    from llm.data import DTYPE
    from llm.tokenizer import EOT, load_tokenizer

    tok = load_tokenizer(tokenizer_path())
    eot = tok.token_to_id(EOT)
    out_dir = shard_dir(src)
    stem = Path(name).stem
    res = {}
    targets = {"train": open(out_dir / f"train_{stem}.bin.tmp", "wb")}
    counts = {"train": [0, 0]}
    if is_first:
        targets["val"] = open(out_dir / "val.bin.tmp", "wb")
        counts["val"] = [0, 0]
    seen = 0
    for batch in pq.ParquetFile(raw_dir(src) / name).iter_batches(columns=["text"], batch_size=2048):
        texts = batch.column(0).to_pylist()
        encs = tok.encode_batch(texts, add_special_tokens=False)
        for t, e in zip(texts, encs):
            split = "val" if is_first and seen < VAL_DOCS else "train"
            seen += 1
            ids = np.asarray(e.ids + [eot], dtype=DTYPE)
            targets[split].write(ids.tobytes())
            counts[split][0] += len(ids)
            counts[split][1] += len(t.encode("utf-8"))
    for split, f in targets.items():
        f.close()
        final = out_dir / (f"train_{stem}.bin" if split == "train" else "val.bin")
        Path(f.name).replace(final)
        res[split] = {"file": final.name, "tokens": counts[split][0], "bytes": counts[split][1]}
    return res


def cmd_tokenize(a):
    from llm.data import write_index
    from llm.tokenizer import load_tokenizer

    vocab = load_tokenizer(tokenizer_path()).get_vocab_size()
    assert vocab <= 65536, "uint16 shards"
    for src in [a.source] if a.source else list(SOURCES):
        out = shard_dir(src)
        out.mkdir(parents=True, exist_ok=True)
        files = sorted(_manifest(src))
        missing = [n for n in files if not (raw_dir(src) / n).exists()]
        state_now = json.loads((out / "tokenize_state.json").read_text()) if (out / "tokenize_state.json").exists() else {}
        if [n for n in missing if n not in state_now]:
            sys.exit(f"[{src}] raw files missing and not tokenized: {missing[:3]}... (re-run fetch)")
        state_p = out / "tokenize_state.json"
        state = json.loads(state_p.read_text()) if state_p.exists() else {}
        todo = [n for n in files if n not in state]
        print(f"[{src}] tokenizing {len(todo)} of {len(files)} files", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(_tokenize_file, src, n, n == files[0]): n for n in todo}
            for fut in as_completed(futs):
                n = futs[fut]
                state[n] = fut.result()
                tmp = state_p.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=1))
                tmp.replace(state_p)
                if not a.keep_raw:  # recorded in tokenize_state.json: the raw copy is no longer needed
                    (raw_dir(src) / n).unlink(missing_ok=True)
                tr = state[n]["train"]
                print(f"[{src}] {n}: {tr['tokens'] / 1e6:,.0f}M tokens "
                      f"({tr['bytes'] / tr['tokens']:.2f} bytes/token)", flush=True)
        train = [state[n]["train"] for n in files if n in state]
        val = state[files[0]].get("val") if files and files[0] in state else None
        write_index(out, vocab, train, val, {"tokenizer": str(tokenizer_path())})
        total = sum(e["tokens"] for e in train)
        print(f"[{src}] index: {total / 1e9:.2f}B train tokens in {len(train)} shards; "
              f"val {val['tokens'] if val else 0:,} tokens", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m llm.prepare_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--source", choices=list(SOURCES))
    f.add_argument("--workers", type=int, default=2)
    f.add_argument("--ipv4", dest="ipv6", action="store_false",
                   help="allow IPv4 (default: IPv6 only, see force_ipv6)")
    f.set_defaults(fn=cmd_fetch, ipv6=True)
    s = sub.add_parser("sample")
    s.add_argument("--gb", type=float, default=2.0)
    s.set_defaults(fn=cmd_sample)
    t = sub.add_parser("train-tokenizer")
    t.add_argument("--vocab-size", type=int, default=32768)
    t.add_argument("--keep-sample", action="store_true", help="don't delete the sample afterwards")
    t.set_defaults(fn=cmd_train_tokenizer)
    k = sub.add_parser("tokenize")
    k.add_argument("--source", choices=list(SOURCES))
    k.add_argument("--keep-raw", action="store_true", help="don't delete each raw file once tokenized")
    k.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    k.set_defaults(fn=cmd_tokenize)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
