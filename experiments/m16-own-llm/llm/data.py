"""Token shards and the deterministic, resumable mixed loader (0025 §5, §7).

On disk, one directory per source (`es`, `en`, ...):

    <root>/<source>/index.json   {"vocab_size", "train": [{"file", "tokens", "bytes"}], "val": {...}}
    <root>/<source>/train_0000.bin ...   raw little-endian uint16 token ids
    <root>/<source>/val.bin

Documents are concatenated, each followed by `<|endoftext|>` (packing, 0025 §5).
`bytes` is the UTF-8 size of the text behind a shard, so validation loss can be
reported as bits per byte (tokenizer-independent, 0025 §3).

`MixedLoader` gives row k of the global row sequence to the source chosen by a
fixed, smoothly interleaved pattern (70/30 -> es es en es es en es ...). Its whole
state is the row counter plus one (shard, offset) cursor per source, so a resume
continues with exactly the rows an uninterrupted run would have seen.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

DTYPE = np.uint16


class ShardWriter:
    """Appends documents' token ids to `<dir>/<prefix>_NNNN.bin`, rolling every `shard_tokens`."""

    def __init__(self, directory: Path, prefix: str = "train", shard_tokens: int = 100_000_000):
        self.dir, self.prefix, self.shard_tokens = Path(directory), prefix, shard_tokens
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        self._f = None
        self._open_new()

    def _name(self, i: int) -> str:
        return f"{self.prefix}_{i:04d}.bin" if self.prefix != "val" else "val.bin"

    def _open_new(self):
        if self._f:
            self._f.close()
        self.entries.append({"file": self._name(len(self.entries)), "tokens": 0, "bytes": 0})
        self._f = open(self.dir / self.entries[-1]["file"], "wb")

    def write(self, ids, nbytes: int):
        arr = np.asarray(ids, dtype=DTYPE)
        e = self.entries[-1]
        if e["tokens"] and e["tokens"] + len(arr) > self.shard_tokens and self.prefix != "val":
            self._open_new()
            e = self.entries[-1]
        self._f.write(arr.tobytes())
        e["tokens"] += len(arr)
        e["bytes"] += nbytes

    @property
    def tokens(self) -> int:
        return sum(e["tokens"] for e in self.entries)

    def close(self) -> list[dict]:
        if self._f:
            self._f.close()
            self._f = None
        if self.entries and self.entries[-1]["tokens"] == 0:
            (self.dir / self.entries[-1]["file"]).unlink(missing_ok=True)
            self.entries.pop()
        return self.entries


def write_index(directory: Path, vocab_size: int, train: list[dict], val: dict | None, extra: dict | None = None):
    idx = {"vocab_size": vocab_size, "train": train, "val": val, **(extra or {})}
    tmp = Path(directory) / "index.json.tmp"
    tmp.write_text(json.dumps(idx, indent=1))
    tmp.replace(Path(directory) / "index.json")


def read_index(directory: Path) -> dict:
    return json.loads((Path(directory) / "index.json").read_text())


class SourceStream:
    """Sequential reader over one source's train shards; wraps around (counting epochs)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        idx = read_index(self.dir)
        self.shards = [e for e in idx["train"] if e["tokens"] > 0]
        if not self.shards:
            raise ValueError(f"{self.dir}: no training tokens")
        self._mm: dict[int, np.memmap] = {}
        self.shard, self.offset, self.epoch = 0, 0, 0

    def _arr(self, i: int) -> np.memmap:
        if i not in self._mm:
            self._mm[i] = np.memmap(self.dir / self.shards[i]["file"], dtype=DTYPE, mode="r")
        return self._mm[i]

    def take(self, n: int) -> np.ndarray:
        out, need = [], n
        while need:
            arr = self._arr(self.shard)
            chunk = arr[self.offset : self.offset + need]
            out.append(np.asarray(chunk))
            need -= len(chunk)
            self.offset += len(chunk)
            if self.offset >= len(arr):
                self.shard, self.offset = self.shard + 1, 0
                if self.shard == len(self.shards):
                    self.shard, self.epoch = 0, self.epoch + 1
        return np.concatenate(out)

    def state(self) -> dict:
        return {"shard": self.shard, "offset": self.offset, "epoch": self.epoch}

    def load_state(self, s: dict):
        self.shard, self.offset, self.epoch = s["shard"], s["offset"], s["epoch"]


def mix_pattern(weights: dict[str, float], length: int = 100) -> list[str]:
    """Smooth weighted round-robin over `length` slots: the shares are exact over a period."""
    names = sorted(weights)
    total = sum(weights.values())
    quota = {n: Fraction(weights[n] / total).limit_denominator(length) for n in names}
    period = 1
    for q in quota.values():
        period = period * q.denominator // np.gcd(period, q.denominator)
    counts = {n: int(quota[n] * period) for n in names}
    current = {n: 0 for n in names}
    out = []
    for _ in range(period):
        for n in names:
            current[n] += counts[n]
        best = max(names, key=lambda n: (current[n], -names.index(n)))
        current[best] -= period
        out.append(best)
    return out


class MixedLoader:
    def __init__(self, root: str | Path, sources: dict[str, float]):
        root = Path(root).expanduser()
        self.streams = {n: SourceStream(root / n) for n in sources}
        self.pattern = mix_pattern(sources)
        self.row = 0

    def next_batch(self, batch: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        for _ in range(batch):
            src = self.pattern[self.row % len(self.pattern)]
            rows.append(self.streams[src].take(seq_len + 1).astype(np.int64))
            self.row += 1
        t = torch.from_numpy(np.stack(rows))
        return t[:, :-1], t[:, 1:]

    def state_dict(self) -> dict:
        return {"row": self.row, "streams": {n: s.state() for n, s in self.streams.items()}}

    def load_state_dict(self, d: dict):
        self.row = d["row"]
        for n, s in d["streams"].items():
            self.streams[n].load_state(s)

    def epochs(self) -> dict[str, int]:
        return {n: s.epoch for n, s in self.streams.items()}


class SFTLoader:
    """Packed SFT rows (llm.sft_data): (x, y, loss_mask) batches in a fixed order, cycling over epochs.

    The state is one row counter, so pause/resume is exact as for pretraining."""

    def __init__(self, directory: str | Path):
        d = Path(directory).expanduser()
        self.index = json.loads((d / "index.json").read_text())
        self.seq_len = self.index["seq_len"]
        row_len = self.seq_len + 1
        self.tokens = np.memmap(d / "tokens.bin", dtype=DTYPE, mode="r").reshape(-1, row_len)
        self.mask = np.memmap(d / "mask.bin", dtype=np.uint8, mode="r").reshape(-1, row_len)
        self.dir = d
        self.row = 0

    def next_batch(self, batch: int, seq_len: int):
        if seq_len != self.seq_len:
            raise ValueError(f"config seq_len {seq_len} != SFT data seq_len {self.seq_len}")
        n = len(self.tokens)
        idx = [(self.row + i) % n for i in range(batch)]
        self.row += batch
        t = torch.from_numpy(self.tokens[idx].astype(np.int64))
        m = torch.from_numpy(self.mask[idx].astype(np.float32))
        return t[:, :-1], t[:, 1:], m[:, 1:]

    def state_dict(self) -> dict:
        return {"row": self.row}

    def load_state_dict(self, d: dict):
        self.row = d["row"]

    def epochs(self) -> dict[str, float]:
        return {"sft": round(self.row / len(self.tokens), 3)}

    def val(self, n_rows: int):
        row_len = self.seq_len + 1
        if (self.dir / "val_tokens.bin").stat().st_size == 0:
            return torch.empty(0, row_len, dtype=torch.long), torch.empty(0, row_len)
        t = np.memmap(self.dir / "val_tokens.bin", dtype=DTYPE, mode="r").reshape(-1, row_len)[:n_rows]
        m = np.memmap(self.dir / "val_mask.bin", dtype=np.uint8, mode="r").reshape(-1, row_len)[:n_rows]
        return torch.from_numpy(t.astype(np.int64)), torch.from_numpy(m.astype(np.float32))


def val_rows(directory: Path, seq_len: int, n_rows: int) -> tuple[torch.Tensor, float] | None:
    """The first `n_rows` validation rows of a source, and its bytes-per-token ratio."""
    idx = read_index(directory)
    v = idx.get("val")
    if not v or v["tokens"] < seq_len + 1:
        return None
    arr = np.memmap(Path(directory) / v["file"], dtype=DTYPE, mode="r")
    n_rows = min(n_rows, (len(arr) - 1) // seq_len)
    rows = np.stack([arr[i * seq_len : i * seq_len + seq_len + 1] for i in range(n_rows)]).astype(np.int64)
    return torch.from_numpy(rows), v["bytes"] / v["tokens"]
