"""Unit embeddings for many videos in one memory-mapped float16 array.

~10k videos x ~500 units x 384 dims is several GB -- too much to keep as
a dict of arrays in RAM next to training. Layout on disk (second drive):

    <dir>/emb.f16      concatenated [n_units_total, dim] float16
    <dir>/index.json   {"dim": d, "videos": [{"vid", "offset", "n", "unit_starts", "duration", "labels"}...]}
"""

import json
from pathlib import Path

import numpy as np


class StoreWriter:
    def __init__(self, directory: Path, dim: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.f = open(self.dir / "emb.f16", "wb")
        self.videos: list[dict] = []
        self.offset = 0

    def add(self, vid: str, emb: np.ndarray, meta: dict) -> None:
        emb = np.ascontiguousarray(emb, dtype=np.float16)
        assert emb.ndim == 2 and emb.shape[1] == self.dim
        self.f.write(emb.tobytes())
        self.videos.append({"vid": vid, "offset": self.offset, "n": len(emb), **meta})
        self.offset += len(emb)

    def close(self) -> None:
        self.f.close()
        (self.dir / "index.json").write_text(json.dumps({"dim": self.dim, "videos": self.videos}))


class Store:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        idx = json.loads((self.dir / "index.json").read_text())
        self.dim = idx["dim"]
        self.videos = idx["videos"]
        total = sum(v["n"] for v in self.videos)
        self.emb = np.memmap(self.dir / "emb.f16", dtype=np.float16, mode="r", shape=(total, self.dim))

    def __len__(self) -> int:
        return len(self.videos)

    def get(self, i: int) -> tuple[np.ndarray, dict]:
        v = self.videos[i]
        return np.asarray(self.emb[v["offset"] : v["offset"] + v["n"]], dtype=np.float32), v
