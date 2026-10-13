"""Encode a split's units once and persist them as a `Store` (0009 §6)."""

from pathlib import Path

from lib import vidchapters as vc
from lib.encode import Encoder
from lib.store import StoreWriter
from lib.units import UnitConfig


def build_store(videos: list[vc.Video], encoder: Encoder, unit_cfg: UnitConfig, out: Path,
                chunk: int = 64, log_every: int = 500) -> int:
    """Units -> embeddings -> store. Videos are encoded in chunks so the
    encoder's length sorting has a large pool. Returns units written."""
    w = StoreWriter(out, encoder.dim)
    total = 0
    for i in range(0, len(videos), chunk):
        batch = [(v, vc.units_for(v, unit_cfg)) for v in videos[i : i + chunk]]
        batch = [(v, us) for v, us in batch if len(us) >= 2]
        texts = [u.text for _, us in batch for u in us]
        emb = encoder(texts) if texts else None
        k = 0
        for v, us in batch:
            starts = [u.start for u in us]
            w.add(v.vid, emb[k : k + len(us)], {
                "unit_starts": starts, "duration": v.duration, "lang": v.lang,
                "chapter_starts": v.chapter_starts, "chapter_titles": v.chapter_titles,
                "labels": vc.boundary_units(starts, v.chapter_starts)})
            k += len(us)
            total += len(us)
        if (i // chunk) % max(1, log_every // chunk) == 0:
            print(f"  {min(i + chunk, len(videos))}/{len(videos)} videos, {total} units", flush=True)
    w.close()
    return total
