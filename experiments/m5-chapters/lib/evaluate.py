"""One call to get every boundary metric for a set of videos (0009 §6)."""

from dataclasses import dataclass

import numpy as np

from lib import metrics as m
from lib.vidchapters import boundary_units


@dataclass
class Prediction:
    vid: str
    duration: float
    unit_starts: list[float]
    ref_starts: list[float]
    pred_starts: list[float]


def evaluate(preds: list[Prediction]) -> dict[str, float]:
    refs = {p.vid: p.ref_starts for p in preds}
    hyp = {p.vid: p.pred_starts for p in preds}
    durs = {p.vid: p.duration for p in preds}
    out = {}
    for dist in (3, 5, 10, 30):
        r = m.f1_at_distance(refs, hyp, dist)
        out[f"F1@{dist}s"] = r["f1"]
        if dist == 5:
            out["P@5s"], out["R@5s"] = r["precision"], r["recall"]
    out["tIoU-F1"] = m.tiou_f1(refs, hyp, durs)
    out["mean_tIoU"] = m.mean_tiou(refs, hyp, durs)
    pks, wds = [], []
    for p in preds:
        n = len(p.unit_starts)
        rb = boundary_units(p.unit_starts, p.ref_starts)
        hb = boundary_units(p.unit_starts, p.pred_starts)
        pks.append(m.pk(rb, hb, n))
        wds.append(m.windowdiff(rb, hb, n))
    out["Pk"], out["WindowDiff"] = float(np.mean(pks)), float(np.mean(wds))
    ce = m.count_error(refs, hyp)
    out["count_MAE"] = ce["mae"]
    out["pred_chapters_mean"] = float(np.mean([len(p.pred_starts) for p in preds]))
    out["ref_chapters_mean"] = float(np.mean([len(m.starts_to_segments(p.ref_starts, p.duration)) for p in preds]))
    return {k: round(v, 3) for k, v in out.items()}
