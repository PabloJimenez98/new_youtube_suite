"""WER of each variant against a reference transcript, per excerpt (M9 benchmark).

    asr_compare.py DIR REF_PREFIX VARIANT [VARIANT...]

DIR holds `<name>__<excerpt>.json` files from asr_variant.py.
Also writes `DIR/diffs_<variant>.txt`: the differing spans, for reading.
"""

import difflib
import json
import re
import sys
from pathlib import Path

import jiwer


def norm(words):
    return [x for x, _ in kept(words)]


def kept(words):
    """(normalized word, start) for every word that survives normalization."""
    out = []
    for w in words:
        x = re.sub(r"[^\w']+", "", w["word"].lower())
        if x:
            out.append((x, w.get("start")))
    return out


d, ref_name, *variants = sys.argv[1:]
d = Path(d)
excerpts = sorted(p.name.split("__")[1][:-5] for p in d.glob(f"{ref_name}__*.json"))
for v in variants:
    errs = words = 0
    secs = 0.0
    lines = []
    drift = []  # |start(hyp) - start(ref)| over aligned (equal) words
    for ex in excerpts:
        ref = json.loads((d / f"{ref_name}__{ex}.json").read_text())
        hyp = json.loads((d / f"{v}__{ex}.json").read_text())
        rw, hw = kept(ref["words"]), kept(hyp["words"])
        r, h = [x for x, _ in rw], [x for x, _ in hw]
        out = jiwer.process_words(" ".join(r), " ".join(h))
        errs += out.substitutions + out.deletions + out.insertions
        words += len(r)
        secs += hyp["seconds"]
        for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
            if tag == "equal":
                drift += [abs((hw[b1 + k][1] or 0) - (rw[a1 + k][1] or 0)) for k in range(a2 - a1)]
            else:
                ctx = " ".join(r[max(0, a1 - 4):a1])
                lines.append(f"[{ex}] ...{ctx} | REF: {' '.join(r[a1:a2])} | HYP: {' '.join(h[b1:b2])}")
    (d / f"diffs_{v}.txt").write_text("\n".join(lines))
    drift.sort()
    p95 = drift[int(0.95 * (len(drift) - 1))] if drift else float("nan")
    print(f"{v}: WER {errs / words:.3%} ({errs}/{words}), drift p95 {p95:.2f}s, {secs:.0f}s total")
