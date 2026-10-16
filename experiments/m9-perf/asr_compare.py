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
    return [x for x in (re.sub(r"[^\w']+", "", w["word"].lower()) for w in words) if x]


d, ref_name, *variants = sys.argv[1:]
d = Path(d)
excerpts = sorted(p.name.split("__")[1][:-5] for p in d.glob(f"{ref_name}__*.json"))
for v in variants:
    errs = words = 0
    secs = 0.0
    lines = []
    for ex in excerpts:
        ref = json.loads((d / f"{ref_name}__{ex}.json").read_text())
        hyp = json.loads((d / f"{v}__{ex}.json").read_text())
        r, h = norm(ref["words"]), norm(hyp["words"])
        out = jiwer.process_words(" ".join(r), " ".join(h))
        errs += out.substitutions + out.deletions + out.insertions
        words += len(r)
        secs += hyp["seconds"]
        for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
            if tag != "equal":
                ctx = " ".join(r[max(0, a1 - 4):a1])
                lines.append(f"[{ex}] ...{ctx} | REF: {' '.join(r[a1:a2])} | HYP: {' '.join(h[b1:b2])}")
    (d / f"diffs_{v}.txt").write_text("\n".join(lines))
    print(f"{v}: WER {errs / words:.3%} ({errs}/{words}), {secs:.0f}s total")
