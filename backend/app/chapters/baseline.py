# Copied from experiments/m5-chapters/lib/baseline.py -- keep in sync (the
# experiment's tests cover the same logic; docs/decisions/0009 §2-4).
"""Unsupervised boundary baseline: TextTiling over unit embeddings.

Hearst's TextTiling with the lexical block similarity replaced by the
cosine similarity of mean unit embeddings (the "embedding TextTiling"
recipe in docs/research/chapter-segmentation-prior-art.md §2). No
training. Output is one score per unit in [0, 1], fed to the same
`decoding.decode` as the trained model (docs/decisions/0009 §4).

For the gap before unit i: similarity between the mean of the `window`
units before it and the `window` units from it on; the depth score is
how far that similarity dips below the nearest peaks on each side.
"""

import numpy as np


def gap_similarities(emb: np.ndarray, window: int) -> np.ndarray:
    """Cosine similarity across each gap; sim[i] is the gap before unit i (sim[0] = 1)."""
    n = len(emb)
    unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    csum = np.vstack([np.zeros((1, unit.shape[1])), np.cumsum(unit, axis=0)])
    sims = np.ones(n)
    for i in range(1, n):
        lo, hi = max(0, i - window), min(n, i + window)
        left = (csum[i] - csum[lo]) / (i - lo)
        right = (csum[hi] - csum[i]) / (hi - i)
        sims[i] = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
    return sims


def depth_scores(sims: np.ndarray) -> np.ndarray:
    """TextTiling depth: (left peak - s) + (right peak - s), climbing while similarity rises."""
    n = len(sims)
    depth = np.zeros(n)
    for i in range(1, n):
        left = sims[i]
        j = i - 1
        while j >= 1 and sims[j] >= left:
            left = sims[j]
            j -= 1
        right = sims[i]
        j = i + 1
        while j < n and sims[j] >= right:
            right = sims[j]
            j += 1
        depth[i] = (left - sims[i]) + (right - sims[i])
    return depth


def boundary_scores(emb: np.ndarray, window: int = 12) -> np.ndarray:
    """Per-unit boundary score in [0, 1] (unit 0 always scores 0)."""
    if len(emb) < 2:
        return np.zeros(len(emb))
    depth = depth_scores(gap_similarities(np.asarray(emb, dtype=np.float64), window))
    top = depth.max()
    return depth / top if top > 0 else depth
