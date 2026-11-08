"""Pure analyses over a `MarketFrame` (docs/decisions/0022 §2-3).

Everything here is numpy/scipy, deterministic (seeded) and testable on
synthetic frames. The shared building block is the *age-adjusted view
residual*: popularity differs by orders of magnitude between channels
and grows with a video's age, so comparing raw views says little. Each
video's log-views is compared with what its own channel gets at that age.
"""

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.cluster.vq import kmeans2

from app.insights.frame import MarketFrame
from app.relevance.heuristics import STOPWORDS

SHORT_MAX_S = 180  # YouTube Shorts can run to 3 min
MIN_CHANNEL_VIDEOS = 20  # below this, a channel's own fit is too noisy: use the pooled one
BOOTSTRAP_B = 300
BOOTSTRAP_MAX_N = 6000
MIN_ANALYSIS_VIDEOS = 30
NOT_ENOUGH = "Not enough videos in the market data yet."
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _seeded() -> np.random.Generator:
    return np.random.default_rng(0)


# ---- Selection --------------------------------------------------------------


def select_format(frame: MarketFrame, fmt: str) -> np.ndarray:
    """Boolean mask: `long` (> 3 min), `short` (≤ 3 min) or `all`, with views known."""
    known = ~np.isnan(frame.views)
    if fmt == "all":
        return known
    dur = frame.duration_s
    is_short = ~np.isnan(dur) & (dur <= SHORT_MAX_S)
    return known & (is_short if fmt == "short" else ~is_short)


def age_days(frame: MarketFrame) -> np.ndarray:
    """Days between publication and the snapshot, at least 1."""
    published_day = frame.published.astype("datetime64[D]")
    return np.maximum((frame.snapshot_dates - published_day).astype(int), 1).astype(float)


# ---- Age-adjusted residuals --------------------------------------------------


@dataclass
class AgeFit:
    intercept: float
    slope: float
    r2: float
    n: int


def fit_log_age(ages: np.ndarray, views: np.ndarray) -> AgeFit:
    """Least squares log10(views+1) = a + b·log10(age)."""
    x = np.log10(ages)
    y = np.log10(views + 1)
    if len(x) < 3 or np.ptp(x) == 0:
        return AgeFit(float(np.mean(y)) if len(y) else 0.0, 0.0, 0.0, len(x))
    res = stats.linregress(x, y)
    return AgeFit(float(res.intercept), float(res.slope), float(res.rvalue**2), len(x))


def residuals(frame: MarketFrame, mask: np.ndarray) -> np.ndarray:
    """Age-adjusted log10 residual per selected video (NaN elsewhere).

    Each channel with ≥ MIN_CHANNEL_VIDEOS videos gets its own fit; the
    rest share the pooled fit. 0 = as expected for its channel and age;
    +0.3 ≈ 2× the expected views.
    """
    out = np.full(len(frame), np.nan)
    ages = age_days(frame)
    pooled = fit_log_age(ages[mask], frame.views[mask])
    for cid in np.unique(frame.channel_ids[mask]):
        m = mask & (frame.channel_ids == cid)
        fit = fit_log_age(ages[m], frame.views[m]) if m.sum() >= MIN_CHANNEL_VIDEOS else pooled
        out[m] = np.log10(frame.views[m] + 1) - (fit.intercept + fit.slope * np.log10(ages[m]))
    return out


# ---- Overview ----------------------------------------------------------------


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def channel_overview(frame: MarketFrame, mask: np.ndarray) -> list[dict]:
    """Per channel: views distribution (box-plot quantiles), engagement, cadence, duration."""
    out = []
    last_day = frame.snapshot_dates.max() if len(frame) else None
    for cid in sorted(set(frame.channel_ids[mask].tolist())):
        m = mask & (frame.channel_ids == cid)
        v = frame.views[m]
        q = np.percentile(v, [10, 25, 50, 75, 90])
        recent = (last_day - frame.published[m].astype("datetime64[D]")).astype(int) <= 90
        out.append(
            {
                "channel_id": int(cid),
                "channel": frame.channel_titles.get(int(cid), str(cid)),
                "own": int(cid) == frame.own_channel_id,
                "videos": int(m.sum()),
                "views_p10": float(q[0]),
                "views_q1": float(q[1]),
                "views_median": float(q[2]),
                "views_q3": float(q[3]),
                "views_p90": float(q[4]),
                "like_rate": _nanmedian(_ratio(frame.likes[m], v)),
                "comment_rate": _nanmedian(_ratio(frame.comments[m], v)),
                "uploads_per_week_90d": float(recent.sum()) / (90 / 7),
                "median_duration_min": _nanmedian(frame.duration_s[m]) / 60 if np.any(~np.isnan(frame.duration_s[m])) else None,
            }
        )
    return out


def _nanmedian(a: np.ndarray) -> float | None:
    a = a[~np.isnan(a)]
    return float(np.median(a)) if len(a) else None


def monthly_output(frame: MarketFrame, mask: np.ndarray, months: int = 24) -> list[dict]:
    """Videos published per month across tracked channels, with their median views."""
    month = frame.published[mask].astype("datetime64[M]")
    views = frame.views[mask]
    if not len(month):
        return []
    last = month.max()
    out = []
    for i in range(months - 1, -1, -1):
        mth = last - np.timedelta64(i, "M")
        sel = month == mth
        out.append(
            {
                "month": str(mth),
                "videos": int(sel.sum()),
                "median_views": float(np.median(views[sel])) if sel.any() else None,
            }
        )
    return out


# ---- Clustering (k-means + PCA) ----------------------------------------------

CLUSTER_FEATURES = [
    ("log_median_views", "Median views (log)"),
    ("like_rate", "Likes per view"),
    ("comment_rate", "Comments per view"),
    ("uploads_per_week_90d", "Uploads per week"),
    ("median_duration_min", "Median length (min)"),
    ("views_spread", "Views spread (IQR, log)"),
]


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() <= 1:
            scores.append(0.0)
            continue
        a = d[i, same].sum() / (same.sum() - 1)
        b = min(d[i, labels == k].mean() for k in set(labels.tolist()) if k != labels[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def cluster_channels(overview: list[dict], k_range: range = range(2, 4)) -> dict:
    """Standardise channel features, pick k by silhouette, k-means, then PCA to 2D."""
    rows = [r for r in overview if r["median_duration_min"] is not None and r["like_rate"] is not None]
    if len(rows) < 4:
        return {"available": False, "reason": "Needs at least 4 channels with complete data."}
    feats = np.array(
        [
            [
                np.log10(r["views_median"] + 1),
                r["like_rate"],
                r["comment_rate"] or 0.0,
                r["uploads_per_week_90d"],
                r["median_duration_min"],
                np.log10(r["views_q3"] + 1) - np.log10(r["views_q1"] + 1),
            ]
            for r in rows
        ]
    )
    sd = feats.std(axis=0)
    z = (feats - feats.mean(axis=0)) / np.where(sd > 0, sd, 1)

    best = None
    for k in k_range:
        if k >= len(rows):
            break
        _, labels = kmeans2(z, k, seed=0, minit="++")
        if len(set(labels.tolist())) < k:
            continue
        score = _silhouette(z, labels)
        if best is None or score > best[0]:
            best = (score, k, labels)
    if best is None:
        return {"available": False, "reason": "No stable clustering found."}
    score, k, labels = best

    u, s, vt = np.linalg.svd(z, full_matrices=False)
    coords = z @ vt[:2].T
    explained = (s**2) / (s**2).sum()
    clusters = []
    for c in range(k):
        sel = labels == c
        clusters.append(
            {
                "id": int(c),
                "size": int(sel.sum()),
                "profile": {name: float(np.median(feats[sel, j])) for j, (name, _) in enumerate(CLUSTER_FEATURES)},
            }
        )
    return {
        "available": True,
        "k": int(k),
        "silhouette": score,
        "explained_variance": [float(explained[0]), float(explained[1])],
        "features": [{"key": key, "label": label} for key, label in CLUSTER_FEATURES],
        "loadings": [
            {"key": key, "pc1": float(vt[0, j]), "pc2": float(vt[1, j])} for j, (key, _) in enumerate(CLUSTER_FEATURES)
        ],
        "points": [
            {"channel": r["channel"], "own": r["own"], "x": float(coords[i, 0]), "y": float(coords[i, 1]), "cluster": int(labels[i])}
            for i, r in enumerate(rows)
        ],
        "clusters": clusters,
    }


# ---- What drives views (Spearman + bootstrap CI, day×hour heatmap) -----------


def _title_features(titles: list[str]) -> dict[str, np.ndarray]:
    return {
        "title_chars": np.array([len(t) for t in titles], dtype=float),
        "title_question": np.array([("?" in t or "¿" in t) for t in titles], dtype=float),
        "title_number": np.array([bool(re.search(r"\d", t)) for t in titles], dtype=float),
        "title_caps": np.array([_caps_ratio(t) for t in titles], dtype=float),
        "title_separator": np.array([bool(re.search(r"[|:–-]", t)) for t in titles], dtype=float),
    }


def _caps_ratio(title: str) -> float:
    letters = [c for c in title if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


DRIVER_LABELS = {
    "duration_min": "Video length",
    "title_chars": "Title length",
    "title_question": "Title is a question",
    "title_number": "Title has a number",
    "title_caps": "Share of capitals in title",
    "title_separator": "Title has a separator ( | : – )",
    "tag_count": "Number of tags",
    "weekend": "Published on a weekend",
}


def spearman_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Spearman ρ with a 95% CI of ρ ± 1.96 · bootstrap SE.

    A percentile interval is biased here: resampling with replacement adds
    ties, which pulls ρ of binary features off-centre. The bootstrap SE is
    estimated on a subsample of BOOTSTRAP_MAX_N rows and rescaled to n.
    """
    rho = float(stats.spearmanr(x, y).statistic)
    n = len(x)
    m = min(n, BOOTSTRAP_MAX_N)
    sub = rng.choice(n, m, replace=False) if n > m else np.arange(n)
    xs, ys = x[sub], y[sub]
    boots = []
    for _ in range(BOOTSTRAP_B):
        i = rng.integers(0, m, m)
        r = stats.spearmanr(xs[i], ys[i]).statistic
        if not np.isnan(r):
            boots.append(r)
    if len(boots) < 10:
        return rho, float("nan"), float("nan")
    se = float(np.std(boots, ddof=1)) * np.sqrt(m / n)
    return rho, rho - 1.96 * se, rho + 1.96 * se


def drivers(frame: MarketFrame, mask: np.ndarray) -> dict:
    if mask.sum() < MIN_ANALYSIS_VIDEOS:
        return {"available": False, "reason": NOT_ENOUGH, "n": int(mask.sum()), "correlations": [], "heatmap": None}
    resid = residuals(frame, mask)
    titles = [t for t, m in zip(frame.titles, mask) if m]
    feats = {
        "duration_min": frame.duration_s[mask] / 60,
        **_title_features(titles),
        "tag_count": frame.tag_counts[mask].astype(float),
        "weekend": (frame.pub_weekday[mask] >= 5).astype(float),
    }
    y = resid[mask]
    rng = _seeded()
    out = []
    for key, x in feats.items():
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 30 or np.ptp(x[ok]) == 0:
            continue
        rho, lo, hi = spearman_ci(x[ok], y[ok], rng)
        out.append({"feature": key, "label": DRIVER_LABELS[key], "rho": rho, "ci_low": lo, "ci_high": hi, "n": int(ok.sum())})
    out.sort(key=lambda r: r["rho"])

    cells = []
    wd, hr = frame.pub_weekday[mask], frame.pub_hour[mask]
    for d in range(7):
        for h in range(24):
            sel = (wd == d) & (hr == h)
            cells.append(
                {
                    "weekday": d,
                    "hour": h,
                    "count": int(sel.sum()),
                    "median_residual": float(np.median(y[sel])) if sel.sum() >= 15 else None,
                }
            )
    return {"available": True, "n": int(mask.sum()), "correlations": out, "heatmap": {"weekdays": WEEKDAYS, "cells": cells}}


# ---- Growth: views vs age, velocity -------------------------------------------

AGE_BUCKETS = [(1, 7, "< 1 week"), (7, 30, "1–4 weeks"), (30, 90, "1–3 months"), (90, 365, "3–12 months"), (365, 10**6, "> 1 year")]


def growth(frame: MarketFrame, mask: np.ndarray, sample: int = 2000) -> dict:
    if mask.sum() < MIN_ANALYSIS_VIDEOS:
        return {"available": False, "reason": NOT_ENOUGH}
    ages = age_days(frame)
    fit = fit_log_age(ages[mask], frame.views[mask])

    edges = np.unique(np.round(np.logspace(0, np.log10(max(ages[mask].max(), 2)), 16)))
    binned = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = mask & (ages >= lo) & (ages < hi)
        if sel.sum() >= 10:
            q = np.percentile(frame.views[sel], [25, 50, 75])
            binned.append({"age_days": float(np.sqrt(lo * hi)), "q1": float(q[0]), "median": float(q[1]), "q3": float(q[2]), "n": int(sel.sum())})

    # Every one of the owner's videos, plus a seeded sample of the rest.
    own = mask & (frame.channel_ids == (frame.own_channel_id if frame.own_channel_id is not None else -1))
    idx = np.flatnonzero(mask & ~own)
    rng = _seeded()
    if len(idx) > sample:
        idx = rng.choice(idx, sample, replace=False)
    idx = np.sort(np.concatenate([idx, np.flatnonzero(own)]))
    points = [
        # Whole days and views: plenty for a log-log scatter, and a third of the payload.
        {"age_days": int(ages[i]), "views": int(frame.views[i]), "channel": frame.channel_titles.get(int(frame.channel_ids[i]), ""), "own": int(frame.channel_ids[i]) == frame.own_channel_id}
        for i in idx
    ]

    per_channel = []
    for cid in sorted(set(frame.channel_ids[mask].tolist())):
        m = mask & (frame.channel_ids == cid)
        if m.sum() >= MIN_CHANNEL_VIDEOS:
            f = fit_log_age(ages[m], frame.views[m])
            per_channel.append({"channel": frame.channel_titles.get(int(cid), str(cid)), "own": int(cid) == frame.own_channel_id, "slope": f.slope, "r2": f.r2, "n": f.n})
    per_channel.sort(key=lambda r: r["slope"])

    return {
        "available": True,
        "fit": {"intercept": fit.intercept, "slope": fit.slope, "r2": fit.r2, "n": fit.n},
        "binned": binned,
        "points": points,
        "per_channel": per_channel,
        "velocity": velocity(frame, mask, ages),
    }


def velocity(frame: MarketFrame, mask: np.ndarray, ages: np.ndarray) -> dict:
    """Views gained per day between the oldest and newest snapshot in the window."""
    vel = np.array([frame.velocity.get(v, (np.nan, 0))[0] for v in frame.video_ids])
    days = max((d for _, d in frame.velocity.values()), default=0)
    ok = mask & ~np.isnan(vel)
    if ok.sum() < 10:
        return {"available": False, "days": days, "reason": "Velocity needs at least two daily snapshots of the same videos."}
    buckets = []
    for lo, hi, label in AGE_BUCKETS:
        sel = ok & (ages >= lo) & (ages < hi)
        if sel.sum():
            q = np.percentile(vel[sel], [25, 50, 75])
            buckets.append({"label": label, "q1": float(q[0]), "median": float(q[1]), "q3": float(q[2]), "n": int(sel.sum())})
    top_idx = np.flatnonzero(ok)[np.argsort(-vel[ok])[:10]]
    top = [
        {"title": frame.titles[i], "channel": frame.channel_titles.get(int(frame.channel_ids[i]), ""), "views_per_day": float(vel[i]), "age_days": float(ages[i])}
        for i in top_idx
    ]
    return {"available": True, "days": days, "buckets": buckets, "top": top}


# ---- Title keywords: TF-IDF candidates, then lift on adjusted views -----------

_TOKEN = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
_STOP = STOPWORDS["en"] | STOPWORDS["es"]


def _norm(token: str) -> str:
    return unicodedata.normalize("NFKD", token.lower()).encode("ascii", "ignore").decode() or token.lower()


def title_tokens(title: str) -> set[str]:
    return {t for t in (_norm(w) for w in _TOKEN.findall(title)) if t not in _STOP and len(t) >= 4}


def keywords(frame: MarketFrame, mask: np.ndarray, min_videos: int = 40, min_channels: int = 3, top: int = 12, candidates: int = 400) -> dict:
    """Title words whose videos beat (or trail) their channel's age-adjusted expectation.

    1. Candidates: the `candidates` tokens with the highest summed TF-IDF over
       all titles, kept only if they appear in ≥ `min_videos` videos across
       ≥ `min_channels` channels, not part of a channel name and with no
       channel holding more than half of them (so a channel's vocabulary
       doesn't pass for a title effect).
    2. Lift: mean residual of videos with the word − mean residual of those
       without, with a Welch standard error for the 95% CI. ×10^lift is the
       multiplier on expected views.
    """
    idx = np.flatnonzero(mask)
    docs = [title_tokens(frame.titles[i]) for i in idx]
    n_docs = len(docs)
    if n_docs < min_videos * 2:
        return {"available": False, "reason": "Not enough videos yet.", "positive": [], "negative": []}

    df: dict[str, int] = {}
    per_channel: dict[str, dict[int, int]] = {}
    for d, i in zip(docs, idx):
        for t in d:
            df[t] = df.get(t, 0) + 1
            counts = per_channel.setdefault(t, {})
            counts[int(frame.channel_ids[i])] = counts.get(int(frame.channel_ids[i]), 0) + 1
    channel_words = set().union(*(title_tokens(name) for name in frame.channel_titles.values())) if frame.channel_titles else set()
    channels = {t: set(c) for t, c in per_channel.items()}

    def spread(t: str) -> bool:
        # Not a channel's name, and not mostly one channel's vocabulary.
        return t not in channel_words and max(per_channel[t].values()) <= 0.5 * df[t] and len(channels[t]) >= min_channels

    # Binary tf per title, so a token's summed TF-IDF is df · idf.
    tfidf = {t: c * np.log(n_docs / c) for t, c in df.items()}
    pool = [t for t in sorted(tfidf, key=tfidf.get, reverse=True)[:candidates] if df[t] >= min_videos and spread(t)]

    y = residuals(frame, mask)[idx]
    rows = []
    for t in pool:
        has = np.array([t in d for d in docs])
        a, b = y[has], y[~has]
        if len(a) < 2 or len(b) < 2:  # a word in (almost) every title has nothing to compare against
            continue
        lift = float(a.mean() - b.mean())
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        rows.append({"token": t, "videos": int(has.sum()), "channels": len(channels[t]), "lift": lift, "ci_low": lift - 1.96 * se, "ci_high": lift + 1.96 * se, "multiplier": float(10**lift)})
    rows.sort(key=lambda r: r["lift"], reverse=True)
    return {
        "available": bool(rows),
        "n": n_docs,
        "positive": [r for r in rows if r["lift"] > 0][:top],
        "negative": [r for r in reversed(rows) if r["lift"] < 0][:top],
    }
