"""Run directories for the M5 notebooks.

The committed `runs/2026-09-2x-*` document the original runs on the GTX
1060. A retrain sets `M5_RUN_TAG` (e.g. `2026-11-07-rtx5070`) so every
notebook writes, and reads its predecessors' results, under
`runs/<tag>-<kind>` instead of overwriting them (docs/decisions/0016).
"""
import os

ORIGINAL = {
    "data": "2026-09-27-data",
    "boundaries": "2026-09-28-boundaries",
    "titles": "2026-09-28-titles",
    "eval": "2026-09-28-eval",
    "titles-llm": "2026-09-28-titles-llm",
}


def run_dir(kind: str) -> str:
    if kind not in ORIGINAL:
        raise KeyError(f"unknown run kind {kind!r}")
    tag = os.environ.get("M5_RUN_TAG", "").strip()
    return f"runs/{tag}-{kind}" if tag else f"runs/{ORIGINAL[kind]}"


def qa_upload_id() -> int:
    """The owner's real episode for the QA cells (05, 06); it was upload 8."""
    return int(os.environ.get("M5_QA_UPLOAD_ID", "8"))
