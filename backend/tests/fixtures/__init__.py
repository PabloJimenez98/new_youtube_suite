"""Shared test media -- see ATTRIBUTION.md for source and license."""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
FIXTURE_CLIP = FIXTURES_DIR / "librispeech_clip.mp4"
FIXTURE_CLIP_REFERENCE = (FIXTURES_DIR / "librispeech_clip.txt").read_text().strip()
