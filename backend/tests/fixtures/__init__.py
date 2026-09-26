"""Shared test media -- see ATTRIBUTION.md (here and in app/assets/) for
sources and licenses."""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
FIXTURE_CLIP = FIXTURES_DIR / "librispeech_clip.mp4"
FIXTURE_CLIP_REFERENCE = (FIXTURES_DIR / "librispeech_clip.txt").read_text().strip()

# Two-speaker A/B/A clip (M4) -- same audio as the worker's diarization
# healthcheck clip in app/assets/, muxed into an MP4 for full-pipeline tests.
CONVERSATION_CLIP = FIXTURES_DIR / "conversation_clip.mp4"
