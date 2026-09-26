"""`.srt` / `.vtt` rendering from transcript segments.

Pure functions, one cue per segment (project-description.md §2.1).
Generated on demand rather than cached in blob storage -- see
docs/decisions/0007-transcription-pipeline.md §5.
"""

from collections.abc import Iterable
from typing import Protocol


class Cue(Protocol):
    start_seconds: float
    end_seconds: float
    text: str


def format_timestamp(seconds: float, ms_separator: str) -> str:
    """`HH:MM:SS<sep>mmm`. Rounds to whole ms first so `,1000` can't occur."""
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_separator}{ms:03d}"


def _clean_text(text: str) -> str:
    # A blank line terminates a cue in both formats, so drop empty lines.
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _cues(segments: Iterable[Cue]) -> Iterable[tuple[float, float, str]]:
    for seg in segments:
        text = _clean_text(seg.text)
        if not text:
            continue
        start = max(0.0, seg.start_seconds)
        yield start, max(start, seg.end_seconds), text


def to_srt(segments: Iterable[Cue]) -> str:
    blocks = [
        f"{i}\n{format_timestamp(start, ',')} --> {format_timestamp(end, ',')}\n{text}\n"
        for i, (start, end, text) in enumerate(_cues(segments), start=1)
    ]
    return "\n".join(blocks)


def _escape_vtt(text: str) -> str:
    # Escaping `>` also neutralizes a literal `-->` inside cue text.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(segments: Iterable[Cue]) -> str:
    blocks = [
        f"{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}\n{_escape_vtt(text)}\n"
        for start, end, text in _cues(segments)
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)
