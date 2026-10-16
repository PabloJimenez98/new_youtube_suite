"""`.srt` / `.vtt` rendering from transcript segments.

Pure functions, one cue per segment by default (project-description.md §2.1).
Generated on demand rather than cached in blob storage -- see
docs/decisions/0007-transcription-pipeline.md §5. Segments with a
`speaker_label` (M4 diarization) get `[SPEAKER_00] ` in SRT and a
`<v SPEAKER_00>` voice span in VTT; unlabelled output is unchanged --
docs/decisions/0008-speaker-diarization.md §6.

A segment whose words span several speakers (a quick interjection that
Whisper kept inside one segment) is split into one cue per speaker run,
using the per-word labels -- otherwise the whole cue would show only the
majority speaker. Runs shorter than `MIN_RUN_WORDS` are folded into a
neighbour so a single mislabelled word doesn't become its own cue.

A cue longer than `MAX_CUE_CHARS` characters or `MAX_CUE_S` seconds is
split on its word timestamps, preferably after punctuation (M9,
docs/decisions/0014 §1): batched Whisper returns ~30 s segments, which
would otherwise be 30 s cues.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol

MIN_RUN_WORDS = 2
MAX_CUE_CHARS = 84  # two lines of ~42
MAX_CUE_S = 7.0
_BREAK_AFTER = (".", "?", "!", "…", ",", ";", ":")


class CueWord(Protocol):
    word: str
    start_seconds: float | None
    end_seconds: float | None
    speaker_label: str | None


class Cue(Protocol):
    start_seconds: float
    end_seconds: float
    text: str
    speaker_label: str | None
    words: Sequence[CueWord]


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


def _speaker_runs(words: Sequence[CueWord]) -> list[list[CueWord]]:
    """Consecutive words grouped by speaker. Unlabelled words stay with the
    run they sit in; runs under `MIN_RUN_WORDS` fold into a neighbour."""
    runs: list[list[CueWord]] = []
    speakers: list[str | None] = []  # each run's speaker (None until one is seen)
    for word in words:
        label = word.speaker_label
        if runs and (label is None or speakers[-1] is None or label == speakers[-1]):
            runs[-1].append(word)
            if speakers[-1] is None:
                # Leading unlabelled words belong to the first labelled speaker.
                speakers[-1] = label
        else:
            runs.append([word])
            speakers.append(label)

    while len(runs) > 1 and (short := next(
        (i for i, run in enumerate(runs) if len(run) < MIN_RUN_WORDS), None
    )) is not None:
        if short == 0:
            runs[1] = runs[0] + runs[1]
        else:
            runs[short - 1] += runs[short]
        del runs[short]
        # Folding can leave two neighbours with the same speaker -- rejoin them.
        merged: list[list[CueWord]] = [runs[0]]
        for run in runs[1:]:
            if _run_speaker(run) == _run_speaker(merged[-1]):
                merged[-1] += run
            else:
                merged.append(run)
        runs = merged
    return runs


def _run_speaker(run: Sequence[CueWord]) -> str | None:
    counts = Counter(w.speaker_label for w in run if w.speaker_label is not None)
    return counts.most_common(1)[0][0] if counts else None


def _first_start(run: Sequence[CueWord]) -> float | None:
    return next((w.start_seconds for w in run if w.start_seconds is not None), None)


def _split_long(start: float, end: float, text: str, words: Sequence[CueWord]
                ) -> Iterable[tuple[float, float, str]]:
    """Cut a cue that is too long into pieces on its timed words; short ones pass unchanged."""
    if (len(text) <= MAX_CUE_CHARS and end - start <= MAX_CUE_S) or not words:
        yield start, end, text
        return
    pieces: list[list[CueWord]] = [[]]
    for w in words:
        cur = pieces[-1]
        if cur:
            chars = len(" ".join(x.word.strip() for x in [*cur, w]))
            first = _first_start(cur)
            span = (w.end_seconds or w.start_seconds or 0.0) - (first if first is not None else start)
            text_so_far = " ".join(x.word.strip() for x in cur)
            soft = len(text_so_far) >= MAX_CUE_CHARS // 2 and cur[-1].word.strip().endswith(_BREAK_AFTER)
            if chars > MAX_CUE_CHARS or span > MAX_CUE_S or soft:
                pieces.append([])
        pieces[-1].append(w)
    starts = [start] + [_first_start(piece) for piece in pieces[1:]]
    for i, piece in enumerate(pieces):
        piece_text = " ".join(w.word.strip() for w in piece if w.word.strip())
        if not piece_text:
            continue
        s = starts[i] if starts[i] is not None else start
        e = next((x for x in starts[i + 1:] if x is not None), end)
        s = min(max(s, start), end)
        yield s, min(max(e, s), end), piece_text


def _cues(segments: Iterable[Cue]) -> Iterable[tuple[float, float, str, str | None]]:
    for seg in segments:
        text = _clean_text(seg.text)
        if not text:
            continue
        seg_start = max(0.0, seg.start_seconds)
        seg_end = max(seg_start, seg.end_seconds)

        runs = _speaker_runs(seg.words) if seg.words else []
        if len(runs) < 2:
            # One speaker (or no word labels): the segment is the cue, as in M3,
            # cut into pieces only when it is too long.
            for s, e, piece in _split_long(seg_start, seg_end, text, seg.words or []):
                yield s, e, piece, seg.speaker_label
            continue

        # Each run's cue lasts until the next run's first timed word.
        starts = [seg_start] + [_first_start(run) for run in runs[1:]]
        for i, run in enumerate(runs):
            start = starts[i] if starts[i] is not None else seg_start
            end = next((s for s in starts[i + 1 :] if s is not None), seg_end)
            start = min(max(start, seg_start), seg_end)
            end = min(max(end, start), seg_end)
            run_text = " ".join(w.word.strip() for w in run if w.word.strip())
            if run_text:
                speaker = _run_speaker(run)
                for s, e, piece in _split_long(start, end, run_text, run):
                    yield s, e, piece, speaker


def to_srt(segments: Iterable[Cue]) -> str:
    blocks = [
        f"{i}\n{format_timestamp(start, ',')} --> {format_timestamp(end, ',')}\n"
        f"{f'[{speaker}] ' if speaker else ''}{text}\n"
        for i, (start, end, text, speaker) in enumerate(_cues(segments), start=1)
    ]
    return "\n".join(blocks)


def _escape_vtt(text: str) -> str:
    # Escaping `>` also neutralizes a literal `-->` inside cue text.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(segments: Iterable[Cue]) -> str:
    blocks = [
        f"{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}\n"
        f"{f'<v {_escape_vtt(speaker)}>' if speaker else ''}{_escape_vtt(text)}\n"
        for start, end, text, speaker in _cues(segments)
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)
