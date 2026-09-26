"""Word/segment speaker assignment from diarization turns.

Reimplements what spec §2.2 delegated to WhisperX's
`assign_word_speakers` (WhisperX itself isn't a dependency -- see
docs/decisions/0008-speaker-diarization.md §3). Pure functions, no I/O.

Turns must be non-overlapping -- pyannote 4's
`exclusive_speaker_diarization` guarantees that -- so sorting by start
also sorts by end, which the bisection below relies on.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def relabel_by_first_appearance(turns: Sequence[Turn]) -> list[Turn]:
    """Rename speakers `SPEAKER_00`, `SPEAKER_01`, ... in the order they
    first speak -- pyannote's own numbering is arbitrary."""
    ordered = sorted(turns, key=lambda t: t.start)
    names: dict[str, str] = {}
    for turn in ordered:
        names.setdefault(turn.speaker, f"SPEAKER_{len(names):02d}")
    return [Turn(t.start, t.end, names[t.speaker]) for t in ordered]


class SpeakerAssigner:
    def __init__(self, turns: Sequence[Turn]) -> None:
        self._turns = sorted(turns, key=lambda t: t.start)
        self._starts = [t.start for t in self._turns]

    def speaker_for(self, start: float | None, end: float | None) -> str | None:
        """Largest-overlap turn; a word in a gap between turns takes the
        nearest turn to its midpoint. No timestamps → None (spec §2.2.4)."""
        if start is None or end is None or not self._turns:
            return None

        best: Turn | None = None
        best_overlap = 0.0
        # Only turns starting before the word ends can overlap it; walk
        # back until turns end before the word starts.
        for i in range(bisect_left(self._starts, end) - 1, -1, -1):
            turn = self._turns[i]
            if turn.end <= start:
                break
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > best_overlap or (overlap == best_overlap and best is not None):
                best, best_overlap = turn, overlap
        if best is not None and best_overlap > 0:
            return best.speaker
        return self._nearest(start, end).speaker

    def _nearest(self, start: float, end: float) -> Turn:
        mid = (start + end) / 2
        i = bisect_right(self._starts, mid)
        neighbours = self._turns[max(0, i - 1) : i + 1]

        def distance(turn: Turn) -> float:
            if turn.start <= mid <= turn.end:
                return 0.0
            return min(abs(mid - turn.start), abs(mid - turn.end))

        return min(neighbours, key=distance)


def segment_speaker(word_speakers: Sequence[str | None]) -> str | None:
    """Majority speaker among a segment's words; ties go to the tied
    speaker who appears first (spec §2.2.3). All-None → None."""
    counts = Counter(s for s in word_speakers if s is not None)
    if not counts:
        return None
    top = max(counts.values())
    return next(s for s in word_speakers if s is not None and counts[s] == top)
