"""Transcript words -> 7.5 s blocks -> 15 s windows with 50 % overlap.

Built by walking word timestamps, independent of Whisper's segment
boundaries (spec §2.4 step 1). A window is two consecutive blocks, so
the hop is one block. See docs/decisions/0012-relevance-v0.md §1.
"""

import math
from dataclasses import dataclass

BLOCK_S = 7.5
BLOCKS_PER_WINDOW = 2


@dataclass(frozen=True)
class Word:
    start: float | None
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class Block:
    index: int
    start: float
    end: float
    words: tuple[Word, ...]


@dataclass(frozen=True)
class Window:
    index: int
    start: float
    end: float
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text.strip() for w in self.words if w.text.strip())


def build_blocks(words: list[Word], duration: float, block_s: float = BLOCK_S) -> list[Block]:
    """Fixed-length blocks over [0, duration]; an untimed word joins the previous timed word's block."""
    n = max(1, math.ceil(duration / block_s))
    buckets: list[list[Word]] = [[] for _ in range(n)]
    current = 0
    for w in words:
        if w.start is not None:
            current = min(max(int(w.start // block_s), 0), n - 1)
        buckets[current].append(w)
    return [
        Block(i, i * block_s, min((i + 1) * block_s, duration), tuple(b))
        for i, b in enumerate(buckets)
    ]


def build_windows(blocks: list[Block], span: int = BLOCKS_PER_WINDOW) -> list[Window]:
    """Window i = blocks i .. i+span-1, so consecutive windows overlap by span-1 blocks."""
    if not blocks:
        return []
    count = max(len(blocks) - span + 1, 1)
    windows = []
    for i in range(count):
        group = blocks[i : i + span]
        windows.append(Window(i, group[0].start, group[-1].end, tuple(w for b in group for w in b.words)))
    return windows
