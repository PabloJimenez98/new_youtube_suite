# Copied from experiments/m5-chapters/lib/units.py -- keep in sync (the
# experiment's tests cover the same logic; docs/decisions/0009 §2-4).
"""Text units shared by training and inference (docs/decisions/0009 §2).

VidChapters ASR comes as ~5 s sentences; our Whisper transcripts come as
sub-second segments with word timestamps. Both are normalised here into
the same kind of unit before encoding, so the model sees one input
distribution.

Input is a flat sequence of timed "atoms":
- ours: words (from `transcript_words`), which may be untimed (None);
- VidChapters: its ASR sentences.

Two strategies, compared in the experiment:
- `"sentence"`: close a unit at terminal punctuation once it is at least
  `min_s` long, or when it reaches `max_s`;
- `"block"`: fixed-duration blocks of about `block_s` seconds.

This module is copied verbatim into the backend (`app/chapters/units.py`)
once the experiment picks a strategy -- keep it dependency-free.
"""

from dataclasses import dataclass
from typing import Literal

TERMINAL = (".", "?", "!", "…", "。", "？", "！")


@dataclass(frozen=True)
class Atom:
    start: float | None
    end: float | None
    text: str


@dataclass(frozen=True)
class Unit:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class UnitConfig:
    mode: Literal["sentence", "block"] = "sentence"
    min_s: float = 2.0
    max_s: float = 15.0
    block_s: float = 10.0


def _join(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p.strip())


def build_units(atoms: list[Atom], cfg: UnitConfig = UnitConfig()) -> list[Unit]:
    """Group atoms into units; every unit has a real start/end timestamp.

    Untimed atoms (None timestamps) join whatever unit is open; untimed
    atoms before the first timed one join the first unit. Empty-text
    atoms are ignored. Units never contain only untimed atoms.
    """
    units: list[Unit] = []
    pending_untimed: list[str] = []
    cur_text: list[str] = []
    cur_start: float | None = None
    cur_end: float | None = None

    def close() -> None:
        nonlocal cur_text, cur_start, cur_end
        if cur_start is not None and cur_text:
            units.append(Unit(cur_start, cur_end, _join(cur_text)))
        cur_text, cur_start, cur_end = [], None, None

    for atom in atoms:
        text = atom.text.strip()
        if not text:
            continue
        if atom.start is None or atom.end is None:
            (cur_text if cur_start is not None else pending_untimed).append(text)
            continue

        if cur_start is not None:
            too_long = (
                atom.end - cur_start > cfg.max_s
                if cfg.mode == "sentence"
                else atom.start - cur_start >= cfg.block_s
            )
            if too_long:
                close()

        if cur_start is None:
            cur_start = atom.start
            cur_text = [*pending_untimed, text]
            pending_untimed = []
        else:
            cur_text.append(text)
        cur_end = max(cur_end or atom.end, atom.end)

        if (
            cfg.mode == "sentence"
            and text.endswith(TERMINAL)
            and cur_end - cur_start >= cfg.min_s
        ):
            close()

    if pending_untimed and cur_start is None and units:
        last = units[-1]
        units[-1] = Unit(last.start, last.end, _join([last.text, *pending_untimed]))
    close()
    return units
