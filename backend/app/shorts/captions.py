"""Karaoke captions for one clip as an ASS file (docs/decisions/0013 §3).

Pure. The clip's transcript words are grouped into short cues; each cue
is shown as one `Dialogue` per word, in which that word is yellow and
the rest white, so the highlight follows the speech. Times are relative
to the clip start.
"""

from dataclasses import dataclass

from app.shorts.selection import TimedWord, fill_times

MIN_WORDS = 3
MAX_WORDS = 6
MAX_CHARS = 28
PAUSE_S = 0.7
BREAK_AFTER = (".", "?", "!", "…", ",", ";", ":")

FONT = "DejaVu Sans"
WHITE = "&H00FFFFFF"  # style colours are &HAABBGGRR
BLACK = "&H00000000"
# Inline colour overrides take &HBBGGRR&.
HL = "&H00FFFF&"
NORMAL = "&HFFFFFF&"


@dataclass(frozen=True)
class CueWord:
    start: float
    end: float
    text: str


def clip_words(words: list[TimedWord], start: float, end: float) -> list[CueWord]:
    """Words whose start falls inside [start, end), relative to `start`, clamped to the clip."""
    out = []
    for w in fill_times(words):
        text = w.text.strip()
        if not text or w.start < start or w.start >= end:
            continue
        out.append(CueWord(w.start - start, min(max(w.end, w.start), end) - start, text))
    return out


def group_cues(words: list[CueWord]) -> list[list[CueWord]]:
    """3-6 words, <= MAX_CHARS; break after punctuation (once MIN_WORDS long) and at pauses."""
    cues: list[list[CueWord]] = []
    cur: list[CueWord] = []
    for w in words:
        if cur:
            chars = len(" ".join(x.text for x in cur + [w]))
            pause = w.start - cur[-1].end > PAUSE_S
            full = len(cur) >= MAX_WORDS or chars > MAX_CHARS
            punct = len(cur) >= MIN_WORDS and cur[-1].text.endswith(BREAK_AFTER)
            if pause or full or punct:
                cues.append(cur)
                cur = []
        cur.append(w)
    if cur:
        cues.append(cur)
    return cues


def escape(text: str) -> str:
    """ASS override blocks use braces and backslashes: keep them out of spoken text."""
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def ts(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def dialogues(cue: list[CueWord]) -> list[tuple[float, float, str]]:
    """One (start, end, text) per word: the word is highlighted until the next word starts."""
    out = []
    for k, w in enumerate(cue):
        end = cue[k + 1].start if k + 1 < len(cue) else max(w.end, w.start + 0.01)
        parts = [
            f"{{\\c{HL}}}{escape(x.text)}{{\\c{NORMAL}}}" if x is w else escape(x.text)
            for x in cue
        ]
        out.append((w.start, max(end, w.start + 0.01), " ".join(parts)))
    return out


def build_ass(words: list[TimedWord], start: float, end: float, width: int, height: int) -> str:
    """The ASS document for a clip rendered at width x height."""
    size = round(min(width, height) * 0.075)
    margin_v = round(height * 0.2)
    outline = max(2, round(size / 14))
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT},{size},{WHITE},{WHITE},{BLACK},{BLACK},"
        f"-1,0,0,0,100,100,0,0,1,{outline},0,2,{round(width * 0.06)},{round(width * 0.06)},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [
        f"Dialogue: 0,{ts(s)},{ts(e)},Default,,0,0,0,,{text}"
        for cue in group_cues(clip_words(words, start, end))
        for s, e, text in dialogues(cue)
    ]
    return header + "\n".join(lines) + ("\n" if lines else "")
