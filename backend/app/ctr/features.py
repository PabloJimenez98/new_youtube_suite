"""Engineered features for the CTR heuristic v0 (docs/decisions/0019 §1).

- Title features are pure string functions.
- Thumbnail features take an image (BGR `numpy` array) plus the detector
  outputs, so the arithmetic is testable without the ONNX models.
- Only `title_chars`, `face_count`, `largest_face_area` and `text_area`
  carry weight in v0. The others are stored with weight 0 for later
  calibration: the research found no CTR evidence for them
  (docs/research/ctr-heuristic-evidence.md).
"""

import re
from dataclasses import asdict, dataclass

import numpy as np

MAX_TITLE_CHARS = 100  # YouTube's cap
THUMBNAIL_BOX = (1280, 720)  # images are scaled to fit this before detection

_DIGIT = re.compile(r"\d")
_BRACKETS = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SECOND_PERSON = re.compile(r"\b(you|your|yours|you're|youre|tú|tu|tus|te|ti|usted|ustedes|vosotros)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TitleFeatures:
    title_chars: int
    title_words: int
    has_digit: bool
    has_question: bool
    has_exclamation: bool
    has_brackets: bool
    caps_words: int
    second_person: bool

    def as_dict(self) -> dict:
        return asdict(self)


def title_features(title: str) -> TitleFeatures:
    text = title.strip()
    words = _WORD.findall(text)
    return TitleFeatures(
        title_chars=len(text),
        title_words=len(text.split()),
        has_digit=bool(_DIGIT.search(text)),
        has_question="?" in text or "¿" in text,
        has_exclamation="!" in text or "¡" in text,
        has_brackets=bool(_BRACKETS.search(text)),
        caps_words=sum(1 for w in words if len(w) >= 2 and w.isupper()),
        second_person=bool(_SECOND_PERSON.search(text)),
    )


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixels of the (resized) thumbnail."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class ThumbnailFeatures:
    width: int
    height: int
    aspect_ratio: float
    face_count: int
    largest_face_area: float  # fraction of the image
    text_area: float  # fraction of the image covered by the union of text boxes
    text_boxes: int
    brightness: float  # 0-1
    contrast: float  # 0-1
    saturation: float  # 0-1

    def as_dict(self) -> dict:
        return asdict(self)


def fit_size(width: int, height: int, box: tuple[int, int] = THUMBNAIL_BOX) -> tuple[int, int]:
    """Scale (width, height) down to fit `box`, keeping the aspect ratio. Never scales up."""
    scale = min(box[0] / width, box[1] / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def union_area(boxes: list[Box], width: int, height: int) -> float:
    """Fraction of a width x height image covered by the union of `boxes` (clipped to the image)."""
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    mask = np.zeros((height, width), dtype=bool)
    for b in boxes:
        x0, y0 = max(0, int(round(b.x))), max(0, int(round(b.y)))
        x1, y1 = min(width, int(round(b.x + b.w))), min(height, int(round(b.y + b.h)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def image_stats(image: np.ndarray) -> tuple[float, float, float]:
    """(brightness, contrast, saturation) of a BGR image, each 0-1."""
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(hsv[:, :, 2].mean() / 255.0)
    saturation = float(hsv[:, :, 1].mean() / 255.0)
    # Std of luminance; 0.5 is the maximum (half black, half white).
    contrast = float(min(gray.std() / 255.0 / 0.5, 1.0))
    return brightness, contrast, saturation


def thumbnail_features(image: np.ndarray, faces: list[Box], texts: list[Box]) -> ThumbnailFeatures:
    height, width = image.shape[:2]
    area = float(width * height)
    largest = max((f.w * f.h for f in faces), default=0.0) / area
    brightness, contrast, saturation = image_stats(image)
    return ThumbnailFeatures(
        width=width,
        height=height,
        aspect_ratio=round(width / height, 4),
        face_count=len(faces),
        largest_face_area=round(min(largest, 1.0), 4),
        text_area=round(union_area(texts, width, height), 4),
        text_boxes=len(texts),
        brightness=round(brightness, 4),
        contrast=round(contrast, 4),
        saturation=round(saturation, 4),
    )
