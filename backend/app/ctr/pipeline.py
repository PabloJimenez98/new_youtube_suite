"""Thumbnail validation, analysis and candidate scoring (docs/decisions/0019 §2, §4).

- **`validate_thumbnail(data)`**, at upload time:
  - JPEG/PNG/WebP, sniffed from the magic bytes
  - ≤ `MAX_THUMBNAIL_BYTES` (YouTube's own 2 MB limit)
  - must decode with OpenCV
  - raises `InvalidThumbnailError` with a reason the API returns as 422
- **`analyse_thumbnail(data, …)`:** decode → scale to fit 1280 × 720 →
  faces + text → `ThumbnailFeatures`.
- **`score_candidate(title, thumbnail_bytes, …)`:** features → `Score`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ctr import detectors
from app.ctr.features import ThumbnailFeatures, TitleFeatures, fit_size, thumbnail_features, title_features
from app.ctr.scorer import Score, score

MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
MAX_CANDIDATES_PER_UPLOAD = 10
_MAGIC = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
}


class InvalidThumbnailError(ValueError):
    pass


def sniff_type(data: bytes) -> str | None:
    for media_type, prefixes in _MAGIC.items():
        if any(data.startswith(p) for p in prefixes):
            return media_type
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def decode(data: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidThumbnailError("the image could not be decoded")
    return image


def validate_thumbnail(data: bytes) -> str:
    """Returns the sniffed media type, or raises `InvalidThumbnailError`."""
    if not data:
        raise InvalidThumbnailError("the file is empty")
    if len(data) > MAX_THUMBNAIL_BYTES:
        raise InvalidThumbnailError(f"the image is larger than {MAX_THUMBNAIL_BYTES // (1024 * 1024)} MB")
    media_type = sniff_type(data)
    if media_type is None:
        raise InvalidThumbnailError("only JPEG, PNG and WebP images are accepted")
    decode(data)
    return media_type


def analyse_thumbnail(data: bytes, face_model: Path | None, text_model: Path | None) -> ThumbnailFeatures:
    import cv2

    image = decode(data)
    height, width = image.shape[:2]
    size = fit_size(width, height)
    if size != (width, height):
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    faces = detectors.detect_faces(image, face_model)
    texts = detectors.detect_text(image, text_model)
    return thumbnail_features(image, faces, texts)


@dataclass(frozen=True)
class CandidateResult:
    score: Score
    title: TitleFeatures
    thumbnail: ThumbnailFeatures | None

    @property
    def features(self) -> dict:
        return self.score.features_json(self.title, self.thumbnail)


def score_candidate(
    title: str, thumbnail: bytes | None, face_model: Path | None, text_model: Path | None
) -> CandidateResult:
    t = title_features(title)
    thumb = analyse_thumbnail(thumbnail, face_model, text_model) if thumbnail is not None else None
    return CandidateResult(score(t, thumb), t, thumb)
