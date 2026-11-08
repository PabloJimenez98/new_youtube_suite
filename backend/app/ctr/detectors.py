"""Face and text detectors for thumbnails (docs/decisions/0019 §1).

- **Faces:** YuNet, the M8 model (`SHORTS_FACE_MODEL_PATH`).
- **Text:** OpenCV's DB detector with the opencv_zoo PP-OCRv3 English
  model (`CTR_TEXT_MODEL_PATH`), using the demo's parameters. It detects
  text only, with no recognition.

Both are loaded once per process and reused. A missing model file raises
`DetectorModelMissingError` with the setting to fix, so the job fails
clearly.
"""

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.ctr.features import Box

FACE_SCORE = 0.6
TEXT_INPUT = (736, 736)  # the demo's size; must be a multiple of 32
_lock = threading.Lock()


class DetectorModelMissingError(Exception):
    """A detector's ONNX file is not configured or not on disk."""


def require_model(path: Path | None, setting: str) -> Path:
    if path is None or not Path(path).is_file():
        raise DetectorModelMissingError(f"No model at {path} -- set {setting} (see README, 'CTR predictor')")
    return Path(path)


@lru_cache(maxsize=2)
def _text_model(path: str):
    import cv2

    model = cv2.dnn.TextDetectionModel_DB(cv2.dnn.readNet(path))
    model.setBinaryThreshold(0.3)
    model.setPolygonThreshold(0.5)
    model.setMaxCandidates(200)
    model.setUnclipRatio(2.0)
    model.setInputParams(1.0 / 255.0, TEXT_INPUT, (123.675, 116.28, 103.53))
    model.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
    return model


def detect_faces(image: np.ndarray, model_path: Path | None) -> list[Box]:
    import cv2

    path = require_model(model_path, "SHORTS_FACE_MODEL_PATH")
    height, width = image.shape[:2]
    # YuNet's input size is per call; creating it is cheap (~230 KB model).
    detector = cv2.FaceDetectorYN.create(str(path), "", (width, height), FACE_SCORE, 0.3, 50)
    _, faces = detector.detect(image)
    if faces is None:
        return []
    return [Box(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in faces]


def detect_text(image: np.ndarray, model_path: Path | None) -> list[Box]:
    import cv2

    path = require_model(model_path, "CTR_TEXT_MODEL_PATH")
    height, width = image.shape[:2]
    resized = cv2.resize(image, TEXT_INPUT)
    with _lock:  # one cv2.dnn net shared by threads
        polygons, _ = _text_model(str(path)).detect(resized)
    sx, sy = width / TEXT_INPUT[0], height / TEXT_INPUT[1]
    boxes = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=float)
        x0, y0 = pts[:, 0].min() * sx, pts[:, 1].min() * sy
        x1, y1 = pts[:, 0].max() * sx, pts[:, 1].max() * sy
        boxes.append(Box(x0, y0, x1 - x0, y1 - y0))
    return boxes
