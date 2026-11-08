from datetime import datetime

from pydantic import BaseModel


class HookOut(BaseModel):
    """The clip's latest hook score (M14, docs/decisions/0020 §5)."""

    score: float  # 0-100, relative among the upload's clips
    rank: int  # among the upload's scored clips; ties share a rank
    features: dict  # the stored feature_vector
    scored_at: datetime


class ClipOut(BaseModel):
    id: int
    rank: int
    start: float
    end: float
    duration: float
    score: float
    aspect_ratio: str
    caption_burned_in: bool
    # How the clip was framed: {source, output, shots: [{start, end, mode}]}.
    framing: dict
    # None until the `hook` job has scored this clip.
    hook: HookOut | None = None
