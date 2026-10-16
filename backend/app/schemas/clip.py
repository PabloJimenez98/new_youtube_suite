from pydantic import BaseModel


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
