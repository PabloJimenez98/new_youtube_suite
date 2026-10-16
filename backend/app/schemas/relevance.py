from pydantic import BaseModel


class RelevanceWindowOut(BaseModel):
    start: float
    end: float
    score: float


class TopMomentOut(RelevanceWindowOut):
    text: str


class RelevanceOut(BaseModel):
    # `version` rather than `model_version`: pydantic reserves the `model_` prefix.
    version: str
    windows: list[RelevanceWindowOut]
    top_moments: list[TopMomentOut]
