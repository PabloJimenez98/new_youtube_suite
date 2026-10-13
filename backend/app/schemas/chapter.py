from pydantic import BaseModel, ConfigDict


class ChapterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    order_index: int
    start_seconds: float
    title: str
    model_version: str
