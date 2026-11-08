from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, field_validator

from app.youtube_link import is_valid_youtube_id


class YoutubeStatsOut(BaseModel):
    """The linked video's latest public numbers, from the market pipeline (0022 §1)."""

    title: str
    published_at: datetime
    view_count: int | None
    like_count: int | None
    comment_count: int | None
    thumbnail_url: str | None
    snapshot_date: date


class UploadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    storage_path: str
    duration_seconds: float | None
    uploaded_at: datetime
    status: str
    youtube_video_id: str | None = None
    #: None when no video is linked or the market pipeline hasn't seen it.
    youtube: YoutubeStatsOut | None = None


class UploadUpdate(BaseModel):
    """`PATCH /uploads/{id}`: link (or unlink, with null) the YouTube video."""

    youtube_video_id: str | None

    @field_validator("youtube_video_id")
    @classmethod
    def _valid_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not is_valid_youtube_id(value):
            raise ValueError("must be an 11-character YouTube video ID")
        return value
