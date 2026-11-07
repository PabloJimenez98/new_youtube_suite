"""Which videos are due for a snapshot on a given day (0015 §3). Pure."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class VideoState:
    youtube_video_id: str
    published_at: datetime
    last_snapshot_date: date | None
    unavailable: bool
    channel_active: bool


def is_due(video: VideoState, today: date, recent_window_days: int, older_interval_days: int) -> bool:
    if video.unavailable or not video.channel_active:
        return False
    if video.last_snapshot_date == today:
        return False
    if (today - video.published_at.date()) <= timedelta(days=recent_window_days):
        return True  # recent: every day
    if video.last_snapshot_date is None:
        return True
    return (today - video.last_snapshot_date) >= timedelta(days=older_interval_days)


def select_due(
    videos: list[VideoState], today: date, recent_window_days: int, older_interval_days: int
) -> list[str]:
    return [
        v.youtube_video_id
        for v in videos
        if is_due(v, today, recent_window_days, older_interval_days)
    ]
