"""Pure selection rules for the analytics pull (0017 §5).

- Retention reuses M10's due rule (0015 §3) with the date of the last
  retention pull as the "last snapshot": daily for videos ≤ 90 days old,
  weekly for older ones. A pull that came back empty still counts, so a
  video with no views isn't re-asked every day.
- Engagement windows: from the publish date on a video's first pull,
  then the last `REPULL_DAYS` days, which absorbs the 48-72 h lag and
  late revisions.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.market.selection import VideoState, is_due

REPULL_DAYS = 7


@dataclass(frozen=True)
class OwnVideoState:
    youtube_video_id: str
    published_at: datetime
    removed: bool
    engagement_backfilled: bool
    retention_pulled_on: date | None


def retention_due(
    videos: list[OwnVideoState], today: date, recent_window_days: int, older_interval_days: int
) -> list[str]:
    return [
        v.youtube_video_id
        for v in videos
        if is_due(
            VideoState(
                youtube_video_id=v.youtube_video_id,
                published_at=v.published_at,
                last_snapshot_date=v.retention_pulled_on,
                unavailable=v.removed,
                channel_active=True,
            ),
            today,
            recent_window_days,
            older_interval_days,
        )
    ]


def engagement_window(video: OwnVideoState, today: date) -> date:
    """The start date for this video's engagement query (end = today)."""
    if not video.engagement_backfilled:
        return min(video.published_at.date(), today)
    return today - timedelta(days=REPULL_DAYS)
