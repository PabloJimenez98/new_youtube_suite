"""OAuth clients for the connected channel (0017 §4-5).

- `OwnChannelClient`: Data API v3 with the bearer token. `channels.list
  mine=true` gives the channel and its uploads playlist, and
  `playlistItems.list` pages it, unlisted and private videos included.
  1 Data API unit per call.
- `AnalyticsClient`: `reports.query` with
  - `dimensions=day,video`: per-video daily engagement, at most
    `ENGAGEMENT_BATCH` IDs per filter
  - `dimensions=elapsedVideoTimeRatio`: retention, exactly one video
  The `columnHeaders` + `rows` response is mapped to dicts.

Every call goes through `TokenProvider.call_with_token`, which refreshes
on a 401, and through M10's `google_get` error policy: quota stop, and
retries for 429/5xx.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from app.market.youtube_api import API_BASE, google_get
from app.youtube_oauth.tokens import TokenProvider

ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
# playlistItems status of a deleted video's placeholder (seen on the first live run).
DELETED_PRIVACY = "privacyStatusUnspecified"
# The `video` filter takes up to 500 IDs; 200 keeps the response small.
ENGAGEMENT_BATCH = 200
# Rows per page for the day x video report (`startIndex` is 1-based).
PAGE_SIZE = 10_000

ENGAGEMENT_METRICS = (
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "likes",
    "comments",
    "shares",
    "subscribersGained",
    "subscribersLost",
)
RETENTION_METRICS = (
    "audienceWatchRatio",
    "relativeRetentionPerformance",
    "startedWatching",
    "stoppedWatching",
    "totalSegmentImpressions",
)
# API metric -> `analytics_snapshots` / `analytics_retention_curve` column.
ENGAGEMENT_COLUMNS = {
    "views": "views",
    "estimatedMinutesWatched": "estimated_minutes_watched",
    "averageViewDuration": "average_view_duration_seconds",
    "averageViewPercentage": "average_view_percentage",
    "likes": "likes",
    "comments": "comments",
    "shares": "shares",
    "subscribersGained": "subscribers_gained",
    "subscribersLost": "subscribers_lost",
}
RETENTION_COLUMNS = {
    "elapsedVideoTimeRatio": "elapsed_video_time_ratio",
    "audienceWatchRatio": "audience_watch_ratio",
    "relativeRetentionPerformance": "relative_retention_performance",
    "startedWatching": "started_watching",
    "stoppedWatching": "stopped_watching",
    "totalSegmentImpressions": "total_segment_impressions",
}
_INT_COLUMNS = {
    "views",
    "likes",
    "comments",
    "shares",
    "subscribers_gained",
    "subscribers_lost",
    "started_watching",
    "stopped_watching",
    "total_segment_impressions",
}


@dataclass(frozen=True)
class OwnChannel:
    channel_id: str
    title: str
    uploads_playlist_id: str | None


@dataclass(frozen=True)
class OwnVideo:
    youtube_video_id: str
    title: str
    published_at: datetime
    privacy_status: str | None


def report_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """`{"columnHeaders": [{"name": ...}], "rows": [[...]]}` -> list of dicts
    keyed by header name. No `rows` key (no data) -> []."""
    names = [h["name"] for h in payload.get("columnHeaders", [])]
    return [dict(zip(names, row, strict=True)) for row in payload.get("rows") or []]


def _cast(column: str, value: Any) -> Any:
    if value is None:
        return None
    return int(value) if column in _INT_COLUMNS else float(value)


def engagement_values(row: dict[str, Any]) -> dict[str, Any]:
    out = {"youtube_video_id": row["video"], "snapshot_date": date.fromisoformat(row["day"])}
    for metric, column in ENGAGEMENT_COLUMNS.items():
        out[column] = _cast(column, row.get(metric))
    return out


def retention_values(row: dict[str, Any]) -> dict[str, Any]:
    return {column: _cast(column, row.get(metric)) for metric, column in RETENTION_COLUMNS.items()}


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class _OAuthGet:
    def __init__(
        self,
        http: httpx.AsyncClient,
        tokens: TokenProvider,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._tokens = tokens
        self._sleep = sleep
        self.calls = 0

    async def _get(self, url: str, params: dict[str, Any], label: str) -> dict[str, Any]:
        self.calls += 1

        async def attempt(token: str) -> dict[str, Any]:
            return await google_get(
                self._http, url, params, headers={"Authorization": f"Bearer {token}"}, label=label, sleep=self._sleep
            )

        return await self._tokens.call_with_token(attempt)


class OwnChannelClient(_OAuthGet):
    async def my_channel(self) -> OwnChannel | None:
        data = await self._get(f"{API_BASE}/channels", {"part": "snippet,contentDetails", "mine": "true"}, "channels")
        items = data.get("items") or []
        if not items:
            return None
        item = items[0]
        return OwnChannel(
            channel_id=item["id"],
            title=item.get("snippet", {}).get("title", ""),
            uploads_playlist_id=item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads"),
        )

    async def uploads(self, playlist_id: str) -> list[OwnVideo]:
        """The whole uploads playlist, newest first.

        `videoPublishedAt` is missing for never-published private videos;
        the playlist-insert time (`snippet.publishedAt`) stands in for it.

        A deleted video lingers in the playlist as a placeholder ("Deleted
        video", `privacyStatusUnspecified`). It is skipped, so the pull marks
        an already-known one `removed_since` like any other departure.
        """
        videos: list[OwnVideo] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {"part": "snippet,contentDetails,status", "playlistId": playlist_id, "maxResults": 50}
            if page_token:
                params["pageToken"] = page_token
            data = await self._get(f"{API_BASE}/playlistItems", params, "playlistItems")
            for item in data.get("items", []):
                details = item.get("contentDetails", {})
                snippet = item.get("snippet", {})
                vid = details.get("videoId")
                published = details.get("videoPublishedAt") or snippet.get("publishedAt")
                privacy = item.get("status", {}).get("privacyStatus")
                if not vid or not published or privacy == DELETED_PRIVACY:
                    continue
                videos.append(
                    OwnVideo(
                        youtube_video_id=vid,
                        title=snippet.get("title", ""),
                        published_at=_parse_ts(published),
                        privacy_status=privacy,
                    )
                )
            page_token = data.get("nextPageToken")
            if not page_token:
                return videos


class AnalyticsClient(_OAuthGet):
    async def _query(self, params: dict[str, Any], label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        start_index = 1
        while True:
            page = report_rows(
                await self._get(
                    ANALYTICS_URL,
                    {"ids": "channel==MINE", **params, "startIndex": start_index, "maxResults": PAGE_SIZE},
                    label,
                )
            )
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            start_index += PAGE_SIZE

    async def engagement(self, video_ids: list[str], start: date, end: date) -> list[dict[str, Any]]:
        if not 0 < len(video_ids) <= ENGAGEMENT_BATCH:
            raise ValueError(f"engagement takes 1-{ENGAGEMENT_BATCH} video IDs")
        rows = await self._query(
            {
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "dimensions": "day,video",
                "metrics": ",".join(ENGAGEMENT_METRICS),
                "filters": "video==" + ",".join(video_ids),
                "sort": "day",
            },
            "reports(engagement)",
        )
        return [engagement_values(r) for r in rows]

    async def retention(self, video_id: str, start: date, end: date) -> list[dict[str, Any]]:
        rows = await self._query(
            {
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "dimensions": "elapsedVideoTimeRatio",
                "metrics": ",".join(RETENTION_METRICS),
                "filters": f"video=={video_id}",
                "sort": "elapsedVideoTimeRatio",
            },
            "reports(retention)",
        )
        return [retention_values(r) for r in rows]
