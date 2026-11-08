"""Thin YouTube Data API v3 client: `channels`, `playlistItems`, `videos`
(each 1 unit per call), with a unit counter and a soft daily budget
(docs/decisions/0015 §5-6).

- A 403 `quotaExceeded`/`dailyLimitExceeded` raises `QuotaExceeded`, never
  retried: the run stops and keeps what it already wrote.
- 429, 5xx, 403 `rateLimitExceeded` and network errors are retried 3 times with exponential
  backoff.
- `UnitLedger` keys spend by the quota day (midnight Pacific, when Google
  resets it), so the budget spans every run in this process that day.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.market.durations import parse_duration

log = logging.getLogger("app.market")

API_BASE = "https://www.googleapis.com/youtube/v3"
QUOTA_TZ = ZoneInfo("America/Los_Angeles")
MAX_RETRIES = 3
_QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
_THUMBNAIL_PREFERENCE = ("high", "medium", "standard", "default", "maxres")


class YouTubeApiError(Exception):
    def __init__(self, message: str, *, status: int | None = None, reason: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason


class QuotaExceeded(YouTubeApiError):
    """Google's quota is spent (403 quotaExceeded). Stop; don't retry."""


class Unauthorized(YouTubeApiError):
    """HTTP 401: the credentials were rejected (an OAuth access token that
    Google no longer accepts, M11). The API-key client never expects it."""


class BudgetExhausted(YouTubeApiError):
    """Our own soft cap (`MARKET_DAILY_UNIT_BUDGET`) is reached."""


def quota_day(now: datetime | None = None) -> date:
    return (now or datetime.now(QUOTA_TZ)).astimezone(QUOTA_TZ).date()


@dataclass
class UnitLedger:
    budget: int
    clock: Callable[[], date] = quota_day
    _spent: dict[date, int] = field(default_factory=dict)

    @property
    def spent_today(self) -> int:
        return self._spent.get(self.clock(), 0)

    def reserve(self, units: int) -> None:
        day = self.clock()
        if self._spent.get(day, 0) + units > self.budget:
            raise BudgetExhausted(f"daily unit budget of {self.budget} reached")
        self._spent[day] = self._spent.get(day, 0) + units


class YouTubeClient:
    def __init__(
        self,
        api_key: str,
        http: httpx.AsyncClient,
        ledger: UnitLedger,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._key = api_key
        self._http = http
        self._ledger = ledger
        self._sleep = sleep
        self.units_used = 0  # by this client (one run)

    async def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ledger.reserve(1)
        self.units_used += 1
        # Key in a header, not the URL, so it never shows up in request
        # logs or exception messages.
        return await google_get(
            self._http,
            f"{API_BASE}/{endpoint}",
            params,
            headers={"X-Goog-Api-Key": self._key},
            label=endpoint,
            sleep=self._sleep,
        )

    async def channels(self, *, handle: str | None = None, channel_id: str | None = None) -> list[dict]:
        params: dict[str, Any] = {"part": "snippet,contentDetails"}
        if handle:
            params["forHandle"] = handle
        elif channel_id:
            params["id"] = channel_id
        else:
            raise ValueError("handle or channel_id required")
        return (await self._get("channels", params)).get("items", [])

    async def playlist_items_page(
        self, playlist_id: str, page_token: str | None = None, max_results: int = 50
    ) -> tuple[list[tuple[str, datetime]], str | None]:
        """One page of a playlist: ([(video ID, published_at)], next page token).

        Private/deleted entries carry no `videoPublishedAt` and are skipped.
        """
        params: dict[str, Any] = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": max_results}
        if page_token:
            params["pageToken"] = page_token
        data = await self._get("playlistItems", params)
        videos = [
            (d["videoId"], _parse_ts(d["videoPublishedAt"]))
            for d in (i.get("contentDetails", {}) for i in data.get("items", []))
            if d.get("videoId") and d.get("videoPublishedAt")
        ]
        return videos, data.get("nextPageToken")

    async def videos(self, ids: list[str]) -> list[dict]:
        if not 0 < len(ids) <= 50:
            raise ValueError("videos.list takes 1-50 IDs")
        data = await self._get(
            "videos", {"part": "snippet,contentDetails,statistics", "id": ",".join(ids), "maxResults": 50}
        )
        return data.get("items", [])


async def google_get(
    http: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    *,
    headers: dict[str, str],
    label: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    """GET a Google API with the M10 error policy (0015 §5), shared by the
    API-key client above and the OAuth clients of M11 (0017 §5).

    - 403 quota reasons -> `QuotaExceeded`, never retried.
    - 401 -> `Unauthorized` (the OAuth caller refreshes once and retries).
    - 429, 5xx, 403 `rateLimitExceeded` and network errors -> 3 retries with
      exponential backoff.
    - Anything else -> `YouTubeApiError` with the status and reason.
    """
    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await http.get(url, params=params, headers=headers, timeout=30)
        except httpx.HTTPError as exc:
            if attempt == MAX_RETRIES:
                raise YouTubeApiError(f"{label}: {exc}") from exc
        else:
            if response.status_code == 200:
                return response.json()
            reason = _error_reason(response)
            if response.status_code == 401:
                raise Unauthorized(f"{label}: HTTP 401 {reason or ''}".strip(), status=401, reason=reason)
            if response.status_code == 403 and reason in _QUOTA_REASONS:
                raise QuotaExceeded(f"{label}: {reason}")
            retryable = response.status_code == 429 or response.status_code >= 500 or reason == "rateLimitExceeded"
            if not retryable:
                raise YouTubeApiError(
                    f"{label}: HTTP {response.status_code} {reason or _error_message(response) or ''}".strip(),
                    status=response.status_code,
                    reason=reason,
                )
            if attempt == MAX_RETRIES:
                raise YouTubeApiError(f"{label}: HTTP {response.status_code} after retries")
        await sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def _error_message(response: httpx.Response) -> str | None:
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return None
    return message if isinstance(message, str) else None


def _error_reason(response: httpx.Response) -> str | None:
    try:
        errors = response.json()["error"]["errors"]
        return errors[0].get("reason")
    except (ValueError, KeyError, IndexError, TypeError):
        return None


# ---- Channel references ------------------------------------------------

_CHANNEL_ID = re.compile(r"UC[0-9A-Za-z_-]{22}")


def parse_channel_ref(ref: str) -> tuple[str, str]:
    """`@handle`, a channel URL or a `UC...` ID -> ("handle"|"id", value)."""
    ref = ref.strip()
    if m := re.search(r"/channel/(UC[0-9A-Za-z_-]{22})", ref):
        return "id", m.group(1)
    if m := re.search(r"youtube\.com/(@[^/?#]+)", ref):
        return "handle", m.group(1)
    if _CHANNEL_ID.fullmatch(ref):
        return "id", ref
    if ref.startswith("@") and len(ref) > 1:
        return "handle", ref
    raise ValueError(f"not a channel handle, URL or ID: {ref!r}")


# ---- Response mapping ----------------------------------------------------


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


def snapshot_fields(item: dict[str, Any]) -> dict[str, Any]:
    """A `videos.list` item -> `market_video_snapshots` column values (spec §4.2).

    Counters come as strings; hidden likes/comments are absent -> NULL. A
    duration that doesn't parse is logged and stored as NULL so the row
    still lands.
    """
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    raw_duration = item.get("contentDetails", {}).get("duration")
    duration = None
    if raw_duration is not None:
        try:
            duration = parse_duration(raw_duration)
        except ValueError:
            log.warning("market: unparseable duration %r for %s", raw_duration, item.get("id"))
    thumbs = snippet.get("thumbnails", {})
    thumbnail = next((thumbs[k]["url"] for k in _THUMBNAIL_PREFERENCE if k in thumbs), None)
    return {
        "youtube_video_id": item["id"],
        "title": snippet.get("title", ""),
        "published_at": _parse_ts(snippet["publishedAt"]),
        "duration_seconds": duration,
        "view_count": _int_or_none(stats.get("viewCount")),
        "like_count": _int_or_none(stats.get("likeCount")),
        "comment_count": _int_or_none(stats.get("commentCount")),
        "category_id": snippet.get("categoryId"),
        "tags": list(snippet.get("tags", [])),
        "thumbnail_url": thumbnail,
    }
