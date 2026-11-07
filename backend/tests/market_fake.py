"""In-memory fake of the three Data API endpoints + the RSS feed, served
through `httpx.MockTransport`, for the M10 client/orchestration tests."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from app.market.youtube_api import UnitLedger, YouTubeClient


def video_item(vid: str, channel_id: str, published: datetime, views: int = 100, title: str | None = None) -> dict:
    return {
        "id": vid,
        "snippet": {
            "publishedAt": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "channelId": channel_id,
            "title": title or f"Video {vid}",
            "thumbnails": {"high": {"url": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"}},
            "categoryId": "22",
        },
        "contentDetails": {"duration": "PT30M"},
        "statistics": {"viewCount": str(views), "likeCount": "10", "commentCount": "2"},
    }


@dataclass
class FakeChannel:
    channel_id: str
    handle: str
    title: str
    videos: list[dict] = field(default_factory=list)  # newest first

    @property
    def uploads(self) -> str:
        return "UU" + self.channel_id[2:]


@dataclass
class FakeYouTube:
    channels: list[FakeChannel] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    hidden: set[str] = field(default_factory=set)  # deleted/private: not returned by videos.list
    # endpoint -> list of status codes to return before succeeding
    fail_next: dict[str, list[int]] = field(default_factory=dict)
    quota_after: int | None = None  # 403 quotaExceeded once this many API calls succeeded
    rss: dict[str, str] = field(default_factory=dict)  # channel_id -> feed XML
    # channels whose uploads playlist 404s (seen live: a channel with no public uploads)
    no_playlist: set[str] = field(default_factory=set)
    # videos.list calls (1-based) that answer 400 badRequest
    bad_videos_calls: set[int] = field(default_factory=set)

    def add_channel(self, n_videos: int, *, idx: int = 0, newest: datetime | None = None, days_apart: int = 7) -> FakeChannel:
        cid = f"UC{idx:022d}"
        newest = newest or datetime(2026, 11, 1, 16, tzinfo=timezone.utc)
        ch = FakeChannel(cid, f"@chan{idx}", f"Channel {idx}")
        ch.videos = [
            video_item(f"c{idx}v{i:05d}", cid, newest - timedelta(days=i * days_apart)) for i in range(n_videos)
        ]
        self.channels.append(ch)
        return ch

    def _all_videos(self) -> dict[str, dict]:
        return {v["id"]: v for ch in self.channels for v in ch.videos}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/feeds/videos.xml":
            xml = self.rss.get(request.url.params["channel_id"])
            return httpx.Response(200, text=xml) if xml else httpx.Response(404, text="not found")
        endpoint = path.rsplit("/", 1)[-1]
        self.calls.append(endpoint)
        assert request.headers.get("X-Goog-Api-Key") == "test-key"
        assert "key" not in request.url.params
        if self.fail_next.get(endpoint):
            return httpx.Response(self.fail_next[endpoint].pop(0), json={"error": {"errors": [{"reason": "backendError"}]}})
        if self.quota_after is not None and len(self.calls) > self.quota_after:
            return httpx.Response(403, json={"error": {"code": 403, "errors": [{"reason": "quotaExceeded"}]}})
        p = request.url.params
        if endpoint == "channels":
            match = [
                c for c in self.channels
                if c.handle.lower() == p.get("forHandle", "").lower() or c.channel_id == p.get("id")
            ]
            return httpx.Response(200, json={"items": [
                {"id": c.channel_id, "snippet": {"title": c.title},
                 "contentDetails": {"relatedPlaylists": {"uploads": c.uploads}}}
                for c in match
            ]})
        if endpoint == "playlistItems":
            ch = next(c for c in self.channels if c.uploads == p["playlistId"])
            if ch.channel_id in self.no_playlist:
                return httpx.Response(
                    404, json={"error": {"code": 404, "errors": [{"reason": "playlistNotFound"}]}}
                )
            size = int(p.get("maxResults", 5))
            start = int(p.get("pageToken", "0"))
            page = ch.videos[start : start + size]
            body: dict = {"items": [
                {"contentDetails": {"videoId": v["id"], "videoPublishedAt": v["snippet"]["publishedAt"]}}
                for v in page
            ]}
            if start + size < len(ch.videos):
                body["nextPageToken"] = str(start + size)
            return httpx.Response(200, json=body)
        if endpoint == "videos":
            if self.calls.count("videos") in self.bad_videos_calls:
                return httpx.Response(400, json={"error": {"code": 400, "errors": [{"reason": "badRequest"}]}})
            ids = p["id"].split(",")
            assert len(ids) <= 50
            known = self._all_videos()
            return httpx.Response(200, json={"items": [known[i] for i in ids if i in known and i not in self.hidden]})
        return httpx.Response(404)

    def http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def client(self, http: httpx.AsyncClient, budget: int = 10_000) -> YouTubeClient:
        async def no_sleep(_s: float) -> None:
            return None

        return YouTubeClient("test-key", http, UnitLedger(budget=budget), sleep=no_sleep)


def feed_xml(channel_id: str, video_ids: list[str]) -> str:
    entries = "".join(
        f"<entry><yt:videoId>{v}</yt:videoId><title>{v}</title>"
        f"<published>2026-11-06T17:00:00+00:00</published></entry>"
        for v in video_ids
    )
    return (
        '<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns="http://www.w3.org/2005/Atom">'
        f"<yt:channelId>{channel_id}</yt:channelId>{entries}</feed>"
    )
