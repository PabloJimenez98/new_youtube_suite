"""In-memory fake of Google's OAuth token/revoke endpoints, the Data API
calls used with the bearer token, and Analytics `reports.query`, served
through `httpx.MockTransport` (M11 tests, docs/decisions/0017)."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
from cryptography.fernet import Fernet

from app.config import Settings, get_settings
from app.youtube_oauth.google_oauth import SCOPES

CHANNEL_ID = "UC" + "t" * 22
FERNET_KEY = Fernet.generate_key().decode()


def oauth_settings(**overrides) -> Settings:
    """Test settings with a (fake) OAuth client and a fresh Fernet key,
    independent of whatever backend/.env holds."""
    values = {
        "google_oauth_client_id": "client-123.apps.googleusercontent.com",
        "google_oauth_client_secret": "client-secret",
        "oauth_token_encryption_key": FERNET_KEY,
        "google_oauth_redirect_uri": "http://localhost:4200/youtube/oauth/callback",
        "frontend_base_url": "http://localhost:4200",
    }
    values.update(overrides)
    base = get_settings().model_dump()
    base.update(values)
    return Settings(**base)  # init kwargs beat env/.env


@dataclass
class FakeVideo:
    vid: str
    published: datetime
    title: str = ""
    privacy: str = "public"
    # day -> views; other metrics derived from views
    daily_views: dict[date, int] = field(default_factory=dict)
    retention_points: int = 0  # 0 = no retention data (no views)


@dataclass
class FakeGoogle:
    channel_id: str = CHANNEL_ID
    channel_title: str = "Test channel"
    has_channel: bool = True
    videos: list[FakeVideo] = field(default_factory=list)  # newest first
    # --- token endpoint ---
    valid_codes: set[str] = field(default_factory=lambda: {"good-code"})
    refresh_tokens: set[str] = field(default_factory=lambda: {"rt-1"})
    issue_refresh_on_code: bool = True
    granted_scopes: tuple[str, ...] = SCOPES
    rotate_refresh: bool = False
    refresh_error: str | None = None  # e.g. "invalid_grant", "invalid_client"
    token_5xx: int = 0  # answer this many token calls with 503 first
    token_calls: list[dict] = field(default_factory=list)
    revoked: list[str] = field(default_factory=list)
    expires_in: int = 3600
    # --- APIs ---
    access_tokens: set[str] = field(default_factory=set)
    api_401_always: bool = False
    api_calls: list[tuple[str, dict]] = field(default_factory=list)
    quota_after: int | None = None  # 403 quotaExceeded once this many analytics calls succeeded
    page_size_override: int | None = None
    _n: int = 0

    # ---- helpers ----
    def issue_access(self) -> str:
        self._n += 1
        token = f"at-{self._n}"
        self.access_tokens.add(token)
        return token

    def add_videos(self, n: int, *, newest: datetime | None = None, days_apart: int = 30) -> None:
        newest = newest or datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
        for i in range(n):
            self.videos.append(FakeVideo(f"own{i:04d}", newest - timedelta(days=i * days_apart), f"Own video {i}"))

    @property
    def refresh_count(self) -> int:
        return sum(1 for c in self.token_calls if c.get("grant_type") == "refresh_token")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ---- routing ----
    def handle(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.host == "oauth2.googleapis.com" and url.path == "/token":
            return self._token(request)
        if url.host == "oauth2.googleapis.com" and url.path == "/revoke":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.revoked.append(form.get("token", ""))
            return httpx.Response(200, json={})
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ")
        params = dict(url.params)
        self.api_calls.append((url.path, params))
        if self.api_401_always or token not in self.access_tokens:
            return httpx.Response(401, json={"error": {"code": 401, "message": "Invalid Credentials", "errors": [{"reason": "authError"}]}})
        if url.path == "/youtube/v3/channels":
            return self._channels(params)
        if url.path == "/youtube/v3/playlistItems":
            return self._playlist(params)
        if url.path == "/v2/reports":
            return self._reports(params)
        return httpx.Response(404, json={"error": {"message": f"unknown {url.path}"}})

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.token_calls.append(form)
        if self.token_5xx > 0:
            self.token_5xx -= 1
            return httpx.Response(503, json={"error": "backend_error"})
        if form.get("client_secret") != "client-secret":
            return httpx.Response(401, json={"error": "invalid_client", "error_description": "Unauthorized"})
        if form["grant_type"] == "authorization_code":
            if form.get("code") not in self.valid_codes:
                return httpx.Response(400, json={"error": "invalid_grant", "error_description": "Bad Request"})
            body = {"access_token": self.issue_access(), "expires_in": self.expires_in, "scope": " ".join(self.granted_scopes), "token_type": "Bearer"}
            if self.issue_refresh_on_code:
                body["refresh_token"] = "rt-1"
                self.refresh_tokens.add("rt-1")
            return httpx.Response(200, json=body)
        # refresh_token grant
        if self.refresh_error:
            return httpx.Response(400 if self.refresh_error == "invalid_grant" else 401, json={"error": self.refresh_error, "error_description": "Token has been expired or revoked."})
        if form.get("refresh_token") not in self.refresh_tokens:
            return httpx.Response(400, json={"error": "invalid_grant", "error_description": "Token has been expired or revoked."})
        body = {"access_token": self.issue_access(), "expires_in": self.expires_in, "scope": " ".join(self.granted_scopes), "token_type": "Bearer"}
        if self.rotate_refresh:
            new = f"rt-{self._n + 100}"
            self.refresh_tokens.add(new)
            body["refresh_token"] = new
        return httpx.Response(200, json=body)

    def _channels(self, params: dict) -> httpx.Response:
        assert params.get("mine") == "true"
        if not self.has_channel:
            return httpx.Response(200, json={"items": []})
        return httpx.Response(200, json={"items": [{
            "id": self.channel_id,
            "snippet": {"title": self.channel_title},
            "contentDetails": {"relatedPlaylists": {"uploads": "UU" + self.channel_id[2:]}},
        }]})

    def _playlist(self, params: dict) -> httpx.Response:
        start = int(params.get("pageToken") or 0)
        size = int(params.get("maxResults", 50))
        page = self.videos[start : start + size]
        body: dict = {"items": [{
            "snippet": {"title": v.title, "publishedAt": v.published.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "contentDetails": {"videoId": v.vid, "videoPublishedAt": v.published.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "status": {"privacyStatus": v.privacy},
        } for v in page]}
        if start + size < len(self.videos):
            body["nextPageToken"] = str(start + size)
        return httpx.Response(200, json=body)

    def _reports(self, params: dict) -> httpx.Response:
        n_reports = sum(1 for p, _ in self.api_calls if p == "/v2/reports")
        if self.quota_after is not None and n_reports > self.quota_after:
            return httpx.Response(403, json={"error": {"message": "quota", "errors": [{"reason": "quotaExceeded"}]}})
        assert params["ids"] == "channel==MINE"
        start, end = date.fromisoformat(params["startDate"]), date.fromisoformat(params["endDate"])
        ids = params["filters"].removeprefix("video==").split(",")
        by_id = {v.vid: v for v in self.videos}
        metrics = params["metrics"].split(",")
        if params["dimensions"] == "day,video":
            headers = ["day", "video", *metrics]
            rows = []
            for vid in ids:
                v = by_id.get(vid)
                if v is None:
                    continue
                for day, views in sorted(v.daily_views.items()):
                    if start <= day <= end:
                        rows.append([day.isoformat(), vid, *(_metric(m, views) for m in metrics)])
            rows.sort(key=lambda r: r[0])
        else:
            assert params["dimensions"] == "elapsedVideoTimeRatio"
            assert len(ids) == 1, "retention takes exactly one video"
            headers = ["elapsedVideoTimeRatio", *metrics]
            v = by_id.get(ids[0])
            n = v.retention_points if v else 0
            rows = [[round((i + 1) / n, 2), *(_retention(m, i, n) for m in metrics)] for i in range(n)]
        size = self.page_size_override or int(params["maxResults"])
        first = int(params["startIndex"]) - 1
        body: dict = {"kind": "youtubeAnalytics#resultTable", "columnHeaders": [{"name": h} for h in headers]}
        page = rows[first : first + size]
        if page:
            body["rows"] = page
        return httpx.Response(200, json=body)


def _metric(name: str, views: int):
    return {
        "views": views,
        "estimatedMinutesWatched": views * 2,
        "averageViewDuration": 120,
        "averageViewPercentage": 41.5,
        "likes": views // 10,
        "comments": views // 50,
        "shares": views // 100,
        "subscribersGained": 1,
        "subscribersLost": 0,
    }[name]


def _retention(name: str, i: int, n: int):
    return {
        "audienceWatchRatio": round(1.0 - i / n / 2, 4),
        "relativeRetentionPerformance": 0.5,
        "startedWatching": 3,
        "stoppedWatching": 2,
        "totalSegmentImpressions": 40,
    }[name]
