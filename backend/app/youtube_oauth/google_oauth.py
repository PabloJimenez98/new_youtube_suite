"""Hand-written Google OAuth 2.0 web-server-flow client (0017 §1): the
authorization URL, plus three POSTs (code exchange, refresh, revoke).

Error classes map Google's token-endpoint errors onto what the caller
must do (0017 §3):

- `GrantError` (`invalid_grant`): the refresh token or code is dead --
  revoked, expired (7 days in "Testing"), password changed. Only a new
  consent fixes it.
- `OAuthConfigError` (`invalid_client`, `unauthorized_client`): the client
  id/secret in `.env` is wrong. Re-consent would not help.
- `OAuthTransientError`: network errors / 5xx after 3 retries. Nothing is
  known about the token; the stored status must not change.
- `OAuthError`: anything else (e.g. `invalid_request`).
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config import Settings

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
# Both read-only. `youtube.readonly` is required by `reports.query` (API
# reference, 2026) and lists the channel's own unlisted/private uploads.
SCOPES = (
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
)
MAX_RETRIES = 3
_CONFIG_ERRORS = {"invalid_client", "unauthorized_client"}


class OAuthError(Exception):
    def __init__(self, message: str, *, error: str | None = None) -> None:
        super().__init__(message)
        self.error = error


class GrantError(OAuthError):
    pass


class OAuthConfigError(OAuthError):
    pass


class OAuthTransientError(OAuthError):
    pass


@dataclass(frozen=True)
class TokenResponse:
    access_token: str
    expires_in: int
    refresh_token: str | None  # only on consent, or when Google rotates it
    scopes: tuple[str, ...]


def authorization_url(settings: Settings, state: str) -> str:
    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.google_oauth_redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        # offline + consent: a refresh token comes back every time, also on
        # a re-consent (Google otherwise only returns it the first time).
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def missing_scopes(granted: tuple[str, ...]) -> list[str]:
    return [s for s in SCOPES if s not in granted]


class GoogleOAuthClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._sleep = sleep

    @classmethod
    def from_settings(cls, http: httpx.AsyncClient, settings: Settings, **kwargs: Any) -> "GoogleOAuthClient":
        if settings.google_oauth_client_id is None or settings.google_oauth_client_secret is None:
            raise OAuthConfigError("GOOGLE_OAUTH_CLIENT_ID/SECRET are not set")
        return cls(
            http,
            client_id=settings.google_oauth_client_id,
            client_secret=settings.google_oauth_client_secret.get_secret_value(),
            redirect_uri=settings.google_oauth_redirect_uri,
            **kwargs,
        )

    async def exchange_code(self, code: str) -> TokenResponse:
        return await self._token_request(
            {"grant_type": "authorization_code", "code": code, "redirect_uri": self._redirect_uri}
        )

    async def refresh(self, refresh_token: str) -> TokenResponse:
        return await self._token_request({"grant_type": "refresh_token", "refresh_token": refresh_token})

    async def revoke(self, token: str) -> bool:
        """Best effort: True if Google confirmed. Never raises."""
        try:
            response = await self._http.post(REVOKE_URL, data={"token": token}, timeout=30)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _token_request(self, data: dict[str, str]) -> TokenResponse:
        body = {**data, "client_id": self._client_id, "client_secret": self._client_secret}
        delay = 1.0
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._http.post(TOKEN_URL, data=body, timeout=30)
            except httpx.HTTPError as exc:
                if attempt == MAX_RETRIES:
                    raise OAuthTransientError(f"token endpoint: {exc}") from exc
            else:
                if response.status_code == 200:
                    return _parse_token(response.json())
                if response.status_code < 500 and response.status_code != 429:
                    raise _token_error(response)
                if attempt == MAX_RETRIES:
                    raise OAuthTransientError(f"token endpoint: HTTP {response.status_code} after retries")
            await self._sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")


def _parse_token(payload: dict[str, Any]) -> TokenResponse:
    try:
        return TokenResponse(
            access_token=payload["access_token"],
            expires_in=int(payload.get("expires_in", 3600)),
            refresh_token=payload.get("refresh_token"),
            scopes=tuple(str(payload.get("scope", "")).split()),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OAuthError("token endpoint returned an unexpected payload") from exc


def _token_error(response: httpx.Response) -> OAuthError:
    try:
        payload = response.json()
        error = payload.get("error")
        description = payload.get("error_description") or ""
    except (ValueError, AttributeError):
        error, description = None, ""
    message = f"{error or f'HTTP {response.status_code}'}: {description}".rstrip(": ")
    if error == "invalid_grant":
        return GrantError(message, error=error)
    if error in _CONFIG_ERRORS:
        return OAuthConfigError(message, error=error)
    return OAuthError(message, error=error)
