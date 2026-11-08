"""YouTube account connection (M11, docs/decisions/0017 §1).

- `GET /youtube/connection`: status for the Analytics-page card. Never
  returns tokens.
- `POST /youtube/oauth/start`: the Google consent URL, with a signed state.
- `GET /youtube/oauth/callback`: Google redirects the browser here, with no
  app auth. The state proves the app started the consent. The endpoint
  always ends in a redirect to `/analytics?youtube=connected|error`.
- `DELETE /youtube/connection`: revoke at Google (best effort), then delete
  the row. Already-pulled analytics rows are kept.
"""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Literal
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth.security import get_current_user
from app.config import Settings, get_settings
from app.database import get_session_factory
from app.market.youtube_api import API_BASE, YouTubeApiError, google_get
from app.models.oauth import OAuthCredential
from app.models.user import User
from app.youtube_oauth.crypto import TokenDecryptError, cipher_from_settings
from app.youtube_oauth.google_oauth import (
    GoogleOAuthClient,
    OAuthError,
    authorization_url,
    missing_scopes,
)
from app.youtube_oauth.state import InvalidState, create_state, verify_state
from app.youtube_oauth.tokens import PROVIDER, get_credential, store_connection

log = logging.getLogger("app.youtube_oauth")

router = APIRouter(prefix="/youtube", tags=["youtube"])

ConnectionStatus = Literal["not_configured", "disconnected", "connected", "needs_reconsent"]


class ConnectionOut(BaseModel):
    status: ConnectionStatus
    channel_id: str | None = None
    channel_title: str | None = None
    scopes: list[str] = []
    connected_at: datetime | None = None
    last_refresh_at: datetime | None = None
    last_error: str | None = None


class StartOut(BaseModel):
    authorization_url: str


def get_oauth_settings() -> Settings:
    return get_settings()


async def get_google_http() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient() as http:
        yield http


def _not_configured() -> Response:
    return Response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content='{"detail":"YouTube OAuth is not configured (GOOGLE_OAUTH_CLIENT_ID/SECRET, OAUTH_TOKEN_ENCRYPTION_KEY)"}',
        media_type="application/json",
    )


@router.get("/connection", response_model=ConnectionOut)
async def connection_status(
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_oauth_settings),
    sf: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> ConnectionOut:
    if not settings.youtube_oauth_configured:
        return ConnectionOut(status="not_configured")
    row = await get_credential(sf)
    if row is None:
        return ConnectionOut(status="disconnected")
    return ConnectionOut(
        status="connected" if row.status == "active" else "needs_reconsent",
        channel_id=row.connected_channel_id,
        channel_title=row.connected_channel_title,
        scopes=list(row.scopes),
        connected_at=row.connected_at,
        last_refresh_at=row.last_refresh_at,
        last_error=row.last_error,
    )


@router.post("/oauth/start", response_model=StartOut)
async def oauth_start(
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_oauth_settings),
):
    if not settings.youtube_oauth_configured:
        return _not_configured()
    return StartOut(authorization_url=authorization_url(settings, create_state(user.id, settings)))


def _back_to_app(settings: Settings, outcome: str, reason: str | None = None) -> RedirectResponse:
    params = {"youtube": outcome} | ({"reason": reason} if reason else {})
    return RedirectResponse(
        f"{settings.frontend_base_url.rstrip('/')}/analytics?{urlencode(params)}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/oauth/callback", include_in_schema=False)
async def oauth_callback(
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    settings: Settings = Depends(get_oauth_settings),
    sf: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    http: httpx.AsyncClient = Depends(get_google_http),
) -> RedirectResponse:
    def fail(reason: str) -> RedirectResponse:
        log.warning("youtube oauth: consent failed (%s)", reason)
        return _back_to_app(settings, "error", reason)

    if not settings.youtube_oauth_configured:
        return fail("not_configured")
    try:
        verify_state(state, settings)
    except InvalidState:
        return fail("state")
    if error:
        # e.g. `access_denied` when the owner cancels on Google's screen.
        return fail(error)
    if not code:
        return fail("no_code")

    oauth = GoogleOAuthClient.from_settings(http, settings)
    try:
        token = await oauth.exchange_code(code)
    except OAuthError as exc:
        log.warning("youtube oauth: code exchange failed: %s", exc)
        return fail("exchange_failed")
    if token.refresh_token is None:
        return fail("no_refresh_token")
    if missing_scopes(token.scopes):
        # The consent screen lets the owner untick scopes.
        return fail("scopes")
    try:
        data = await google_get(
            http,
            f"{API_BASE}/channels",
            {"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {token.access_token}"},
            label="channels",
        )
    except YouTubeApiError as exc:
        log.warning("youtube oauth: channels.list mine=true failed: %s", exc)
        return fail("channel_lookup_failed")
    items = data.get("items") or []
    if not items:
        return fail("no_channel")
    channel = items[0]
    await store_connection(
        sf,
        cipher_from_settings(settings),
        token,
        channel_id=channel["id"],
        channel_title=channel.get("snippet", {}).get("title", ""),
    )
    log.info("youtube oauth: connected channel %s", channel["id"])
    return _back_to_app(settings, "connected")


@router.delete("/connection", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect(
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_oauth_settings),
    sf: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    http: httpx.AsyncClient = Depends(get_google_http),
) -> Response:
    row = await get_credential(sf)
    if row is not None and settings.youtube_oauth_configured:
        try:
            refresh_token = cipher_from_settings(settings).decrypt(row.refresh_token_enc)
        except TokenDecryptError:
            refresh_token = None
        if refresh_token and not await GoogleOAuthClient.from_settings(http, settings).revoke(refresh_token):
            log.warning("youtube oauth: revoke at Google not confirmed; the local credential is deleted anyway")
    async with sf() as db:
        await db.execute(delete(OAuthCredential).where(OAuthCredential.provider == PROVIDER))
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
