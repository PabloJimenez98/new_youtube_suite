"""M11 connection endpoints (docs/decisions/0017 §1): status payloads,
consent start, the callback's happy path and failure redirects, and
disconnect. Google is `FakeGoogle`; settings are injected, so this never
depends on what backend/.env holds."""

from collections.abc import AsyncGenerator
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import func, select

from app.main import app
from app.models.oauth import OAuthCredential
from app.routers.youtube import get_google_http, get_oauth_settings
from app.youtube_oauth.crypto import TokenCipher
from app.youtube_oauth.google_oauth import SCOPES
from app.youtube_oauth.state import create_state
from app.youtube_oauth.tokens import get_credential
from tests.google_fake import CHANNEL_ID, FERNET_KEY, FakeGoogle, oauth_settings


@pytest.fixture
def google() -> FakeGoogle:
    return FakeGoogle()


@pytest.fixture
def settings():
    return oauth_settings()


@pytest_asyncio.fixture(autouse=True)
async def _wire(google: FakeGoogle, settings) -> AsyncGenerator[None, None]:
    async def http() -> AsyncGenerator[httpx.AsyncClient, None]:
        async with httpx.AsyncClient(transport=google.transport()) as c:
            yield c

    app.dependency_overrides[get_google_http] = http
    app.dependency_overrides[get_oauth_settings] = lambda: settings
    yield
    app.dependency_overrides.pop(get_google_http, None)
    app.dependency_overrides.pop(get_oauth_settings, None)


def _state(test_user, settings) -> str:
    return create_state(test_user.id, settings)


async def _callback(client, **params) -> httpx.Response:
    return await client.get("/youtube/oauth/callback", params=params, follow_redirects=False)


def _outcome(response: httpx.Response) -> dict[str, str]:
    assert response.status_code == 303
    loc = urlparse(response.headers["location"])
    assert loc.netloc == "localhost:4200" and loc.path == "/analytics"
    return {k: v[0] for k, v in parse_qs(loc.query).items()}


async def _rows(session_factory) -> int:
    async with session_factory() as db:
        return await db.scalar(select(func.count()).select_from(OAuthCredential))


# ---- auth & status -------------------------------------------------------------


async def test_endpoints_require_auth_except_the_callback(client) -> None:
    assert (await client.get("/youtube/connection")).status_code == 401
    assert (await client.post("/youtube/oauth/start")).status_code == 401
    assert (await client.delete("/youtube/connection")).status_code == 401
    assert (await _callback(client)).status_code == 303  # redirects with reason=state


async def test_status_not_configured(client, auth_headers) -> None:
    app.dependency_overrides[get_oauth_settings] = lambda: oauth_settings(oauth_token_encryption_key=None)
    body = (await client.get("/youtube/connection", headers=auth_headers)).json()
    assert body["status"] == "not_configured"
    assert (await client.post("/youtube/oauth/start", headers=auth_headers)).status_code == 503


async def test_status_disconnected(client, auth_headers) -> None:
    assert (await client.get("/youtube/connection", headers=auth_headers)).json()["status"] == "disconnected"


async def test_start_returns_google_consent_url(client, auth_headers, test_user, settings) -> None:
    body = (await client.post("/youtube/oauth/start", headers=auth_headers)).json()
    url = urlparse(body["authorization_url"])
    q = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert url.netloc == "accounts.google.com"
    assert q["client_id"] == settings.google_oauth_client_id
    assert q["redirect_uri"] == "http://localhost:4200/youtube/oauth/callback"
    assert q["access_type"] == "offline" and q["prompt"] == "consent" and q["response_type"] == "code"
    assert set(q["scope"].split()) == set(SCOPES)
    from app.youtube_oauth.state import verify_state

    assert verify_state(q["state"], settings) == test_user.id


# ---- callback ----------------------------------------------------------------


async def test_callback_happy_path_connects_and_status_shows_it(
    client, auth_headers, test_user, settings, session_factory, google
) -> None:
    out = _outcome(await _callback(client, code="good-code", state=_state(test_user, settings)))
    assert out == {"youtube": "connected"}
    row = await get_credential(session_factory)
    assert row.connected_channel_id == CHANNEL_ID and row.status == "active"
    assert set(row.scopes) == set(SCOPES)
    assert TokenCipher(FERNET_KEY).decrypt(row.refresh_token_enc) == "rt-1"

    body = (await client.get("/youtube/connection", headers=auth_headers)).json()
    assert body["status"] == "connected" and body["channel_id"] == CHANNEL_ID
    assert body["channel_title"] == "Test channel"
    # Never any token material in the payload.
    flat = str(body)
    assert "rt-1" not in flat and "at-" not in flat and "token" not in "".join(body)


@pytest.mark.parametrize(
    ("params", "setup", "reason"),
    [
        ({"code": "good-code", "state": "garbage"}, None, "state"),
        ({"code": "good-code"}, None, "state"),
        ({"error": "access_denied"}, None, "access_denied"),
        ({}, None, "no_code"),
        ({"code": "bad-code"}, None, "exchange_failed"),
        ({"code": "good-code"}, "no_refresh", "no_refresh_token"),
        ({"code": "good-code"}, "one_scope", "scopes"),
        ({"code": "good-code"}, "no_channel", "no_channel"),
    ],
)
async def test_callback_failures_redirect_with_a_reason_and_write_nothing(
    client, test_user, settings, session_factory, google, params, setup, reason
) -> None:
    if setup == "no_refresh":
        google.issue_refresh_on_code = False
    elif setup == "one_scope":
        google.granted_scopes = (SCOPES[0],)
    elif setup == "no_channel":
        google.has_channel = False
    if "state" not in params and reason != "state":
        params = {**params, "state": _state(test_user, settings)}
    out = _outcome(await _callback(client, **params))
    assert out == {"youtube": "error", "reason": reason}
    assert await _rows(session_factory) == 0


async def test_reconsent_overwrites_and_reactivates(
    client, auth_headers, test_user, settings, session_factory
) -> None:
    await _callback(client, code="good-code", state=_state(test_user, settings))
    async with session_factory() as db:
        row = (await db.execute(select(OAuthCredential))).scalar_one()
        row.status, row.last_error = "needs_reconsent", "invalid_grant: Token has been expired or revoked."
        await db.commit()
    body = (await client.get("/youtube/connection", headers=auth_headers)).json()
    assert body["status"] == "needs_reconsent" and "invalid_grant" in body["last_error"]

    assert _outcome(await _callback(client, code="good-code", state=_state(test_user, settings)))["youtube"] == "connected"
    assert await _rows(session_factory) == 1
    body = (await client.get("/youtube/connection", headers=auth_headers)).json()
    assert body["status"] == "connected" and body["last_error"] is None


# ---- disconnect --------------------------------------------------------------


async def test_disconnect_revokes_and_deletes(client, auth_headers, test_user, settings, session_factory, google) -> None:
    await _callback(client, code="good-code", state=_state(test_user, settings))
    response = await client.delete("/youtube/connection", headers=auth_headers)
    assert response.status_code == 204
    assert google.revoked == ["rt-1"]
    assert await _rows(session_factory) == 0
    assert (await client.get("/youtube/connection", headers=auth_headers)).json()["status"] == "disconnected"


async def test_disconnect_when_not_connected_is_a_noop(client, auth_headers, google) -> None:
    assert (await client.delete("/youtube/connection", headers=auth_headers)).status_code == 204
    assert google.revoked == []
