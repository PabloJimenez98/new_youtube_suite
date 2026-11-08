"""M11 token refresh and expiry (docs/decisions/0017 §3): the plan's
flagged silent-failure source. No network: `FakeGoogle` behind
`httpx.MockTransport`, and an injectable clock instead of sleeping."""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select, text

from app.market.youtube_api import Unauthorized
from app.models.oauth import OAuthCredential
from app.youtube_oauth.crypto import TokenCipher, TokenDecryptError
from app.youtube_oauth.google_oauth import (
    GoogleOAuthClient,
    OAuthConfigError,
    OAuthTransientError,
    TokenResponse,
)
from app.youtube_oauth.state import InvalidState, create_state, verify_state
from app.youtube_oauth.tokens import (
    NotConnected,
    ReconsentRequired,
    TokenProvider,
    get_credential,
    store_connection,
)
from tests.google_fake import CHANNEL_ID, FERNET_KEY, FakeGoogle, oauth_settings

T0 = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)


async def _no_sleep(_s: float) -> None:
    return None


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _oauth(google: FakeGoogle, http: httpx.AsyncClient, secret: str = "client-secret") -> GoogleOAuthClient:
    return GoogleOAuthClient(
        http, client_id="client-123", client_secret=secret, redirect_uri="http://x/cb", sleep=_no_sleep
    )


async def _connect(session_factory, google: FakeGoogle, cipher: TokenCipher, *, now=T0) -> str:
    access = google.issue_access()
    await store_connection(
        session_factory,
        cipher,
        TokenResponse(access, 3600, "rt-1", tuple(google.granted_scopes)),
        channel_id=CHANNEL_ID,
        channel_title="Test channel",
        now=now,
    )
    return access


@pytest.fixture
def cipher() -> TokenCipher:
    return TokenCipher(FERNET_KEY)


# ---- crypto ------------------------------------------------------------------


def test_cipher_roundtrip_and_wrong_key() -> None:
    a = TokenCipher(FERNET_KEY)
    blob = a.encrypt("ya29.secret")
    assert blob != "ya29.secret" and "ya29" not in blob
    assert a.decrypt(blob) == "ya29.secret"
    with pytest.raises(TokenDecryptError):
        TokenCipher(Fernet.generate_key().decode()).decrypt(blob)


def test_invalid_key_is_reported_without_echoing_it() -> None:
    with pytest.raises(ValueError) as exc:
        TokenCipher("not-a-key")
    assert "not-a-key" not in str(exc.value)


async def test_tokens_are_not_stored_in_plaintext(session_factory, cipher) -> None:
    google = FakeGoogle()
    access = await _connect(session_factory, google, cipher)
    async with session_factory() as db:
        raw = (await db.execute(text("SELECT access_token_enc, refresh_token_enc FROM oauth_credentials"))).one()
    assert access not in raw[0] and "rt-1" not in raw[1]
    assert cipher.decrypt(raw[0]) == access and cipher.decrypt(raw[1]) == "rt-1"


# ---- state -------------------------------------------------------------------


def test_state_roundtrip_and_rejections() -> None:
    s = oauth_settings()
    assert verify_state(create_state(42, s), s) == 42
    with pytest.raises(InvalidState):
        verify_state(None, s)
    with pytest.raises(InvalidState):  # expired (10-minute TTL)
        verify_state(create_state(42, s, now=datetime.now(timezone.utc) - timedelta(minutes=11)), s)
    with pytest.raises(InvalidState):  # tampered / signed with another secret
        verify_state(create_state(42, oauth_settings(jwt_secret="other")), s)
    from jose import jwt

    login_token = jwt.encode({"sub": "42", "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp())}, s.jwt_secret, algorithm=s.jwt_algorithm)
    with pytest.raises(InvalidState):  # an app access token is not a state
        verify_state(login_token, s)


# ---- refresh -----------------------------------------------------------------


async def test_valid_token_is_returned_without_any_http_call(session_factory, cipher) -> None:
    google = FakeGoogle()
    access = await _connect(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0 + timedelta(minutes=30)))
        assert await provider.get() == access
    assert google.token_calls == [] and provider.channel_id == CHANNEL_ID


async def test_simulated_expiry_refreshes_once_and_stores_the_new_token(session_factory, cipher) -> None:
    google = FakeGoogle()
    old = await _connect(session_factory, google, cipher)
    clock = Clock(T0 + timedelta(seconds=3600 - 119))  # inside the 120 s margin
    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=clock)
        new = await provider.get()
        assert new != old and google.refresh_count == 1
        assert await provider.get() == new  # cached in the row now
    assert google.refresh_count == 1
    row = await get_credential(session_factory)
    assert row.token_expiry == clock.now + timedelta(seconds=3600)
    assert row.last_refresh_at == clock.now and row.status == "active"
    assert cipher.decrypt(row.access_token_enc) == new


async def test_rotated_refresh_token_is_stored(session_factory, cipher) -> None:
    google = FakeGoogle(rotate_refresh=True)
    await _connect(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        await TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0 + timedelta(hours=2))).get()
    row = await get_credential(session_factory)
    assert cipher.decrypt(row.refresh_token_enc) != "rt-1"
    assert cipher.decrypt(row.refresh_token_enc) in google.refresh_tokens


async def test_invalid_grant_marks_needs_reconsent_loudly(session_factory, cipher, caplog) -> None:
    google = FakeGoogle(refresh_error="invalid_grant")
    await _connect(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0 + timedelta(hours=2)))
        with caplog.at_level("WARNING", logger="app.youtube_oauth"), pytest.raises(ReconsentRequired):
            await provider.get()
        # Stays dead: no further refresh attempts until a new consent.
        with pytest.raises(ReconsentRequired):
            await provider.get()
    assert google.refresh_count == 1
    row = await get_credential(session_factory)
    assert row.status == "needs_reconsent" and "invalid_grant" in row.last_error
    assert "re-consent required" in caplog.text


async def test_bad_client_secret_is_a_config_error_not_reconsent(session_factory, cipher) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http, secret="wrong"), cipher, clock=Clock(T0 + timedelta(hours=2)))
        with pytest.raises(OAuthConfigError):
            await provider.get()
    row = await get_credential(session_factory)
    assert row.status == "active" and row.last_error is None


async def test_transient_errors_retry_then_leave_status_unchanged(session_factory, cipher) -> None:
    google = FakeGoogle(token_5xx=10)
    await _connect(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0 + timedelta(hours=2)))
        with pytest.raises(OAuthTransientError):
            await provider.get()
    assert len(google.token_calls) == 4  # 1 + 3 retries
    assert (await get_credential(session_factory)).status == "active"


async def test_transient_error_then_success(session_factory, cipher) -> None:
    google = FakeGoogle(token_5xx=2)
    await _connect(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0 + timedelta(hours=2)))
        assert (await provider.get()).startswith("at-")


async def test_network_error_does_not_ask_for_reconsent(session_factory, cipher) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google, cipher)

    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0 + timedelta(hours=2)))
        with pytest.raises(OAuthTransientError):
            await provider.get()
    assert (await get_credential(session_factory)).status == "active"


async def test_undecryptable_token_marks_needs_reconsent(session_factory, cipher) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google, cipher)
    other = TokenCipher(Fernet.generate_key().decode())  # key rotated by mistake
    async with httpx.AsyncClient(transport=google.transport()) as http:
        with pytest.raises(ReconsentRequired):
            await TokenProvider(session_factory, _oauth(google, http), other, clock=Clock(T0)).get()
    row = await get_credential(session_factory)
    assert row.status == "needs_reconsent" and row.last_error == "token decryption failed"


async def test_not_connected(session_factory, cipher) -> None:
    async with httpx.AsyncClient(transport=FakeGoogle().transport()) as http:
        with pytest.raises(NotConnected):
            await TokenProvider(session_factory, _oauth(FakeGoogle(), http), cipher).get()


async def test_concurrent_callers_refresh_only_once(session_factory, cipher) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google, cipher)
    clock = Clock(T0 + timedelta(hours=2))
    async with httpx.AsyncClient(transport=google.transport()) as http:
        a = TokenProvider(session_factory, _oauth(google, http), cipher, clock=clock)
        b = TokenProvider(session_factory, _oauth(google, http), cipher, clock=clock)
        ta, tb = await asyncio.gather(a.get(), b.get())
    assert ta == tb and google.refresh_count == 1


async def test_reconsent_overwrites_the_row_and_reactivates(session_factory, cipher) -> None:
    google = FakeGoogle(refresh_error="invalid_grant")
    await _connect(session_factory, google, cipher)
    async with httpx.AsyncClient(transport=google.transport()) as http:
        with pytest.raises(ReconsentRequired):
            await TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0 + timedelta(hours=2))).get()
    await _connect(session_factory, google, cipher, now=T0 + timedelta(hours=3))
    async with session_factory() as db:
        rows = (await db.execute(select(OAuthCredential))).scalars().all()
    assert len(rows) == 1 and rows[0].status == "active" and rows[0].last_error is None


# ---- 401 handling ------------------------------------------------------------


async def test_api_401_forces_one_refresh_and_retries(session_factory, cipher) -> None:
    google = FakeGoogle()
    stale = await _connect(session_factory, google, cipher)
    google.access_tokens.discard(stale)  # Google stopped honouring it early
    seen: list[str] = []

    async def call(token: str) -> str:
        seen.append(token)
        if token not in google.access_tokens:
            raise Unauthorized("HTTP 401", status=401)
        return "ok"

    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0))
        assert await provider.call_with_token(call) == "ok"
    assert seen[0] == stale and seen[1] != stale and google.refresh_count == 1


async def test_second_401_after_refresh_marks_needs_reconsent(session_factory, cipher) -> None:
    google = FakeGoogle()
    await _connect(session_factory, google, cipher)

    async def call(_token: str) -> str:
        raise Unauthorized("HTTP 401", status=401)

    async with httpx.AsyncClient(transport=google.transport()) as http:
        provider = TokenProvider(session_factory, _oauth(google, http), cipher, clock=Clock(T0))
        with pytest.raises(ReconsentRequired):
            await provider.call_with_token(call)
    row = await get_credential(session_factory)
    assert row.status == "needs_reconsent" and "freshly refreshed" in row.last_error
