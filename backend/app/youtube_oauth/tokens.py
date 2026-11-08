"""Access tokens for the connected account, and the status transitions
that keep a dead refresh token from failing silently (0017 §3).

`TokenProvider.get()` is the only way any code obtains an access token:

- **More than `REFRESH_MARGIN` of validity left:** returns the decrypted
  token, with no network call.
- **Otherwise:** refreshes it. The row is held `SELECT ... FOR UPDATE`
  across the refresh, so concurrent callers refresh once; the second
  one sees the new expiry and returns it.
- **`invalid_grant`, or a token the current key can't decrypt:** the row
  goes to `needs_reconsent` with `last_error`, a WARNING is logged and
  `ReconsentRequired` is raised.
- **Config errors and transient errors:** propagate. The status is left
  alone; only a dead grant asks for re-consent.

`call_with_token` wraps an API call: one forced refresh on a 401, and a
second 401 marks the row `needs_reconsent`.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market.youtube_api import Unauthorized
from app.models.oauth import OAuthCredential
from app.youtube_oauth.crypto import TokenCipher, TokenDecryptError
from app.youtube_oauth.google_oauth import GoogleOAuthClient, GrantError, TokenResponse

log = logging.getLogger("app.youtube_oauth")

PROVIDER = "youtube"
REFRESH_MARGIN = timedelta(seconds=120)
SessionFactory = async_sessionmaker[AsyncSession]
T = TypeVar("T")


class NotConnected(Exception):
    """No credential row: the owner has not connected a channel yet."""


class ReconsentRequired(Exception):
    """The stored grant is dead; only a new consent in the app fixes it."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def get_credential(sf: SessionFactory) -> OAuthCredential | None:
    async with sf() as db:
        return (await db.execute(select(OAuthCredential).where(OAuthCredential.provider == PROVIDER))).scalar_one_or_none()


async def store_connection(
    sf: SessionFactory,
    cipher: TokenCipher,
    token: TokenResponse,
    *,
    channel_id: str,
    channel_title: str,
    now: datetime | None = None,
) -> None:
    """Insert or overwrite the single row after a successful consent;
    resets `status` to `active` and clears `last_error`."""
    if token.refresh_token is None:
        raise ValueError("a consent must return a refresh token")
    now = now or utcnow()
    values = {
        "provider": PROVIDER,
        "access_token_enc": cipher.encrypt(token.access_token),
        "refresh_token_enc": cipher.encrypt(token.refresh_token),
        "scopes": list(token.scopes),
        "connected_channel_id": channel_id,
        "connected_channel_title": channel_title,
        "token_expiry": now + timedelta(seconds=token.expires_in),
        "status": "active",
        "last_error": None,
        "connected_at": now,
        "last_refresh_at": now,
        "updated_at": now,
    }
    async with sf() as db:
        stmt = insert(OAuthCredential).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[OAuthCredential.provider],
            set_={k: stmt.excluded[k] for k in values if k != "provider"},
        )
        await db.execute(stmt)
        await db.commit()


async def mark_reconsent(sf: SessionFactory, reason: str) -> None:
    async with sf() as db:
        row = (
            await db.execute(select(OAuthCredential).where(OAuthCredential.provider == PROVIDER).with_for_update())
        ).scalar_one_or_none()
        if row is not None:
            _set_reconsent(row, reason)
            await db.commit()


def _set_reconsent(row: OAuthCredential, reason: str) -> None:
    row.status = "needs_reconsent"
    row.last_error = reason
    log.warning("youtube oauth: re-consent required (%s); reconnect from the Analytics page", reason)


class TokenProvider:
    def __init__(
        self,
        sf: SessionFactory,
        oauth: GoogleOAuthClient,
        cipher: TokenCipher,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._sf = sf
        self._oauth = oauth
        self._cipher = cipher
        self._clock = clock
        self.channel_id: str | None = None
        self.refreshes = 0  # refresh calls made by this provider

    async def get(self, *, force: bool = False) -> str:
        async with self._sf() as db:
            row = (
                await db.execute(
                    select(OAuthCredential).where(OAuthCredential.provider == PROVIDER).with_for_update()
                )
            ).scalar_one_or_none()
            if row is None:
                raise NotConnected("no YouTube account connected")
            if row.status == "needs_reconsent":
                raise ReconsentRequired(row.last_error or "re-consent required")
            self.channel_id = row.connected_channel_id
            now = self._clock()
            try:
                if not force and row.token_expiry - now > REFRESH_MARGIN:
                    return self._cipher.decrypt(row.access_token_enc)
                refresh_token = self._cipher.decrypt(row.refresh_token_enc)
            except TokenDecryptError as exc:
                _set_reconsent(row, str(exc))
                await db.commit()
                raise ReconsentRequired(str(exc)) from exc
            try:
                token = await self._oauth.refresh(refresh_token)
            except GrantError as exc:
                _set_reconsent(row, str(exc))
                await db.commit()
                raise ReconsentRequired(str(exc)) from exc
            self.refreshes += 1
            row.access_token_enc = self._cipher.encrypt(token.access_token)
            if token.refresh_token:  # Google rotated it
                row.refresh_token_enc = self._cipher.encrypt(token.refresh_token)
            row.token_expiry = now + timedelta(seconds=token.expires_in)
            row.last_refresh_at = now
            row.last_error = None
            await db.commit()
            log.info("youtube oauth: access token refreshed")
            return token.access_token

    async def call_with_token(self, fn: Callable[[str], Awaitable[T]]) -> T:
        """Run `fn(access_token)`; on a 401, refresh once and retry. A second
        401 means Google no longer honours the grant -> `needs_reconsent`."""
        try:
            return await fn(await self.get())
        except Unauthorized:
            pass
        try:
            return await fn(await self.get(force=True))
        except Unauthorized as exc:
            reason = f"API rejected a freshly refreshed token ({exc})"
            await mark_reconsent(self._sf, reason)
            raise ReconsentRequired(reason) from exc
