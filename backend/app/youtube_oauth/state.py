"""The OAuth `state` parameter (0017 §1): a short-lived JWT signed with
`JWT_SECRET`, so the unauthenticated callback can tell that Google is
returning a consent the app itself started, for a known app user.
"""

import secrets
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.config import Settings

PURPOSE = "youtube_oauth"
STATE_TTL = timedelta(minutes=10)


class InvalidState(Exception):
    pass


def create_state(user_id: int, settings: Settings, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "purpose": PURPOSE,
        "nonce": secrets.token_urlsafe(16),
        "iat": int(now.timestamp()),
        "exp": int((now + STATE_TTL).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_state(state: str | None, settings: Settings) -> int:
    """The app user id the consent was started for, or `InvalidState`.

    Expiry is checked by `jose` against the wall clock.
    """
    if not state:
        raise InvalidState("missing state")
    try:
        payload = jwt.decode(state, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise InvalidState(f"bad state: {exc}") from exc
    if payload.get("purpose") != PURPOSE or not payload.get("nonce"):
        raise InvalidState("state is not a YouTube OAuth state")
    try:
        return int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidState("state has no user") from exc
