from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.user import User

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# auto_error=False: this dependency is shared by get_current_user (header
# only, 401s itself if missing) and get_current_user_for_media (header OR
# query-param fallback), so it must not raise before the media variant
# gets a chance to check the query param.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


async def _user_from_token(token: str, db: AsyncSession) -> User:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise CREDENTIALS_ERROR
    except JWTError as exc:
        raise CREDENTIALS_ERROR from exc

    user = await db.get(User, int(user_id))
    if user is None:
        raise CREDENTIALS_ERROR
    return user


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if token is None:
        raise CREDENTIALS_ERROR
    return await _user_from_token(token, db)


async def get_current_user_for_media(
    header_token: str | None = Depends(oauth2_scheme),
    query_token: str | None = Query(default=None, alias="token"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Same auth as `get_current_user`, but also accepts the token as a
    `?token=` query param.

    Needed for `<video src>`-style requests: the browser fetches those
    directly, with no way for our code to attach a custom Authorization
    header. See docs/decisions/0005-video-preview-streaming.md for the
    trade-off this accepts (a token in a URL, logs, browser history).
    Only used for the media-streaming endpoint -- every other route
    still requires the header.
    """
    token = header_token or query_token
    if token is None:
        raise CREDENTIALS_ERROR
    return await _user_from_token(token, db)


async def authenticate_user(
    db: AsyncSession, email: str, password: str
) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user
