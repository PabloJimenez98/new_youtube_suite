from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Text, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

CREDENTIAL_STATUSES = ("active", "needs_reconsent")


class OAuthCredential(Base):
    """The single platform-level connected account (spec §4.2, v1-scope:
    not multi-tenant), plus the M11 bookkeeping columns
    (docs/decisions/0017 §2). `provider` is UNIQUE, so there is one row
    per provider -- one, `youtube`, in v1.

    Tokens are Fernet ciphertext (`app.youtube_oauth.crypto`), never
    plaintext. `status` is how a dead refresh token is made loud rather
    than silent (0017 §3): the cron stops and the frontend card asks for
    re-consent with `last_error`.
    """

    __tablename__ = "oauth_credentials"
    __table_args__ = (
        CheckConstraint(f"status IN {CREDENTIAL_STATUSES!r}", name="oauth_credentials_status_check"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    access_token_enc: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token_enc: Mapped[str] = mapped_column(Text, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    connected_channel_id: Mapped[str] = mapped_column(Text, nullable=False)
    connected_channel_title: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    token_expiry: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    connected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_refresh_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
