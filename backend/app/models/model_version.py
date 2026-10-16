from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Index, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

MODEL_NAMES = ("relevance", "ctr", "hook")


class ModelVersion(Base):
    """A version of one of the three custom models (spec §4).

    Created in M7 for relevance v0 (docs/decisions/0012 §5). At most one
    live version per `model_name`, enforced by a partial unique index;
    switching the live version is M15's.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_name", "version_label", name="uq_model_versions_name_label"),
        Index("uq_model_versions_one_live", "model_name", unique=True, postgresql_where=text("is_live")),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    version_label: Mapped[str] = mapped_column(Text, nullable=False)
    architecture_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    training_data_reference: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_live: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
