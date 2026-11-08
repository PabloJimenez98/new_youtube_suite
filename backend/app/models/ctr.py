from datetime import datetime

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

CANDIDATE_SOURCES = ("manual", "show_notes")


class CtrPrediction(Base):
    """A title/thumbnail candidate and its latest heuristic score (spec §4.1
    `ctr_predictions`, docs/decisions/0019 §3).

    - **Candidate first:** the row exists before it is scored.
    - **Edits:** changing the title or the thumbnail clears
      `predicted_score`, `features` and `scored_at`, so a stale score is
      never shown.
    - **The score:** relative within one upload, 0-100. It is not a CTR %.
    - **`partial`:** the score came from the title alone (no thumbnail).
    - **`features`:** the raw feature values and sub-scores, for the
      breakdown and for later calibration (M15).
    """

    __tablename__ = "ctr_predictions"
    __table_args__ = (CheckConstraint(f"source IN {CANDIDATE_SOURCES!r}", name="ctr_predictions_source_check"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # Nullable as in the spec (standalone candidates); the M13 API always sets it.
    upload_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("uploads.id", ondelete="CASCADE"), nullable=True, index=True
    )
    candidate_title: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    predicted_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    partial: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    model_version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="manual")
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
