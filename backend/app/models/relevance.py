from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class RelevanceScore(Base):
    """One window's relevance (spec §4), with its two components kept apart
    for M15's pseudo-labels (docs/decisions/0012 §5)."""

    __tablename__ = "relevance_scores"
    __table_args__ = (
        UniqueConstraint("upload_id", "window_index", name="uq_relevance_scores_upload_window"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    upload_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    window_index: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    window_end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    heuristic_score: Mapped[float] = mapped_column(Float, nullable=False)
    # NULL when the window's LLM batch failed every attempt.
    llm_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("model_versions.id"), nullable=False
    )


class RelevancePseudoLabel(Base):
    """A teacher's per-window components, kept as M15's training targets
    (docs/decisions/0024 §2).

    Separate from `relevance_scores`: there, a re-run replaces the upload's
    rows, and once a trained model is live its own outputs would overwrite
    the targets it learned from. Only teacher versions (no `artifact_path`)
    write here; several teachers per upload can coexist.
    """

    __tablename__ = "relevance_pseudo_labels"
    __table_args__ = (
        UniqueConstraint(
            "upload_id", "window_index", "teacher_version_id", name="uq_relevance_pseudo_labels_upload_window_teacher"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    upload_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    window_index: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    window_end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    heuristic_score: Mapped[float] = mapped_column(Float, nullable=False)
    # NULL when the window's LLM batch failed every attempt.
    llm_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    teacher_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("model_versions.id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
