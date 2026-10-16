from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

ASPECT_RATIOS = ("9:16", "1:1", "4:5", "16:9")


class Clip(Base):
    """One rendered short (spec §4 `clips`, docs/decisions/0013 §6)."""

    __tablename__ = "clips"
    __table_args__ = (
        UniqueConstraint("upload_id", "rank", name="uq_clips_upload_rank"),
        CheckConstraint(f"aspect_ratio IN {ASPECT_RATIOS!r}", name="clips_aspect_ratio_check"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    upload_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    # Smoothed relevance peak height: what clips are ranked by.
    score: Mapped[float] = mapped_column(Float, nullable=False)
    peak_window_index: Mapped[int] = mapped_column(Integer, nullable=False)
    aspect_ratio: Mapped[str] = mapped_column(Text, nullable=False)
    caption_burned_in: Mapped[bool] = mapped_column(Boolean, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    # [{t, x, y, w, h}] in source pixels, t relative to the clip.
    crop_bounding_box_track: Mapped[list] = mapped_column(JSONB, nullable=False)
    # {source, output, shots: [{start, end, mode}]}
    framing: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # The relevance version whose scores picked this clip.
    model_version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("model_versions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
