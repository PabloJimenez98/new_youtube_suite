from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class HookScore(Base):
    """The latest heuristic hook-strength score of one clip (spec §4.1
    `hook_scores`, docs/decisions/0020 §4).

    - **One row per clip:** a re-run overwrites it. A shorts re-run replaces
      the clips, so their scores cascade away.
    - **The score:** relative among one upload's clips, 0-100. It is not a
      retention %.
    - **`feature_vector`:** every feature, sub-score and weight, for the
      breakdown and for M15's LightGBM.
    """

    __tablename__ = "hook_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    clip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    feature_vector: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
