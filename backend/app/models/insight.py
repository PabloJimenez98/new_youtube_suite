from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class InsightResult(Base):
    """A precomputed Analytics result (docs/decisions/0023).

    One row per (analysis, format). The worker's `insights_refresh` task
    rewrites a row only when its source data changed (`source_key`); the
    API only reads these rows, so no request ever runs the analyses.
    """

    __tablename__ = "insight_results"
    __table_args__ = (UniqueConstraint("name", "format", name="uq_insight_results_name_format"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    #: overview | clusters | drivers | growth | keywords | channel
    name: Mapped[str] = mapped_column(Text, nullable=False)
    #: long | short | all for the market analyses; "-" for the channel summary
    format: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    #: Fingerprint of the source tables when computed: unchanged → no recompute.
    source_key: Mapped[str] = mapped_column(Text, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
