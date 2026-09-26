from sqlalchemy import BigInteger, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint(
            "upload_id", "segment_index", name="uq_transcript_segments_upload_index"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    upload_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    avg_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_speech_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Populated by diarization (M4).
    speaker_label: Mapped[str | None] = mapped_column(Text, nullable=True)

    words: Mapped[list["TranscriptWord"]] = relationship(
        back_populates="segment",
        order_by="TranscriptWord.word_index",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class TranscriptWord(Base):
    __tablename__ = "transcript_words"
    __table_args__ = (
        UniqueConstraint(
            "segment_id", "word_index", name="uq_transcript_words_segment_index"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    segment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("transcript_segments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    word_index: Mapped[int] = mapped_column(Integer, nullable=False)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    # NULL for characters Whisper couldn't align.
    start_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    speaker_label: Mapped[str | None] = mapped_column(Text, nullable=True)

    segment: Mapped[TranscriptSegment] = relationship(back_populates="words")
