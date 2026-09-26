"""add uploads.audio_storage_path

Storage key of the 16kHz mono WAV extracted by the transcribe job (M3),
kept for reuse by diarization (M4). See
docs/decisions/0007-transcription-pipeline.md.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-26

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("uploads", sa.Column("audio_storage_path", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("uploads", "audio_storage_path")
