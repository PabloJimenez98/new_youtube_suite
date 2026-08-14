"""create jobs table

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JOB_TOOL_TYPES = (
    "transcribe",
    "diarize",
    "chapters",
    "relevance",
    "shorts",
    "shownotes",
    "ctr",
    "hook",
)
JOB_STATUSES = ("queued", "running", "succeeded", "failed")


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "upload_id",
            sa.BigInteger(),
            sa.ForeignKey("uploads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("progress_pct", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            f"tool_type IN {JOB_TOOL_TYPES!r}", name="jobs_tool_type_check"
        ),
        sa.CheckConstraint(f"status IN {JOB_STATUSES!r}", name="jobs_status_check"),
    )
    op.create_index("ix_jobs_upload_id", "jobs", ["upload_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_upload_id", table_name="jobs")
    op.drop_table("jobs")
