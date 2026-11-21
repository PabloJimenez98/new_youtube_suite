"""create app_settings

Platform-wide key/value settings. First key: `llm_default` ("qwen" | "own"), the LLM the
chapter-titles, show-notes and relevance tools use when a run doesn't pick one. See
docs/decisions/0025-own-llm-m16.md §11. No row means "qwen".

Revision ID: 0018
Revises: 0017
Create Date: 2026-11-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", postgresql.JSONB(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
