"""create reporting_jobs, reporting_reports and analytics_reach_snapshots

The YouTube Reporting API pipeline for `channel_reach_basic_a1` (spec
§4.2/§6.3): the job is a subscription and each daily report file is the
unit of work. See docs/decisions/0018-reporting-api-reach.md §1.

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-08

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JOB_STATUSES = ("active", "expired", "superseded")
REPORT_STATUSES = ("pending", "downloaded", "superseded", "failed")


def upgrade() -> None:
    op.create_table(
        "reporting_jobs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("report_type", sa.Text(), nullable=False),
        sa.Column("youtube_job_id", sa.Text(), nullable=False, unique=True),
        sa.Column("status", sa.Text(), server_default="active", nullable=False),
        sa.Column("job_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expire_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reports_cursor", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_polled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.CheckConstraint(f"status IN {JOB_STATUSES!r}", name="reporting_jobs_status_check"),
    )
    op.create_index(
        "uq_reporting_jobs_active",
        "reporting_jobs",
        ["channel_id", "report_type"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "reporting_reports",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("job_id", sa.BigInteger(), sa.ForeignKey("reporting_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("youtube_report_id", sa.Text(), nullable=False, unique=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("create_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("download_url", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("downloaded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.CheckConstraint(f"status IN {REPORT_STATUSES!r}", name="reporting_reports_status_check"),
    )
    op.create_index("ix_reporting_reports_job_id", "reporting_reports", ["job_id"])
    op.create_table(
        "analytics_reach_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("channel_id", sa.Text(), nullable=False),
        sa.Column("youtube_video_id", sa.Text(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("video_thumbnail_impressions", sa.BigInteger(), nullable=False),
        sa.Column("video_thumbnail_impressions_ctr", sa.Float(), nullable=True),
        sa.Column(
            "source_report_id",
            sa.BigInteger(),
            sa.ForeignKey("reporting_reports.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("pulled_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("channel_id", "youtube_video_id", "report_date", name="uq_analytics_reach_video_date"),
    )
    op.create_index("ix_analytics_reach_snapshots_report_date", "analytics_reach_snapshots", ["report_date"])


def downgrade() -> None:
    op.drop_index("ix_analytics_reach_snapshots_report_date", table_name="analytics_reach_snapshots")
    op.drop_table("analytics_reach_snapshots")
    op.drop_index("ix_reporting_reports_job_id", table_name="reporting_reports")
    op.drop_table("reporting_reports")
    op.drop_index("uq_reporting_jobs_active", table_name="reporting_jobs")
    op.drop_table("reporting_jobs")
