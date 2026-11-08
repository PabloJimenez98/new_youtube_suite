"""Which report files count (0018 §2, step 4). Pure.

YouTube corrects a day by posting a **new** report with the same period
and a newer `createTime`. Per `report_date` only the newest file counts:

- every older file becomes `superseded`, whatever its status;
- the newest one is downloaded if it is still `pending`.

A `failed` newest file stays failed until a newer correction supersedes
it.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class ReportState:
    id: int
    report_date: date
    create_time: datetime
    status: str


@dataclass(frozen=True)
class Resolution:
    download: list[int]  # oldest date first
    supersede: list[int]


def resolve_reports(reports: list[ReportState]) -> Resolution:
    by_date: dict[date, list[ReportState]] = defaultdict(list)
    for r in reports:
        by_date[r.report_date].append(r)
    download: list[int] = []
    supersede: list[int] = []
    for day in sorted(by_date):
        newest, *older = sorted(by_date[day], key=lambda r: (r.create_time, r.id), reverse=True)
        supersede.extend(r.id for r in older if r.status != "superseded")
        if newest.status == "pending":
            download.append(newest.id)
    return Resolution(download=download, supersede=supersede)
