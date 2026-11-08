"""Parser for `channel_reach_basic_a1` CSV files (0018 §3). Pure.

- Columns are mapped **by header name**: YouTube may add columns, and
  unknown ones are ignored.
- A missing required column, or a malformed value, raises
  `ReportFormatError`. The report is then marked `failed`, so no wrong
  data is stored.
- A header-only file is a valid day with no data.
- Rows with `video_id == ""` (channel-level or deleted video) are counted,
  not returned.
- CTR is returned as a **fraction 0-1**. The docs say "percentage" without
  a scale, so `CTR_SCALE` decides the scale:
  - `"auto"`, until the first live file settles it: a file with any value
    above 1 is read as percentages.
  - `"fraction"` or `"percent"` once verified.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

REQUIRED_COLUMNS = ("date", "video_id", "video_thumbnail_impressions", "video_thumbnail_impressions_ctr")
CTR_SCALE: Literal["auto", "fraction", "percent"] = "auto"


class ReportFormatError(ValueError):
    """The file doesn't look like the report we asked for."""


@dataclass(frozen=True)
class ReachRow:
    youtube_video_id: str
    report_date: date
    video_thumbnail_impressions: int
    video_thumbnail_impressions_ctr: float | None


@dataclass
class ParsedReport:
    header: list[str]
    rows: list[ReachRow] = field(default_factory=list)
    channel_level_rows: int = 0
    ctr_scale: str = "fraction"

    @property
    def dates(self) -> set[date]:
        return {r.report_date for r in self.rows}


def _parse_date(value: str, line: int) -> date:
    value = value.strip()
    try:
        if len(value) == 8 and value.isdigit():
            return datetime.strptime(value, "%Y%m%d").date()
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ReportFormatError(f"line {line}: bad date {value!r}") from exc


def _parse_number(value: str, column: str, line: int, kind: type) -> float | int | None:
    value = value.strip()
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ReportFormatError(f"line {line}: bad {column} {value!r}") from exc
    if number < 0:
        raise ReportFormatError(f"line {line}: negative {column} {value!r}")
    return int(number) if kind is int else number


def parse_reach_csv(data: bytes, *, ctr_scale: str = CTR_SCALE) -> ParsedReport:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ReportFormatError(f"not UTF-8: {exc}") from exc
    reader = csv.reader(io.StringIO(text))
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration as exc:
        raise ReportFormatError("empty file (no header row)") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ReportFormatError(f"missing column(s) {', '.join(missing)}; header was {header}")
    index = {name: header.index(name) for name in REQUIRED_COLUMNS}

    parsed = ParsedReport(header=header)
    raw: list[tuple[str, date, int, float | None]] = []
    for line, record in enumerate(reader, start=2):
        if not record or all(not cell.strip() for cell in record):
            continue
        if len(record) < len(header):
            raise ReportFormatError(f"line {line}: {len(record)} fields, header has {len(header)}")
        video_id = record[index["video_id"]].strip()
        if not video_id:
            parsed.channel_level_rows += 1
            continue
        impressions = _parse_number(record[index["video_thumbnail_impressions"]], "impressions", line, int)
        ctr = _parse_number(record[index["video_thumbnail_impressions_ctr"]], "ctr", line, float)
        raw.append((video_id, _parse_date(record[index["date"]], line), impressions or 0, ctr))

    if ctr_scale == "auto":
        ctr_scale = "percent" if any(ctr is not None and ctr > 1 for *_, ctr in raw) else "fraction"
    parsed.ctr_scale = ctr_scale
    divisor = 100.0 if ctr_scale == "percent" else 1.0
    for video_id, day, impressions, ctr in raw:
        value = None if ctr is None else ctr / divisor
        if value is not None and value > 1:
            raise ReportFormatError(f"CTR {ctr} for {video_id} on {day} is above 100%")
        parsed.rows.append(ReachRow(video_id, day, impressions, value))
    return parsed
