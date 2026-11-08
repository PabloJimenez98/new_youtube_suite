"""YouTube Reporting API v1 client (M12, docs/decisions/0018).

This is an async bulk-report system, not a live query like `reports.query`:

- `jobs.list` / `jobs.create`: a job is a standing subscription to one
  report type.
- `jobs.reports.list`: the daily report files the job has produced so
  far. `createdAfter` also catches backfilled corrections.
- `downloadUrl`: an authorized GET for the CSV. It is gzip-encoded on the
  wire; httpx decodes it.

Every call goes through `TokenProvider.call_with_token`, which refreshes
on a 401, and through the shared `google_request` policy: quota stop, and
retries for 429/5xx.
"""

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from app.market.youtube_api import google_request
from app.youtube_oauth.tokens import TokenProvider

REPORTING_BASE = "https://youtubereporting.googleapis.com/v1"
JOB_NAME_PREFIX = "youtube-suite"


@dataclass(frozen=True)
class RemoteJob:
    id: str
    report_type_id: str
    create_time: datetime
    expire_time: datetime | None


@dataclass(frozen=True)
class RemoteReport:
    id: str
    start_time: datetime
    end_time: datetime
    create_time: datetime
    download_url: str
    job_expire_time: datetime | None


def parse_time(value: str | None) -> datetime | None:
    """RFC 3339, with `Z` and with or without fractional seconds (Google
    sends nanoseconds, more than `fromisoformat` takes)."""
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    if "." in value:
        head, rest = value.split(".", 1)
        digits = "".join(itertools.takewhile(str.isdigit, rest))
        tz = rest[len(digits):]
        value = f"{head}.{digits[:6].ljust(6, '0')}{tz}"
    return datetime.fromisoformat(value)


def _job(item: dict[str, Any]) -> RemoteJob:
    return RemoteJob(
        id=item["id"],
        report_type_id=item["reportTypeId"],
        create_time=parse_time(item["createTime"]),
        expire_time=parse_time(item.get("expireTime")),
    )


def _report(item: dict[str, Any]) -> RemoteReport:
    return RemoteReport(
        id=item["id"],
        start_time=parse_time(item["startTime"]),
        end_time=parse_time(item["endTime"]),
        create_time=parse_time(item["createTime"]),
        download_url=item["downloadUrl"],
        job_expire_time=parse_time(item.get("jobExpireTime")),
    )


def rfc3339(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class ReportingClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        tokens: TokenProvider,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._tokens = tokens
        self._sleep = sleep
        self.calls = 0

    async def _call(
        self,
        method: str,
        url: str,
        label: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        self.calls += 1

        async def attempt(token: str) -> httpx.Response:
            return await google_request(
                self._http,
                method,
                url,
                params,
                headers={"Authorization": f"Bearer {token}", **(extra_headers or {})},
                label=label,
                json_body=json_body,
                sleep=self._sleep,
            )

        return await self._tokens.call_with_token(attempt)

    async def _paged(self, url: str, key: str, label: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            page_params = {**params, **({"pageToken": page_token} if page_token else {})}
            data = (await self._call("GET", url, label, params=page_params)).json()
            items.extend(data.get(key) or [])
            page_token = data.get("nextPageToken")
            if not page_token:
                return items

    async def list_jobs(self) -> list[RemoteJob]:
        return [_job(j) for j in await self._paged(f"{REPORTING_BASE}/jobs", "jobs", "jobs.list", {})]

    async def create_job(self, report_type: str) -> RemoteJob:
        response = await self._call(
            "POST",
            f"{REPORTING_BASE}/jobs",
            "jobs.create",
            json_body={"reportTypeId": report_type, "name": f"{JOB_NAME_PREFIX}-{report_type}"},
        )
        return _job(response.json())

    async def list_reports(self, job_id: str, created_after: datetime | None = None) -> list[RemoteReport]:
        params: dict[str, Any] = {}
        if created_after is not None:
            params["createdAfter"] = rfc3339(created_after)
        items = await self._paged(f"{REPORTING_BASE}/jobs/{job_id}/reports", "reports", "jobs.reports.list", params)
        return [_report(r) for r in items]

    async def download(self, download_url: str) -> bytes:
        response = await self._call(
            "GET", download_url, "report download", extra_headers={"Accept-Encoding": "gzip"}
        )
        return response.content
