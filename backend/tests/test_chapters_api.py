"""`GET /uploads/{id}/chapters` -- status rules from 0009 §7 (mirroring
0007 §5), with rows inserted by hand."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chapter import Chapter
from app.models.job import Job
from app.models.upload import Upload


async def _upload(db: AsyncSession, *job_statuses: str) -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="k")
    db.add(upload)
    await db.commit()
    for s in job_statuses:
        db.add(Job(upload_id=upload.id, tool_type="chapters", status=s))
    await db.commit()
    return upload


async def test_200_ordered(client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession) -> None:
    upload = await _upload(db_session, "succeeded")
    db_session.add_all([
        Chapter(upload_id=upload.id, order_index=1, start_seconds=600.0, title="Second", model_version="v"),
        Chapter(upload_id=upload.id, order_index=0, start_seconds=0.0, title="First", model_version="v"),
    ])
    await db_session.commit()

    r = await client.get(f"/uploads/{upload.id}/chapters", headers=auth_headers)

    assert r.status_code == 200
    assert r.json() == [
        {"order_index": 0, "start_seconds": 0.0, "title": "First", "model_version": "v"},
        {"order_index": 1, "start_seconds": 600.0, "title": "Second", "model_version": "v"},
    ]


async def test_previous_result_served_during_rerun(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "succeeded", "running")
    r = await client.get(f"/uploads/{upload.id}/chapters", headers=auth_headers)
    assert r.status_code == 200


@pytest.mark.parametrize("statuses,code", [((), 404), (("failed",), 404), (("queued",), 409), (("running", "failed"), 409)])
async def test_not_ready_or_missing(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession, statuses, code
) -> None:
    upload = await _upload(db_session, *statuses)
    r = await client.get(f"/uploads/{upload.id}/chapters", headers=auth_headers)
    assert r.status_code == code


async def test_unknown_upload_404(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    r = await client.get("/uploads/999999/chapters", headers=auth_headers)
    assert r.status_code == 404


async def test_requires_auth(client: AsyncClient, db_session: AsyncSession) -> None:
    upload = await _upload(db_session, "succeeded")
    r = await client.get(f"/uploads/{upload.id}/chapters")
    assert r.status_code == 401
