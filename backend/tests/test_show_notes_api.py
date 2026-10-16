"""`GET`/`PATCH /uploads/{id}/shownotes` -- status rules from 0011 §5
(mirroring chapters), with rows inserted by hand."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.models.show_notes import ShowNotesDraft
from app.models.upload import Upload


async def _upload(db: AsyncSession, *job_statuses: str, draft: bool = False) -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="k")
    db.add(upload)
    await db.commit()
    for s in job_statuses:
        db.add(Job(upload_id=upload.id, tool_type="shownotes", status=s))
    if draft:
        db.add(ShowNotesDraft(
            upload_id=upload.id, title_suggestions=["T1", "T2", "T3"], description="Desc.",
            tags=["a", "b"], summary="Sum.", language="en", llm_model_used="m+shownotes-v1",
        ))
    await db.commit()
    return upload


async def test_get_200(client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession) -> None:
    upload = await _upload(db_session, "succeeded", draft=True)
    r = await client.get(f"/uploads/{upload.id}/shownotes", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert {k: body[k] for k in ("title_suggestions", "description", "tags", "summary", "language", "llm_model_used")} == {
        "title_suggestions": ["T1", "T2", "T3"], "description": "Desc.", "tags": ["a", "b"],
        "summary": "Sum.", "language": "en", "llm_model_used": "m+shownotes-v1",
    }
    assert body["created_at"] and body["updated_at"]


async def test_previous_draft_served_during_rerun(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "succeeded", "running", draft=True)
    r = await client.get(f"/uploads/{upload.id}/shownotes", headers=auth_headers)
    assert r.status_code == 200


@pytest.mark.parametrize("statuses,code", [((), 404), (("failed",), 404), (("queued",), 409), (("running", "failed"), 409)])
async def test_not_ready_or_missing(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession, statuses, code
) -> None:
    upload = await _upload(db_session, *statuses)
    r = await client.get(f"/uploads/{upload.id}/shownotes", headers=auth_headers)
    assert r.status_code == code


async def test_unknown_upload_404(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    assert (await client.get("/uploads/999999/shownotes", headers=auth_headers)).status_code == 404
    assert (await client.patch("/uploads/999999/shownotes", json={"summary": "x"}, headers=auth_headers)).status_code == 404


@pytest.mark.parametrize("method", ["get", "patch"])
async def test_requires_auth(client: AsyncClient, db_session: AsyncSession, method: str) -> None:
    upload = await _upload(db_session, "succeeded", draft=True)
    kwargs = {"json": {"summary": "x"}} if method == "patch" else {}
    r = await getattr(client, method)(f"/uploads/{upload.id}/shownotes", **kwargs)
    assert r.status_code == 401


async def test_patch_edits_only_the_given_fields(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "succeeded", draft=True)
    before = (await client.get(f"/uploads/{upload.id}/shownotes", headers=auth_headers)).json()

    r = await client.patch(
        f"/uploads/{upload.id}/shownotes",
        json={"title_suggestions": ["My pick"], "tags": ["x", "y z"]},
        headers=auth_headers,
    )

    assert r.status_code == 200
    body = r.json()
    assert body["title_suggestions"] == ["My pick"] and body["tags"] == ["x", "y z"]
    assert body["description"] == "Desc." and body["summary"] == "Sum."
    assert body["updated_at"] >= before["updated_at"]
    again = (await client.get(f"/uploads/{upload.id}/shownotes", headers=auth_headers)).json()
    assert again["title_suggestions"] == ["My pick"]


@pytest.mark.parametrize(
    "body", [{"title_suggestions": []}, {"tags": ["a,b"]}, {"description": "x" * 5001}, {"summary": None},
             {"language": "fr"}],
)
async def test_patch_422_outside_limits(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession, body: dict
) -> None:
    upload = await _upload(db_session, "succeeded", draft=True)
    r = await client.patch(f"/uploads/{upload.id}/shownotes", json=body, headers=auth_headers)
    assert r.status_code == 422


async def test_patch_409_before_the_first_draft(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "running")
    r = await client.patch(f"/uploads/{upload.id}/shownotes", json={"summary": "x"}, headers=auth_headers)
    assert r.status_code == 409
