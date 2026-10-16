"""`GET /uploads/{id}/clips` and `GET /clips/{id}/file` (docs/decisions/0013 §7)."""

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.clip import Clip
from app.models.job import Job
from app.models.upload import Upload
from app.storage import LocalStorageBackend

storage = LocalStorageBackend(get_settings().storage_base_path)
CLIP_BYTES = b"0123456789-fake-mp4"


async def _upload(db: AsyncSession, *job_statuses: str, n_clips: int = 0) -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="k")
    db.add(upload)
    await db.commit()
    for s in job_statuses:
        db.add(Job(upload_id=upload.id, tool_type="shorts", status=s))
    for rank in range(1, n_clips + 1):
        key = storage.save(f"api_clip_u{upload.id}_{rank}.mp4", CLIP_BYTES)
        db.add(Clip(
            upload_id=upload.id, rank=rank, start_seconds=100.0 * rank, end_seconds=100.0 * rank + 30,
            score=1 - rank / 10, peak_window_index=rank, aspect_ratio="9:16", caption_burned_in=True,
            storage_path=key, crop_bounding_box_track=[], framing={"shots": [{"start": 0, "end": 30, "mode": "crop"}]},
        ))
    await db.commit()
    return upload


async def test_200_lists_clips_by_rank(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "failed", "succeeded", n_clips=2)
    r = await client.get(f"/uploads/{upload.id}/clips", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert [c["rank"] for c in body] == [1, 2]
    assert body[0] | {"id": 0} == {
        "id": 0, "rank": 1, "start": 100.0, "end": 130.0, "duration": 30.0, "score": 0.9,
        "aspect_ratio": "9:16", "caption_burned_in": True,
        "framing": {"shots": [{"start": 0, "end": 30, "mode": "crop"}]},
    }


async def test_404_and_409(client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession) -> None:
    assert (await client.get("/uploads/999999/clips", headers=auth_headers)).status_code == 404
    never = await _upload(db_session)
    assert (await client.get(f"/uploads/{never.id}/clips", headers=auth_headers)).status_code == 404
    running = await _upload(db_session, "running")
    assert (await client.get(f"/uploads/{running.id}/clips", headers=auth_headers)).status_code == 409


async def test_requires_auth(client: AsyncClient, db_session: AsyncSession) -> None:
    upload = await _upload(db_session, "succeeded", n_clips=1)
    assert (await client.get(f"/uploads/{upload.id}/clips")).status_code == 401


async def test_file_streams_with_range_and_query_token(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "succeeded", n_clips=1)
    clip_id = (await client.get(f"/uploads/{upload.id}/clips", headers=auth_headers)).json()[0]["id"]
    token = auth_headers["Authorization"].removeprefix("Bearer ")

    full = await client.get(f"/clips/{clip_id}/file?token={token}")
    assert full.status_code == 200 and full.content == CLIP_BYTES
    assert full.headers["content-type"] == "video/mp4" and "content-disposition" not in full.headers

    part = await client.get(f"/clips/{clip_id}/file", headers={**auth_headers, "Range": "bytes=2-5"})
    assert part.status_code == 206 and part.content == b"2345"
    assert part.headers["content-range"] == f"bytes 2-5/{len(CLIP_BYTES)}"

    dl = await client.get(f"/clips/{clip_id}/file?token={token}&download=true")
    assert dl.headers["content-disposition"] == f'attachment; filename="short-{upload.id}-1.mp4"'


async def test_file_auth_and_404(client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession) -> None:
    upload = await _upload(db_session, "succeeded", n_clips=1)
    clip_id = (await client.get(f"/uploads/{upload.id}/clips", headers=auth_headers)).json()[0]["id"]
    assert (await client.get(f"/clips/{clip_id}/file")).status_code == 401
    assert (await client.get("/clips/999999/file", headers=auth_headers)).status_code == 404


async def test_deleting_the_upload_deletes_clip_files(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "succeeded", n_clips=2)
    keys = [f"api_clip_u{upload.id}_{r}.mp4" for r in (1, 2)]
    assert all(storage.path_for(k).exists() for k in keys)
    assert (await client.delete(f"/uploads/{upload.id}", headers=auth_headers)).status_code == 204
    assert not any(storage.path_for(k).exists() for k in keys)


async def test_count_all_clips(client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession) -> None:
    assert (await client.get("/clips/count", headers=auth_headers)).json() == {"count": 0}
    await _upload(db_session, "succeeded", n_clips=2)
    await _upload(db_session, "succeeded", n_clips=1)
    assert (await client.get("/clips/count", headers=auth_headers)).json() == {"count": 3}
    assert (await client.get("/clips/count")).status_code == 401
