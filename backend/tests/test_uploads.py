from pathlib import Path

from httpx import AsyncClient


async def test_create_upload_requires_auth(client: AsyncClient) -> None:
    response = await client.post(
        "/uploads", files={"file": ("clip.mp4", b"fake bytes", "video/mp4")}
    )
    assert response.status_code == 401


async def test_create_upload_success(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "clip.mp4"
    assert body["status"] == "uploaded"
    assert body["duration_seconds"] is None
    assert body["storage_path"].endswith("_clip.mp4")

    # The file actually landed on disk under the (test-isolated) storage dir.
    from app.config import get_settings

    storage_dir = Path(get_settings().storage_base_path)
    assert (storage_dir / body["storage_path"]).is_file()


async def test_list_uploads_empty(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/uploads", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


async def test_list_uploads_returns_created(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
    )

    response = await client.get("/uploads", headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json()) == 1


async def test_get_upload_detail(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
    )
    upload_id = created.json()["id"]

    response = await client.get(f"/uploads/{upload_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["id"] == upload_id


async def test_get_upload_404(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    response = await client.get("/uploads/999999", headers=auth_headers)
    assert response.status_code == 404


async def test_delete_upload_removes_row_and_file(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    from app.config import get_settings

    created = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
    )
    body = created.json()
    upload_id = body["id"]
    file_path = Path(get_settings().storage_base_path) / body["storage_path"]
    assert file_path.is_file()

    response = await client.delete(f"/uploads/{upload_id}", headers=auth_headers)
    assert response.status_code == 204

    assert not file_path.is_file()
    assert (await client.get(f"/uploads/{upload_id}", headers=auth_headers)).status_code == 404


async def test_delete_upload_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.delete("/uploads/999999", headers=auth_headers)
    assert response.status_code == 404


async def test_delete_upload_cascades_jobs(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
    )
    upload_id = created.json()["id"]
    await client.post(
        f"/uploads/{upload_id}/jobs", headers=auth_headers, json={"tool_type": "transcribe"}
    )

    await client.delete(f"/uploads/{upload_id}", headers=auth_headers)

    # The job row is gone too (ON DELETE CASCADE) -- listing jobs for a
    # now-nonexistent upload 404s rather than returning an orphaned job.
    response = await client.get(f"/uploads/{upload_id}/jobs", headers=auth_headers)
    assert response.status_code == 404
