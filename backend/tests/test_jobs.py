from httpx import AsyncClient


async def _create_upload(client: AsyncClient, auth_headers: dict[str, str]) -> int:
    response = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
    )
    return response.json()["id"]


async def test_create_job_requires_auth(client: AsyncClient) -> None:
    response = await client.post("/uploads/1/jobs", json={"tool_type": "transcribe"})
    assert response.status_code == 401


async def test_create_job_success(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_id = await _create_upload(client, auth_headers)

    response = await client.post(
        f"/uploads/{upload_id}/jobs",
        headers=auth_headers,
        json={"tool_type": "transcribe"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["upload_id"] == upload_id
    assert body["tool_type"] == "transcribe"
    assert body["status"] == "queued"
    assert body["progress_pct"] is None


async def test_create_job_rejects_unknown_tool_type(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_id = await _create_upload(client, auth_headers)

    response = await client.post(
        f"/uploads/{upload_id}/jobs",
        headers=auth_headers,
        json={"tool_type": "not-a-real-tool"},
    )
    assert response.status_code == 422


async def test_create_job_404_for_missing_upload(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/uploads/999999/jobs", headers=auth_headers, json={"tool_type": "transcribe"}
    )
    assert response.status_code == 404


async def test_list_jobs_for_upload(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_id = await _create_upload(client, auth_headers)
    await client.post(
        f"/uploads/{upload_id}/jobs",
        headers=auth_headers,
        json={"tool_type": "transcribe"},
    )
    await client.post(
        f"/uploads/{upload_id}/jobs", headers=auth_headers, json={"tool_type": "chapters"}
    )

    response = await client.get(f"/uploads/{upload_id}/jobs", headers=auth_headers)
    assert response.status_code == 200
    tool_types = {job["tool_type"] for job in response.json()}
    assert tool_types == {"transcribe", "chapters"}


async def test_list_jobs_for_upload_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/uploads/999999/jobs", headers=auth_headers)
    assert response.status_code == 404


async def test_list_all_jobs_requires_auth(client: AsyncClient) -> None:
    response = await client.get("/jobs")
    assert response.status_code == 401


async def test_list_all_jobs_spans_uploads(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_a = await _create_upload(client, auth_headers)
    upload_b = await _create_upload(client, auth_headers)
    await client.post(
        f"/uploads/{upload_a}/jobs", headers=auth_headers, json={"tool_type": "transcribe"}
    )
    await client.post(
        f"/uploads/{upload_b}/jobs", headers=auth_headers, json={"tool_type": "hook"}
    )

    response = await client.get("/jobs", headers=auth_headers)
    assert response.status_code == 200
    upload_ids = {job["upload_id"] for job in response.json()}
    assert upload_ids == {upload_a, upload_b}
