from httpx import AsyncClient

FILE_BYTES = b"0123456789" * 10  # 100 bytes, easy to slice for range checks


async def _create_upload(client: AsyncClient, auth_headers: dict[str, str]) -> int:
    response = await client.post(
        "/uploads",
        headers=auth_headers,
        files={"file": ("clip.mp4", FILE_BYTES, "video/mp4")},
    )
    return response.json()["id"]


async def test_stream_requires_auth(client: AsyncClient) -> None:
    response = await client.get("/uploads/1/file")
    assert response.status_code == 401


async def test_stream_with_header_auth(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_id = await _create_upload(client, auth_headers)

    response = await client.get(f"/uploads/{upload_id}/file", headers=auth_headers)
    assert response.status_code == 200
    assert response.content == FILE_BYTES
    assert response.headers["content-type"] == "video/mp4"


async def test_stream_with_query_token_auth(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_id = await _create_upload(client, auth_headers)
    token = auth_headers["Authorization"].removeprefix("Bearer ")

    # No header at all -- only the ?token= query param, matching what a
    # <video src> request looks like.
    response = await client.get(f"/uploads/{upload_id}/file?token={token}")
    assert response.status_code == 200
    assert response.content == FILE_BYTES


async def test_stream_query_token_rejects_garbage(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_id = await _create_upload(client, auth_headers)

    response = await client.get(f"/uploads/{upload_id}/file?token=not-a-real-token")
    assert response.status_code == 401


async def test_stream_404_for_missing_upload(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/uploads/999999/file", headers=auth_headers)
    assert response.status_code == 404


async def test_stream_supports_range_requests(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload_id = await _create_upload(client, auth_headers)

    response = await client.get(
        f"/uploads/{upload_id}/file",
        headers={**auth_headers, "Range": "bytes=0-3"},
    )
    assert response.status_code == 206
    assert response.content == FILE_BYTES[0:4]
    assert response.headers["content-range"] == f"bytes 0-3/{len(FILE_BYTES)}"
