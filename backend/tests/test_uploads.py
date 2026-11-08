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


# ---- YouTube link (docs/decisions/0022 §1) ------------------------------------


async def test_upload_reads_the_youtube_id_from_the_filename(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    name = "SHOW 1X09： ¿Torres? [oSCkZ8yVz7M].mkv"
    body = (await client.post("/uploads", headers=auth_headers, files={"file": (name, b"x", "video/x-matroska")})).json()
    assert body["youtube_video_id"] == "oSCkZ8yVz7M"
    assert body["youtube"] is None  # the market pipeline hasn't seen it

    plain = (await client.post("/uploads", headers=auth_headers, files={"file": ("clip.mp4", b"x", "video/mp4")})).json()
    assert plain["youtube_video_id"] is None


async def test_patch_links_and_unlinks_the_video(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = (await client.post("/uploads", headers=auth_headers, files={"file": ("clip.mp4", b"x", "video/mp4")})).json()
    url = f"/uploads/{created['id']}"
    assert (await client.patch(url, headers=auth_headers, json={"youtube_video_id": " jQ0W3HEcO7M "})).json()["youtube_video_id"] == "jQ0W3HEcO7M"
    assert (await client.patch(url, headers=auth_headers, json={"youtube_video_id": "too-short"})).status_code == 422
    assert (await client.patch(url, headers=auth_headers, json={"youtube_video_id": None})).json()["youtube_video_id"] is None
    assert (await client.patch("/uploads/999999", headers=auth_headers, json={"youtube_video_id": None})).status_code == 404


async def test_list_attaches_the_latest_market_snapshot(
    client: AsyncClient, auth_headers: dict[str, str], db_session
) -> None:
    from datetime import date, datetime, timezone

    from app.models.market import MarketChannel, MarketVideoSnapshot

    channel = MarketChannel(youtube_channel_id="UCx", channel_title="Show")
    db_session.add(channel)
    await db_session.flush()
    for day, views in [(date(2026, 11, 7), 100), (date(2026, 11, 8), 150)]:
        db_session.add(
            MarketVideoSnapshot(
                market_channel_id=channel.id,
                youtube_video_id="oSCkZ8yVz7M",
                snapshot_date=day,
                title="Show 1X09",
                published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                view_count=views,
                thumbnail_url="https://i.ytimg.com/vi/oSCkZ8yVz7M/hqdefault.jpg",
            )
        )
    await db_session.commit()
    await client.post("/uploads", headers=auth_headers, files={"file": ("Ep [oSCkZ8yVz7M].mkv", b"x", "video/x-matroska")})

    item = (await client.get("/uploads", headers=auth_headers)).json()[0]
    assert item["youtube"]["view_count"] == 150
    assert item["youtube"]["snapshot_date"] == "2026-11-08"
    assert item["youtube"]["thumbnail_url"].endswith("hqdefault.jpg")
