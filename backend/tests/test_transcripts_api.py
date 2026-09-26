"""`GET /uploads/{id}/transcript.{srt,vtt}` -- status rules from
docs/decisions/0007-transcription-pipeline.md §5, with rows inserted by
hand (no worker/model involved)."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload


async def _make_upload(db: AsyncSession, *job_statuses: str) -> Upload:
    upload = Upload(filename="My Episode.final.mp4", storage_path="fake-key")
    db.add(upload)
    await db.commit()
    await db.refresh(upload)
    for job_status in job_statuses:
        db.add(Job(upload_id=upload.id, tool_type="transcribe", status=job_status))
    await db.commit()
    return upload


async def _add_segments(db: AsyncSession, upload: Upload) -> None:
    # Inserted out of order on purpose: the endpoint must sort by index.
    db.add_all(
        [
            TranscriptSegment(
                upload_id=upload.id,
                segment_index=1,
                start_seconds=2.5,
                end_seconds=4.0,
                text="Second <cue>.",
            ),
            TranscriptSegment(
                upload_id=upload.id,
                segment_index=0,
                start_seconds=0.0,
                end_seconds=2.5,
                text="First cue.",
                words=[TranscriptWord(word_index=0, word="First")],
            ),
        ]
    )
    await db.commit()


async def test_srt_200(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _make_upload(db_session, "succeeded")
    await _add_segments(db_session, upload)

    response = await client.get(f"/uploads/{upload.id}/transcript.srt", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-subrip; charset=utf-8"
    assert response.headers["content-disposition"] == (
        'attachment; filename="My Episode.final.srt"; '
        "filename*=UTF-8''My%20Episode.final.srt"
    )
    assert response.text == (
        "1\n00:00:00,000 --> 00:00:02,500\nFirst cue.\n"
        "\n"
        "2\n00:00:02,500 --> 00:00:04,000\nSecond <cue>.\n"
    )


async def test_vtt_200(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _make_upload(db_session, "succeeded")
    await _add_segments(db_session, upload)

    response = await client.get(f"/uploads/{upload.id}/transcript.vtt", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/vtt; charset=utf-8"
    assert 'filename="My Episode.final.vtt"' in response.headers["content-disposition"]
    assert response.text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nFirst cue.\n")
    assert "Second &lt;cue&gt;." in response.text


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
async def test_non_ascii_filename_is_encoded_in_the_header(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession, fmt: str
) -> None:
    # Regression: a YouTube-style title with a fullwidth colon used to
    # crash with UnicodeEncodeError (header values must be latin-1).
    upload = Upload(filename='Episodio 12：Ñandú "live" 🎙.mp4', storage_path="fake-key")
    db_session.add(upload)
    await db_session.commit()
    db_session.add(Job(upload_id=upload.id, tool_type="transcribe", status="succeeded"))
    await db_session.commit()

    response = await client.get(f"/uploads/{upload.id}/transcript.{fmt}", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        f'attachment; filename="Episodio 12__and_ _live_ _.{fmt}"; '
        "filename*=UTF-8''Episodio%2012%EF%BC%9A%C3%91and%C3%BA%20%22live%22%20%F0%9F%8E%99"
        f".{fmt}"
    )


async def test_succeeded_with_no_segments_is_an_empty_valid_file(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _make_upload(db_session, "succeeded")

    response = await client.get(f"/uploads/{upload.id}/transcript.vtt", headers=auth_headers)

    assert response.status_code == 200
    assert response.text == "WEBVTT\n\n"


async def test_rerun_in_flight_still_serves_previous_transcript(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _make_upload(db_session, "succeeded", "running")
    await _add_segments(db_session, upload)

    response = await client.get(f"/uploads/{upload.id}/transcript.srt", headers=auth_headers)

    assert response.status_code == 200


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
@pytest.mark.parametrize(
    "job_statuses", [("queued",), ("running",), ("failed", "queued")]
)
async def test_409_while_not_ready(
    client: AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
    fmt: str,
    job_statuses: tuple[str, ...],
) -> None:
    upload = await _make_upload(db_session, *job_statuses)

    response = await client.get(f"/uploads/{upload.id}/transcript.{fmt}", headers=auth_headers)

    assert response.status_code == 409


@pytest.mark.parametrize("job_statuses", [(), ("failed",)])
async def test_404_without_a_transcript(
    client: AsyncClient,
    auth_headers: dict[str, str],
    db_session: AsyncSession,
    job_statuses: tuple[str, ...],
) -> None:
    upload = await _make_upload(db_session, *job_statuses)

    response = await client.get(f"/uploads/{upload.id}/transcript.srt", headers=auth_headers)

    assert response.status_code == 404


async def test_other_tools_jobs_do_not_count(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _make_upload(db_session)
    db_session.add(Job(upload_id=upload.id, tool_type="chapters", status="succeeded"))
    await db_session.commit()

    response = await client.get(f"/uploads/{upload.id}/transcript.srt", headers=auth_headers)

    assert response.status_code == 404


async def test_404_unknown_upload(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    response = await client.get("/uploads/999999/transcript.srt", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
async def test_401_without_token(client: AsyncClient, fmt: str) -> None:
    response = await client.get(f"/uploads/1/transcript.{fmt}")
    assert response.status_code == 401
