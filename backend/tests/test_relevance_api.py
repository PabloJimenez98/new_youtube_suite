"""`GET /uploads/{id}/relevance` -- status rules and top moments (0012 §6)."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevanceScore
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.routers.relevance import pick_top_moments


async def _upload(db: AsyncSession, *job_statuses: str, scores: list[float] | None = None) -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="k")
    db.add(upload)
    await db.commit()
    for s in job_statuses:
        db.add(Job(upload_id=upload.id, tool_type="relevance", status=s))
    if scores is not None:
        version = ModelVersion(model_name="relevance", version_label="heuristic-v0+x", is_live=True)
        db.add(version)
        await db.flush()
        db.add_all(
            RelevanceScore(upload_id=upload.id, window_index=i, window_start_seconds=7.5 * i,
                           window_end_seconds=7.5 * i + 15, score=s, heuristic_score=s, llm_score=None,
                           model_version_id=version.id)
            for i, s in enumerate(scores)
        )
        db.add(TranscriptSegment(
            upload_id=upload.id, segment_index=0, start_seconds=0.0, end_seconds=200.0, text="x",
            words=[TranscriptWord(word_index=i, word=f"w{i}", start_seconds=float(i * 5)) for i in range(40)],
        ))
    await db.commit()
    return upload


async def test_200_with_windows_and_top_moments(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    scores = [0.1] * 60  # 7.5 s hop -> ~7.5 min, room for 5 moments 60 s apart
    scores[2], scores[3], scores[20] = 0.9, 0.8, 0.7  # window 3 is within 60 s of window 2
    upload = await _upload(db_session, "succeeded", scores=scores)

    r = await client.get(f"/uploads/{upload.id}/relevance", headers=auth_headers)

    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "heuristic-v0+x"
    assert len(body["windows"]) == 60 and body["windows"][1] == {"start": 7.5, "end": 22.5, "score": 0.1}
    top = body["top_moments"]
    assert [(m["start"], m["score"]) for m in top[:2]] == [(15.0, 0.9), (150.0, 0.7)]
    assert len(top) == 5
    assert top[0]["text"] == "w3 w4 w5"  # words starting in [15, 30)
    starts = [m["start"] for m in top]
    assert all(abs(a - b) >= 60 for i, a in enumerate(starts) for b in starts[i + 1:])


def test_pick_top_moments_respects_gap_and_count() -> None:
    rows = [RelevanceScore(window_start_seconds=float(t), score=s)
            for t, s in [(0, 0.5), (30, 0.9), (100, 0.8), (130, 0.85), (400, 0.1)]]
    picked = pick_top_moments(rows, k=3, gap=60)
    assert [r.window_start_seconds for r in picked] == [30.0, 130.0, 400.0]


@pytest.mark.parametrize("statuses,code", [((), 404), (("failed",), 404), (("queued",), 409), (("running", "failed"), 409)])
async def test_not_ready_or_missing(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession, statuses, code
) -> None:
    upload = await _upload(db_session, *statuses)
    assert (await client.get(f"/uploads/{upload.id}/relevance", headers=auth_headers)).status_code == code


async def test_previous_result_served_during_rerun(
    client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
) -> None:
    upload = await _upload(db_session, "succeeded", "running", scores=[0.5, 0.6])
    assert (await client.get(f"/uploads/{upload.id}/relevance", headers=auth_headers)).status_code == 200


async def test_unknown_upload_404_and_auth(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    assert (await client.get("/uploads/999999/relevance", headers=auth_headers)).status_code == 404
    assert (await client.get("/uploads/999999/relevance")).status_code == 401
