"""Qwen / own-LLM choice (M16, docs/decisions/0025-own-llm-m16.md §11): registry, global setting,
per-run params recorded at job creation, and the three LLM tools routing their runs."""

import json
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chapters import pipeline as chapters_pipeline
from app.chapters.pipeline import ChapterResult
from app.config import get_settings
from app.llm import registry
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevancePseudoLabel, RelevanceScore
from app.models.show_notes import ShowNotesDraft
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.relevance import pipeline as relevance_pipeline
from app.relevance.pipeline import RelevanceResult, WindowScore
from app.relevance.windows import Window
from app.settings_store import job_llm
from app.shownotes import pipeline as shownotes_pipeline
from app.shownotes.pipeline import ShowNotes
from app.shownotes.validate import Draft
from app.worker.tasks import process_job


@pytest.fixture
def own_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installed own model: LLM_OWN_DIR with a manifest and a (dummy) GGUF file."""
    d = tmp_path / "own"
    d.mkdir()
    (d / "yts-own-a.gguf").write_bytes(b"GGUF")
    (d / "manifest.json").write_text(json.dumps({"file": "yts-own-a.gguf", "n_ctx": 4096, "label": "yts-own-a"}))
    monkeypatch.setattr(get_settings(), "llm_own_dir", d)
    return d


@pytest.fixture
def no_own_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "llm_own_dir", tmp_path / "missing")


@pytest.fixture
def qwen_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    b = tmp_path / "bundle"
    (b / "llm").mkdir(parents=True)
    (b / "llm" / "Qwen3.5-9B-Q4_K_M.gguf").write_bytes(b"GGUF")
    (b / "manifest.json").write_text(json.dumps(
        {"version": "v2", "titles": {"llm": {"file": "llm/Qwen3.5-9B-Q4_K_M.gguf", "n_ctx": 4096}}}))
    monkeypatch.setattr(get_settings(), "chapters_model_dir", b)
    return b


# ---------------------------------------------------------------- registry


def test_resolve_qwen_and_own(qwen_bundle: Path, own_llm: Path) -> None:
    q = registry.resolve("qwen")
    assert q.path == qwen_bundle / "llm" / "Qwen3.5-9B-Q4_K_M.gguf" and q.label == "Qwen3.5-9B-Q4_K_M"
    assert registry.resolve(None) == q  # no choice = Qwen
    o = registry.resolve("own")
    assert o.path == own_llm / "yts-own-a.gguf" and o.n_ctx == 4096 and o.label == "yts-own-a"
    assert registry.own_available() and registry.own_label() == "yts-own-a"


def test_resolve_own_missing_or_unknown(no_own_llm) -> None:
    assert not registry.own_available()
    with pytest.raises(registry.LlmUnavailableError, match="isn't installed"):
        registry.resolve("own")
    with pytest.raises(registry.LlmUnavailableError, match="Unknown"):
        registry.resolve("gpt")


def test_own_manifest_pointing_at_missing_gguf_is_unavailable(own_llm: Path) -> None:
    (own_llm / "yts-own-a.gguf").unlink()
    assert not registry.own_available()


def test_job_llm_defaults_to_qwen_for_old_jobs() -> None:
    assert job_llm(None) == "qwen"
    assert job_llm({"aspect_ratio": "9:16"}) == "qwen"
    assert job_llm({"llm": "own"}) == "own"


# ---------------------------------------------------------------- settings API + job creation


async def _upload(db: AsyncSession) -> Upload:
    u = Upload(filename="ep.mp4", storage_path="k", duration_seconds=600.0)
    db.add(u)
    await db.commit()
    return u


async def test_settings_default_qwen_and_own_rejected_when_missing(
    client: AsyncClient, auth_headers: dict, no_own_llm
) -> None:
    r = await client.get("/settings/llm", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["default"] == "qwen" and r.json()["own_available"] is False
    r = await client.put("/settings/llm", json={"default": "own"}, headers=auth_headers)
    assert r.status_code == 422
    assert (await client.get("/settings/llm", headers=auth_headers)).json()["default"] == "qwen"


async def test_settings_switch_to_own_and_back(client: AsyncClient, auth_headers: dict, own_llm) -> None:
    r = await client.put("/settings/llm", json={"default": "own"}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["default"] == "own" and r.json()["own_label"] == "yts-own-a"
    assert (await client.get("/settings/llm", headers=auth_headers)).json()["default"] == "own"
    r = await client.put("/settings/llm", json={"default": "qwen"}, headers=auth_headers)
    assert r.json()["default"] == "qwen"
    assert (await client.put("/settings/llm", json={"default": "gpt"}, headers=auth_headers)).status_code == 422


async def test_settings_need_auth(client: AsyncClient) -> None:
    assert (await client.get("/settings/llm")).status_code == 401


@pytest.mark.parametrize("tool", ["chapters", "shownotes", "relevance"])
async def test_job_records_llm_from_global_default_and_request(
    client: AsyncClient, auth_headers: dict, db_session: AsyncSession, own_llm, tool: str
) -> None:
    u = await _upload(db_session)
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": tool}, headers=auth_headers)
    assert r.status_code == 201, r.text
    assert r.json()["params"] == {"llm": "qwen"}  # no row in app_settings = Qwen
    await client.put("/settings/llm", json={"default": "own"}, headers=auth_headers)
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": tool}, headers=auth_headers)
    assert r.json()["params"] == {"llm": "own"}  # the global default, recorded on the job
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": tool, "params": {"llm": "qwen"}},
                          headers=auth_headers)
    assert r.json()["params"] == {"llm": "qwen"}  # a per-run choice wins


async def test_job_with_own_rejected_when_missing(
    client: AsyncClient, auth_headers: dict, db_session: AsyncSession, no_own_llm
) -> None:
    u = await _upload(db_session)
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": "chapters", "params": {"llm": "own"}},
                          headers=auth_headers)
    assert r.status_code == 422
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": "chapters", "params": {"llm": "gpt"}},
                          headers=auth_headers)
    assert r.status_code == 422


async def test_non_llm_tools_take_no_llm_param(client: AsyncClient, auth_headers: dict, db_session: AsyncSession) -> None:
    u = await _upload(db_session)
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": "transcribe"}, headers=auth_headers)
    assert r.status_code == 201 and r.json()["params"] is None
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": "transcribe", "params": {"llm": "own"}},
                          headers=auth_headers)
    assert r.status_code == 422
    r = await client.post(f"/uploads/{u.id}/jobs", json={"tool_type": "shorts"}, headers=auth_headers)
    assert "llm" not in r.json()["params"]


# ---------------------------------------------------------------- handlers route the run's LLM


async def _transcribed(db: AsyncSession) -> Upload:
    u = await _upload(db)
    db.add(Job(upload_id=u.id, tool_type="transcribe", status="succeeded"))
    db.add(TranscriptSegment(upload_id=u.id, segment_index=0, start_seconds=0.0, end_seconds=2.0, text="hola"))
    await db.commit()
    return u


async def _run(db, sf, upload, tool, params) -> Job:
    job = Job(upload_id=upload.id, tool_type=tool, params=params)
    db.add(job)
    await db.commit()
    await process_job({"session_factory": sf}, job.id)
    await db.refresh(job)
    return job


@pytest.mark.parametrize("params,expected", [(None, ()), ({"llm": "qwen"}, ()), ({"llm": "own"}, ("own",))])
async def test_chapters_handler_passes_choice(db_session, session_factory, monkeypatch, params, expected) -> None:
    calls = []

    def _gen(atoms, duration, *extra):
        calls.append(extra)
        return "v+titles:yts-own-a" if extra else "v", [ChapterResult(0.0, "Intro")]

    monkeypatch.setattr(chapters_pipeline, "generate_chapters", _gen)
    job = await _run(db_session, session_factory, await _transcribed(db_session), "chapters", params)
    assert job.status == "succeeded", job.error_message
    assert calls == [expected]  # Qwen: the exact pre-M16 call (no extra argument)


@pytest.mark.parametrize("params,expected", [(None, ()), ({"llm": "own"}, ("own",))])
async def test_shownotes_handler_passes_choice(db_session, session_factory, monkeypatch, params, expected) -> None:
    calls = []

    def _gen(segs, chapters, *extra):
        calls.append(extra)
        return "yts-own-a+p1" if extra else "qwen+p1", ShowNotes(Draft(["T"], "d", ["t"], "s"), "es")

    monkeypatch.setattr(shownotes_pipeline, "generate_show_notes", _gen)
    job = await _run(db_session, session_factory, await _transcribed(db_session), "shownotes", params)
    assert job.status == "succeeded", job.error_message
    assert calls == [expected]
    draft = await db_session.scalar(select(ShowNotesDraft).where(ShowNotesDraft.upload_id == job.upload_id))
    assert draft.llm_model_used == ("yts-own-a+p1" if expected else "qwen+p1")


def _fake_relevance(calls):
    def _score(words, duration, chapters, *extra):
        calls.append(extra)
        scores = [WindowScore(Window(0, 0.0, 15.0, ()), 0.5, 0.9, 0.74)]
        label = "heuristic-v0+yts-own-a+relevance-p3" if extra else "heuristic-v0+qwen+relevance-p3"
        return label, {"architecture_notes": "n", "training_data_reference": {"kind": "untrained"}}, \
            RelevanceResult("es", scores, 1, 0)
    return _score


async def test_relevance_own_runs_zero_shot_even_with_bilstm_live_and_writes_no_pseudo_labels(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(relevance_pipeline, "score_relevance", _fake_relevance(calls))
    db_session.add(ModelVersion(model_name="relevance", version_label="bilstm-live", is_live=True,
                                artifact_path="/nonexistent/bundle"))
    await db_session.commit()
    job = await _run(db_session, session_factory, await _transcribed(db_session), "relevance", {"llm": "own"})
    assert job.status == "succeeded", job.error_message
    assert calls == [("own",)]
    v = await db_session.scalar(select(ModelVersion).where(
        ModelVersion.version_label == "heuristic-v0+yts-own-a+relevance-p3"))
    assert v is not None and v.is_live is False
    assert (await db_session.scalar(select(RelevanceScore).where(RelevanceScore.upload_id == job.upload_id))) is not None
    assert (await db_session.scalar(select(RelevancePseudoLabel))) is None  # never the BiLSTM's targets


async def test_relevance_own_never_goes_live_on_empty_registry(db_session, session_factory, monkeypatch) -> None:
    monkeypatch.setattr(relevance_pipeline, "score_relevance", _fake_relevance([]))
    job = await _run(db_session, session_factory, await _transcribed(db_session), "relevance", {"llm": "own"})
    assert job.status == "succeeded", job.error_message
    assert (await db_session.scalar(select(ModelVersion).where(ModelVersion.is_live))) is None


async def test_relevance_qwen_unchanged_writes_pseudo_labels(db_session, session_factory, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(relevance_pipeline, "score_relevance", _fake_relevance(calls))
    job = await _run(db_session, session_factory, await _transcribed(db_session), "relevance", {"llm": "qwen"})
    assert job.status == "succeeded", job.error_message
    assert calls == [()]
    assert (await db_session.scalar(select(RelevancePseudoLabel))) is not None
