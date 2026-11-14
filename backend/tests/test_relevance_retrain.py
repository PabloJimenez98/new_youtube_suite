"""M15 with the database: handler dispatch, pseudo-label store, live switch,
dataset loading and the train -> eval -> A/B pipeline (fake encoder, tiny config).
See docs/decisions/0024."""

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.model_registry import VersionNotFoundError, list_versions, set_live
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevancePseudoLabel, RelevanceScore
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.relevance import bilstm, pipeline, retrain
from app.relevance.bilstm import RelevanceBundleError
from app.relevance.dataset import TeacherNotFoundError, load_episodes, resolve_teacher
from app.relevance.pipeline import RelevanceResult, WindowScore
from app.relevance.training import TrainConfig
from app.relevance.windows import Window, build_blocks, build_windows
from app.worker.handlers.relevance import words_from_segments, write_pseudo_labels
from app.worker.tasks import process_job
from tests.test_relevance_bilstm import _bundle, fake_encoder

DESCRIPTION = {"architecture_notes": "notes", "training_data_reference": {"kind": "untrained"}}
TINY = TrainConfig(proj=8, hidden=8, head=4, max_epochs=2, patience=2, lambdas=(0.0,), seeds=(0,))


async def _upload(db: AsyncSession, n_words: int = 20, name: str = "ep.mp4") -> Upload:
    """A transcribed upload: one word per second, so n_words s long."""
    upload = Upload(filename=name, storage_path="k", duration_seconds=float(n_words))
    db.add(upload)
    await db.commit()
    db.add(Job(upload_id=upload.id, tool_type="transcribe", status="succeeded"))
    db.add(TranscriptSegment(
        upload_id=upload.id, segment_index=0, start_seconds=0.0, end_seconds=float(n_words), text="x",
        words=[TranscriptWord(word_index=i, word=f"palabra{i % 7}", start_seconds=float(i), end_seconds=i + 0.5)
               for i in range(n_words)],
    ))
    await db.commit()
    return upload


async def _windows(db: AsyncSession, upload: Upload) -> list[Window]:
    from sqlalchemy.orm import selectinload

    segs = list((await db.execute(select(TranscriptSegment).where(TranscriptSegment.upload_id == upload.id)
                                  .options(selectinload(TranscriptSegment.words)))).scalars())
    return build_windows(build_blocks(words_from_segments(segs), upload.duration_seconds))


async def _teacher(db: AsyncSession, label: str = "heuristic-v0+fake", live: bool = True) -> ModelVersion:
    v = ModelVersion(model_name="relevance", version_label=label, is_live=live)
    db.add(v)
    await db.commit()
    return v


async def _label(db: AsyncSession, upload: Upload, teacher: ModelVersion, seed: int = 0) -> None:
    windows = await _windows(db, upload)
    rng = np.random.default_rng(seed)
    scores = [WindowScore(w, float(rng.random()), None if i == 0 else float(rng.random()), 0.0)
              for i, w in enumerate(windows)]
    await write_pseudo_labels(db, upload.id, teacher, scores)
    await db.commit()


async def _run_job(db, session_factory, upload) -> Job:
    job = Job(upload_id=upload.id, tool_type="relevance")
    db.add(job)
    await db.commit()
    await process_job({"session_factory": session_factory}, job.id)
    await db.refresh(job)
    return job


@pytest.fixture
def fake_teacher(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []

    def _score(words, duration, chapters):
        calls.append(duration)
        windows = build_windows(build_blocks(words, duration))
        scores = [WindowScore(w, 0.5, 0.25, 0.4 * 0.5 + 0.6 * 0.25) for w in windows]
        return "fake-teacher", DESCRIPTION, RelevanceResult("es", scores, 1, 0)

    monkeypatch.setattr(pipeline, "score_relevance", _score)
    return calls


# ---- handler dispatch ----------------------------------------------------------------


async def test_a_live_teacher_writes_scores_and_pseudo_labels(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_teacher
) -> None:
    upload = await _upload(db_session)
    job = await _run_job(db_session, session_factory, upload)
    assert job.status == "succeeded", job.error_message
    scores = (await db_session.execute(select(RelevanceScore))).scalars().all()
    labels = (await db_session.execute(select(RelevancePseudoLabel).order_by(RelevancePseudoLabel.window_index))).scalars().all()
    assert len(scores) == len(labels) == len(await _windows(db_session, upload)) == 2  # 20 s -> 3 blocks
    assert [(lab.heuristic_score, lab.llm_score) for lab in labels] == [(0.5, 0.25)] * len(labels)
    assert labels[0].teacher_version_id == scores[0].model_version_id
    # A re-run replaces both, it doesn't append.
    await _run_job(db_session, session_factory, upload)
    assert len((await db_session.execute(select(RelevancePseudoLabel))).scalars().all()) == len(labels)


async def test_a_live_bilstm_scores_without_the_llm_and_keeps_the_pseudo_labels(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], fake_teacher,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    upload = await _upload(db_session, n_words=40)
    teacher = await _teacher(db_session, live=False)
    await _label(db_session, upload, teacher)
    n_labels = len((await db_session.execute(select(RelevancePseudoLabel))).scalars().all())
    path, _ = _bundle(tmp_path)
    db_session.add(ModelVersion(model_name="relevance", version_label="bilstm-test", is_live=True,
                                artifact_path=str(path)))
    await db_session.commit()
    monkeypatch.setattr(bilstm, "_scorers", {str(path): bilstm.BiLSTMScorer(path, encoder=fake_encoder)})

    job = await _run_job(db_session, session_factory, upload)

    assert job.status == "succeeded", job.error_message
    assert fake_teacher == []  # no LLM
    rows = (await db_session.execute(select(RelevanceScore).order_by(RelevanceScore.window_index))).scalars().all()
    live = await db_session.scalar(select(ModelVersion).where(ModelVersion.version_label == "bilstm-test"))
    assert len(rows) == len(await _windows(db_session, upload)) and {r.model_version_id for r in rows} == {live.id}
    assert all(r.score == pytest.approx(0.4 * r.heuristic_score + 0.6 * r.llm_score) for r in rows)
    assert len((await db_session.execute(select(RelevancePseudoLabel))).scalars().all()) == n_labels


async def test_a_live_bilstm_with_a_missing_bundle_fails_the_job_clearly(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bilstm, "_scorers", {})
    upload = await _upload(db_session)
    db_session.add(ModelVersion(model_name="relevance", version_label="gone", is_live=True,
                                artifact_path=str(tmp_path / "gone")))
    await db_session.commit()
    job = Job(upload_id=upload.id, tool_type="relevance")
    db_session.add(job)
    await db_session.commit()
    with pytest.raises(RelevanceBundleError):
        await process_job({"session_factory": session_factory}, job.id)
    await db_session.refresh(job)
    assert job.status == "failed" and "set-live" in job.error_message


async def test_only_teachers_write_pseudo_labels(db_session: AsyncSession) -> None:
    upload = await _upload(db_session)
    trained = ModelVersion(model_name="relevance", version_label="bilstm", artifact_path="x")
    db_session.add(trained)
    await db_session.commit()
    with pytest.raises(ValueError, match="trained model"):
        await write_pseudo_labels(db_session, upload.id, trained, [])


async def test_two_teachers_keep_their_own_labels(db_session: AsyncSession) -> None:
    upload = await _upload(db_session)
    a, b = await _teacher(db_session, "a"), await _teacher(db_session, "b", live=False)
    await _label(db_session, upload, a, seed=1)
    await _label(db_session, upload, b, seed=2)
    await _label(db_session, upload, a, seed=3)  # a re-label replaces a's only
    rows = (await db_session.execute(select(RelevancePseudoLabel))).scalars().all()
    n = len(await _windows(db_session, upload))
    assert sorted(r.teacher_version_id for r in rows) == sorted([a.id] * n + [b.id] * n)


# ---- live switch -------------------------------------------------------------------


async def test_set_live_switches_and_rolls_back(db_session: AsyncSession, tmp_path: Path) -> None:
    teacher = await _teacher(db_session, "p3")
    path, _ = _bundle(tmp_path)
    db_session.add(ModelVersion(model_name="relevance", version_label="bilstm-test", artifact_path=str(path)))
    await db_session.commit()

    switch = await set_live(db_session, "relevance", "bilstm-test")
    assert (switch.previous, switch.current, switch.changed) == ("p3", "bilstm-test", True)
    live = [v.version_label for v in await list_versions(db_session, "relevance") if v.is_live]
    assert live == ["bilstm-test"]

    back = await set_live(db_session, "relevance", "p3")
    assert back.previous == "bilstm-test"
    await db_session.refresh(teacher)
    assert teacher.is_live
    assert not (await set_live(db_session, "relevance", "p3")).changed


async def test_set_live_refuses_unknown_labels_and_broken_bundles(db_session: AsyncSession, tmp_path: Path) -> None:
    await _teacher(db_session, "p3")
    db_session.add(ModelVersion(model_name="relevance", version_label="broken", artifact_path=str(tmp_path / "x")))
    await db_session.commit()
    with pytest.raises(VersionNotFoundError):
        await set_live(db_session, "relevance", "nope")
    with pytest.raises(VersionNotFoundError):
        await set_live(db_session, "nonsense", "p3")
    with pytest.raises(RelevanceBundleError):
        await set_live(db_session, "relevance", "broken")
    live = [v.version_label for v in await list_versions(db_session) if v.is_live]
    assert live == ["p3"]


# ---- dataset ----------------------------------------------------------------------


async def test_load_episodes_aligns_labels_and_skips_stale_ones(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    teacher = await _teacher(db_session)
    good, stale = await _upload(db_session, 40, "good.mp4"), await _upload(db_session, 40, "stale.mp4")
    await _label(db_session, good, teacher)
    await _label(db_session, stale, teacher)
    await db_session.execute(RelevancePseudoLabel.__table__.delete().where(
        RelevancePseudoLabel.upload_id == stale.id, RelevancePseudoLabel.window_index == 0))
    await db_session.commit()
    episodes, skipped = await load_episodes(session_factory, teacher)
    assert [e.upload_id for e in episodes] == [good.id]
    assert skipped and skipped[0][0] == stale.id and "windows" in skipped[0][1]
    ep = episodes[0]
    assert len(ep.windows) == len(ep.heuristic) == len(ep.llm) and np.isnan(ep.llm[0])
    assert ep.target[0] == pytest.approx(ep.heuristic[0])  # the heuristic alone where the LLM is missing
    assert ep.language == "unknown" and ep.name == "good.mp4"


async def test_resolve_teacher(db_session: AsyncSession) -> None:
    with pytest.raises(TeacherNotFoundError, match="pseudo-label"):
        await resolve_teacher(db_session, None)
    small, big = await _teacher(db_session, "small"), await _teacher(db_session, "big", live=False)
    u1, u2 = await _upload(db_session), await _upload(db_session)
    await _label(db_session, u1, small)
    await _label(db_session, u1, big)
    await _label(db_session, u2, big)
    assert (await resolve_teacher(db_session, None)).version_label == "big"
    assert (await resolve_teacher(db_session, "small")).version_label == "small"
    db_session.add(ModelVersion(model_name="relevance", version_label="trained", artifact_path="x"))
    await db_session.commit()
    with pytest.raises(TeacherNotFoundError, match="trained model"):
        await resolve_teacher(db_session, "trained")


# ---- train -> eval -> A/B -----------------------------------------------------------------


@pytest.fixture
def model_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "relevance"
    monkeypatch.setattr(get_settings(), "relevance_model_root", root)
    return root


async def test_train_eval_and_ab_end_to_end(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], model_root: Path, tmp_path: Path
) -> None:
    teacher = await _teacher(db_session, "p3")
    for i in range(18):
        await _label(db_session, await _upload(db_session, 60, f"ep{i}.mp4"), teacher, seed=i)
    enc_dir = tmp_path / "enc"
    enc_dir.mkdir()
    (enc_dir / "config.json").write_text("{}")
    spec = retrain.EncoderSpec(enc_dir, "fake", "passage: ")

    version = await retrain.train_version(session_factory, "bilstm-t-i1", cfg=TINY, encoder=spec,
                                          encode=fake_encoder, cache_dir=tmp_path / "cache")

    assert not version.is_live and version.artifact_path == str(model_root / "bilstm-t-i1")
    assert (model_root / "split.json").is_file() and (model_root / "bilstm-t-i1" / "encoder" / "config.json").is_file()
    t = version.metrics["train"]
    assert t["teacher"] == "p3" and t["episodes"] == {"train": 3, "val": 7, "test": 8}
    assert version.training_data_reference["kind"] == "distilled"
    with pytest.raises(retrain.RetrainError, match="already exists"):
        await retrain.train_version(session_factory, "bilstm-t-i1", cfg=TINY, encoder=spec, encode=fake_encoder)

    report = await retrain.eval_version(session_factory, "bilstm-t-i1", encode=fake_encoder,
                                        cache_dir=tmp_path / "cache")
    assert report["episodes"] == 8 and set(report["summary"]) == {"bilstm", "heuristic", "constant"}
    assert isinstance(report["gate"]["passed"], bool)
    stored = await db_session.scalar(select(ModelVersion).where(ModelVersion.version_label == "bilstm-t-i1"))
    assert stored.metrics["test"]["gate"] == report["gate"] and "train" in stored.metrics

    sheet_path, key_path = await retrain.write_ab_sheet(session_factory, "bilstm-t-i1", tmp_path / "ab",
                                                        encode=fake_encoder, cache_dir=tmp_path / "cache")
    sheet = sheet_path.read_text(encoding="utf-8")
    assert sheet.count("## Episode ") == 5 and "bilstm-t-i1" not in sheet
    filled = "\n".join(
        (line[:-2] + " tie |") if line.startswith("| ") and line.endswith("| |") else line
        for line in sheet.splitlines()
    )
    sheet_path.write_text(filled, encoding="utf-8")
    ab = await retrain.score_ab_sheet(session_factory, "bilstm-t-i1", tmp_path / "ab")
    assert (ab["ties"], ab["passed"]) == (5, True)

    status = await retrain.promotion_status(session_factory, "bilstm-t-i1")
    assert status["live"] == "p3" and status["ab"] is True
    assert status["promotable"] == report["gate"]["passed"]


async def test_train_and_eval_in_exact_heuristic_mode(
    db_session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], model_root: Path, tmp_path: Path
) -> None:
    import dataclasses
    import json

    teacher = await _teacher(db_session, "p3")
    for i in range(16):
        await _label(db_session, await _upload(db_session, 60, f"ep{i}.mp4"), teacher, seed=i)
    enc_dir = tmp_path / "enc"
    enc_dir.mkdir()
    (enc_dir / "config.json").write_text("{}")
    lid = tmp_path / "lid.176.ftz"
    lid.write_bytes(b"fake")
    cfg = dataclasses.replace(TINY, combine_mode="exact_heuristic")

    version = await retrain.train_version(session_factory, "bilstm-t-i2", cfg=cfg,
                                          encoder=retrain.EncoderSpec(enc_dir, "fake", "passage: "),
                                          encode=fake_encoder, cache_dir=tmp_path / "cache", lid_src_path=lid)

    bundle = Path(version.artifact_path)
    assert json.loads((bundle / "config.json").read_text())["combine"]["mode"] == "exact_heuristic"
    assert (bundle / "lid.176.ftz").is_file() and version.training_data_reference["combine_mode"] == "exact_heuristic"
    report = await retrain.eval_version(session_factory, "bilstm-t-i2", encode=fake_encoder,
                                        cache_dir=tmp_path / "cache")
    # The exact heuristic is the teacher's own component, so the heuristic part agrees perfectly.
    assert report["summary"]["bilstm"]["spearman_heuristic_head"] == pytest.approx(1.0)
