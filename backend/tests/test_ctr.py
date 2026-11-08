"""M13 CTR predictor v0 (docs/decisions/0019): title features, scorer
shapes, thumbnail features (real detectors; skipped without the ONNX
models), validation, the candidate endpoints and the `ctr` job handler."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.ctr import scorer
from app.ctr.detectors import DetectorModelMissingError
from app.ctr.features import Box, ThumbnailFeatures, fit_size, image_stats, thumbnail_features, title_features, union_area
from app.ctr.pipeline import InvalidThumbnailError, analyse_thumbnail, sniff_type, validate_thumbnail
from app.models.ctr import CtrPrediction
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.show_notes import ShowNotesDraft
from app.models.upload import Upload
from app.routers.ctr import storage
from app.worker.handlers.ctr import NoCandidatesError
from app.worker.tasks import process_job

FIXTURES = Path(__file__).parent / "fixtures"
settings = get_settings()
FACE_MODEL, TEXT_MODEL = settings.shorts_face_model_path, settings.ctr_text_model_path
has_models = all(p is not None and Path(p).is_file() for p in (FACE_MODEL, TEXT_MODEL))
needs_models = pytest.mark.skipif(not has_models, reason="YuNet / PP-OCRv3 models not downloaded (README, 'CTR predictor')")


def _png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def _blank(w: int = 1280, h: int = 720, value: int = 40) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def _with_text(text: str = "HUGE NEWS", scale: float = 6, thickness: int = 18) -> np.ndarray:
    img = _blank()
    cv2.putText(img, text, (80, 400), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
    return img


# ---- title features --------------------------------------------------------------


@pytest.mark.parametrize(
    "title, field, expected",
    [
        ("  Hello   world  ", "title_chars", 13),
        ("Hello world", "title_words", 2),
        ("", "title_chars", 0),
        ("Top 5 tips", "has_digit", True),
        ("No digits here", "has_digit", False),
        ("Why does it work?", "has_question", True),
        ("¿Por qué funciona", "has_question", True),
        ("Wow!", "has_exclamation", True),
        ("¡Vaya", "has_exclamation", True),
        ("Episode (with Ana)", "has_brackets", True),
        ("[LIVE] episode", "has_brackets", True),
        ("Unclosed ( bracket", "has_brackets", False),
        ("THIS is HUGE news", "caps_words", 2),
        ("I am A person", "caps_words", 0),  # single letters don't count
        ("Your next move", "second_person", True),
        ("Lo que tú no sabes", "second_person", True),
        ("A young founder", "second_person", False),  # whole words only
        ("Café ☕ con Ana", "title_chars", 14),  # accents and emoji are characters
    ],
)
def test_title_features(title: str, field: str, expected) -> None:
    assert getattr(title_features(title), field) == expected


# ---- scorer ------------------------------------------------------------------------


def test_weights_sum_to_one_and_cover_the_sub_scores() -> None:
    assert sum(scorer.WEIGHTS.values()) == pytest.approx(1.0)
    assert set(scorer.WEIGHTS) == {"title_length", *scorer.SHAPES} | {"faces"}


@pytest.mark.parametrize(
    "x, expected",
    [(0, 0.0), (5, 0.0), (22.5, 0.5), (40, 1.0), (50, 1.0), (60, 1.0), (80, 0.7), (100, 0.4), (150, 0.4)],
)
def test_title_length_shape(x: float, expected: float) -> None:
    assert scorer.piecewise(x, scorer.SHAPES["title_length"]) == pytest.approx(expected)


@pytest.mark.parametrize("x, expected", [(0.0, 0.0), (0.05, 0.5), (0.10, 1.0), (0.35, 1.0), (0.60, 0.6), (0.9, 0.6)])
def test_face_size_shape(x: float, expected: float) -> None:
    assert scorer.piecewise(x, scorer.SHAPES["face_size"]) == pytest.approx(expected)


@pytest.mark.parametrize("x, expected", [(0.0, 0.7), (0.01, 0.85), (0.02, 1.0), (0.20, 1.0), (0.35, 0.65), (0.5, 0.3), (0.8, 0.3)])
def test_text_coverage_shape(x: float, expected: float) -> None:
    assert scorer.piecewise(x, scorer.SHAPES["text_coverage"]) == pytest.approx(expected)


def test_face_count_scores() -> None:
    assert [scorer.face_count_score(n) for n in range(5)] == [0.2, 0.9, 1.0, 0.9, 0.9]


def _thumb(**kw) -> ThumbnailFeatures:
    base = dict(width=1280, height=720, aspect_ratio=1.7778, face_count=2, largest_face_area=0.2,
                text_area=0.1, text_boxes=1, brightness=0.5, contrast=0.5, saturation=0.5)
    return ThumbnailFeatures(**(base | kw))


def test_best_case_scores_100_and_breakdown_is_stored() -> None:
    title = title_features("A" * 50)
    result = scorer.score(title, _thumb())
    assert result.score == 100.0 and not result.partial
    features = result.features_json(title, _thumb())
    assert features["version"] == "heuristic-v0" and features["thumbnail"]["face_count"] == 2
    assert features["sub_scores"] == {"title_length": 1.0, "faces": 1.0, "face_size": 1.0, "text_coverage": 1.0}


def test_weighted_sum() -> None:
    # title 0.7 (80 chars), 0 faces (0.2), no face (0), no text (0.7)
    result = scorer.score(title_features("x" * 80), _thumb(face_count=0, largest_face_area=0.0, text_area=0.0))
    assert result.score == pytest.approx(100 * (0.35 * 0.7 + 0.30 * 0.2 + 0.20 * 0.0 + 0.15 * 0.7), abs=0.05)


def test_title_only_is_partial_and_rescaled() -> None:
    result = scorer.score(title_features("x" * 80), None)
    assert result.partial and result.score == pytest.approx(70.0)
    assert set(result.sub_scores) == {"title_length"}


def test_zero_weight_features_do_not_move_the_score() -> None:
    plain_title, loud_title = "The story of how we started", "YOU WON'T BELIEVE 3 THINGS?"
    assert len(plain_title) == len(loud_title)
    loud_features = title_features(loud_title)
    assert loud_features.has_digit and loud_features.has_question and loud_features.caps_words == 4  # WON.T -> "WON" + "T"; single letters do not count
    plain = scorer.score(title_features(plain_title), _thumb())
    loud = scorer.score(loud_features, _thumb(brightness=0.9, saturation=0.9, text_boxes=7))
    assert plain == loud


def test_architecture_notes_record_the_formula() -> None:
    notes = scorer.architecture_notes()
    assert "0.35" in notes and "has_digit" in notes and "partial" in notes


# ---- thumbnail features (pure parts) ---------------------------------------------


def test_fit_size_scales_down_only() -> None:
    assert fit_size(1920, 1080) == (1280, 720)
    assert fit_size(640, 360) == (640, 360)
    assert fit_size(1000, 1000) == (720, 720)


def test_union_area_handles_overlap_and_clipping() -> None:
    assert union_area([], 100, 100) == 0.0
    assert union_area([Box(0, 0, 50, 50), Box(25, 25, 50, 50)], 100, 100) == pytest.approx(0.4375)
    assert union_area([Box(-10, -10, 20, 20)], 100, 100) == pytest.approx(0.01)


def test_image_stats_on_solid_colours() -> None:
    assert image_stats(np.zeros((10, 10, 3), np.uint8)) == pytest.approx((0.0, 0.0, 0.0))
    white = image_stats(np.full((10, 10, 3), 255, np.uint8))
    assert white[0] == pytest.approx(1.0) and white[1] == 0.0 and white[2] == 0.0
    half = np.zeros((10, 10, 3), np.uint8)
    half[:5] = 255
    assert image_stats(half)[1] == pytest.approx(1.0)  # max contrast
    red = np.zeros((10, 10, 3), np.uint8)
    red[:, :, 2] = 255
    assert image_stats(red)[2] == pytest.approx(1.0)


def test_thumbnail_features_from_boxes() -> None:
    f = thumbnail_features(_blank(200, 100), [Box(0, 0, 20, 20), Box(50, 0, 40, 50)], [Box(0, 50, 100, 50)])
    assert (f.face_count, f.largest_face_area, f.text_area, f.text_boxes) == (2, 0.1, 0.25, 1)
    assert f.aspect_ratio == 2.0


# ---- validation ------------------------------------------------------------------


def test_sniff_type() -> None:
    assert sniff_type(_png(_blank(4, 4))) == "image/png"
    ok, jpg = cv2.imencode(".jpg", _blank(4, 4))
    assert sniff_type(jpg.tobytes()) == "image/jpeg"
    ok, webp = cv2.imencode(".webp", _blank(4, 4))
    assert sniff_type(webp.tobytes()) == "image/webp"
    assert sniff_type(b"GIF89a....") is None


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "empty"),
        (b"GIF89a" + b"0" * 100, "only JPEG, PNG and WebP"),
        (b"\x89PNG\r\n\x1a\n" + b"garbage", "could not be decoded"),
        (b"\xff\xd8\xff" + b"0" * (2 * 1024 * 1024), "larger than 2 MB"),
    ],
)
def test_validate_rejects(data: bytes, message: str) -> None:
    with pytest.raises(InvalidThumbnailError, match=message):
        validate_thumbnail(data)


def test_missing_model_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DetectorModelMissingError, match="SHORTS_FACE_MODEL_PATH"):
        analyse_thumbnail(_png(_blank()), tmp_path / "nope.onnx", TEXT_MODEL)


# ---- thumbnail features (real detectors) -----------------------------------------


@needs_models
def test_blank_image_has_no_faces_or_text() -> None:
    f = analyse_thumbnail(_png(_blank()), FACE_MODEL, TEXT_MODEL)
    assert (f.face_count, f.largest_face_area, f.text_area, f.text_boxes) == (0, 0.0, 0.0, 0)
    assert (f.width, f.height) == (1280, 720)


@needs_models
def test_big_text_is_detected() -> None:
    f = analyse_thumbnail(_png(_with_text()), FACE_MODEL, TEXT_MODEL)
    assert f.text_boxes >= 1 and 0.05 < f.text_area < 0.6 and f.face_count == 0


@needs_models
def test_face_fixture_is_detected() -> None:
    f = analyse_thumbnail((FIXTURES / "face.jpg").read_bytes(), FACE_MODEL, TEXT_MODEL)
    assert f.face_count >= 1 and 0.01 < f.largest_face_area < 0.9


@needs_models
def test_large_images_are_scaled_to_fit() -> None:
    f = analyse_thumbnail(_png(_blank(2560, 1440)), FACE_MODEL, TEXT_MODEL)
    assert (f.width, f.height) == (1280, 720)


# ---- endpoints --------------------------------------------------------------------


async def _upload(db: AsyncSession) -> Upload:
    upload = Upload(filename="ep.mp4", storage_path="fake-key", duration_seconds=600.0)
    db.add(upload)
    await db.commit()
    return upload


async def _create(client, headers, upload_id: int, title: str, image: bytes | None = None):
    files = {"thumbnail": ("t.png", image, "image/png")} if image is not None else None
    return await client.post(f"/uploads/{upload_id}/ctr-candidates", data={"title": title}, files=files, headers=headers)


async def test_endpoints_require_auth(client, db_session) -> None:
    upload = await _upload(db_session)
    assert (await client.get(f"/uploads/{upload.id}/ctr-candidates")).status_code == 401
    assert (await client.post(f"/uploads/{upload.id}/ctr-candidates", data={"title": "x"})).status_code == 401
    assert (await client.delete("/ctr-candidates/1")).status_code == 401


async def test_create_list_thumbnail_and_delete(client, auth_headers, db_session) -> None:
    upload = await _upload(db_session)
    image = _png(_blank(64, 36))
    r = await _create(client, auth_headers, upload.id, "  First   title ", image)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["candidate_title"] == "First title" and body["has_thumbnail"] and body["group"] == "unscored"
    assert (await _create(client, auth_headers, upload.id, "Title only")).status_code == 201

    listed = (await client.get(f"/uploads/{upload.id}/ctr-candidates", headers=auth_headers)).json()
    assert [c["candidate_title"] for c in listed] == ["First title", "Title only"]

    thumb = await client.get(f"/ctr-candidates/{body['id']}/thumbnail", headers=auth_headers)
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/png" and thumb.content == image
    no_thumb = listed[1]["id"]
    assert (await client.get(f"/ctr-candidates/{no_thumb}/thumbnail", headers=auth_headers)).status_code == 404

    row = await db_session.get(CtrPrediction, body["id"])
    path = storage.path_for(row.candidate_thumbnail_path)
    assert path.is_file()
    assert (await client.delete(f"/ctr-candidates/{body['id']}", headers=auth_headers)).status_code == 204
    assert not path.exists()


@pytest.mark.parametrize(
    "title, image, message",
    [
        ("   ", None, "empty"),
        ("x" * 101, None, "over 100"),
        ("ok", b"GIF89a0000", "only JPEG, PNG and WebP"),
    ],
)
async def test_create_validation(client, auth_headers, db_session, title, image, message) -> None:
    upload = await _upload(db_session)
    r = await _create(client, auth_headers, upload.id, title, image)
    assert r.status_code == 422 and message in r.text


async def test_at_most_ten_candidates(client, auth_headers, db_session) -> None:
    upload = await _upload(db_session)
    for i in range(10):
        assert (await _create(client, auth_headers, upload.id, f"t{i}")).status_code == 201
    r = await _create(client, auth_headers, upload.id, "eleventh")
    assert r.status_code == 422 and "at most 10" in r.text


async def test_unknown_upload_is_404(client, auth_headers) -> None:
    assert (await client.get("/uploads/999/ctr-candidates", headers=auth_headers)).status_code == 404


async def test_edit_clears_the_score(client, auth_headers, db_session) -> None:
    upload = await _upload(db_session)
    cid = (await _create(client, auth_headers, upload.id, "Old title")).json()["id"]
    row = await db_session.get(CtrPrediction, cid)
    row.predicted_score, row.features, row.partial = 70.0, {"x": 1}, True
    await db_session.commit()

    same = await client.patch(f"/ctr-candidates/{cid}", data={"title": "Old title"}, headers=auth_headers)
    assert same.json()["predicted_score"] == 70.0  # no change, no reset
    r = await client.patch(f"/ctr-candidates/{cid}", data={"title": "New title"}, headers=auth_headers)
    assert r.status_code == 200
    assert (r.json()["candidate_title"], r.json()["predicted_score"], r.json()["features"]) == ("New title", None, None)

    r = await client.patch(
        f"/ctr-candidates/{cid}", files={"thumbnail": ("t.png", _png(_blank(8, 8)), "image/png")}, headers=auth_headers
    )
    assert r.json()["has_thumbnail"]
    await db_session.refresh(row)
    first = storage.path_for(row.candidate_thumbnail_path)
    r = await client.patch(f"/ctr-candidates/{cid}", data={"remove_thumbnail": "true"}, headers=auth_headers)
    assert not r.json()["has_thumbnail"] and not first.exists()


async def test_from_show_notes_dedupes_and_needs_a_draft(client, auth_headers, db_session) -> None:
    upload = await _upload(db_session)
    url = f"/uploads/{upload.id}/ctr-candidates/from-show-notes"
    assert (await client.post(url, headers=auth_headers)).status_code == 404

    await _create(client, auth_headers, upload.id, "Already here")
    db_session.add(
        ShowNotesDraft(
            upload_id=upload.id,
            title_suggestions=["already HERE", "A brand new title", "x" * 120, "A brand new title"],
            description="d",
            tags=[],
            summary="s",
            language="en",
            llm_model_used="m",
        )
    )
    await db_session.commit()
    listed = (await client.post(url, headers=auth_headers)).json()
    assert [(c["candidate_title"], c["source"]) for c in listed] == [
        ("Already here", "manual"),
        ("A brand new title", "show_notes"),
    ]
    again = (await client.post(url, headers=auth_headers)).json()
    assert len(again) == 2


async def test_ranking_groups_and_ties() -> None:
    from app.routers.ctr import ranked

    def c(i, score, partial=False):
        return CtrPrediction(id=i, upload_id=1, candidate_title=f"t{i}", predicted_score=score, partial=partial, source="manual")

    out = ranked([c(1, None), c(2, 50, True), c(3, 80), c(4, 90), c(5, 80), c(6, 60, True)])
    assert [(o.id, o.group, o.rank) for o in out] == [
        (4, "full", 1),
        (3, "full", 2),
        (5, "full", 2),  # a tie is a tie
        (6, "partial", 1),
        (2, "partial", 2),
        (1, "unscored", None),
    ]


async def test_deleting_the_upload_removes_thumbnail_files(client, auth_headers, db_session) -> None:
    upload = await _upload(db_session)
    cid = (await _create(client, auth_headers, upload.id, "t", _png(_blank(8, 8)))).json()["id"]
    row = await db_session.get(CtrPrediction, cid)
    path = storage.path_for(row.candidate_thumbnail_path)
    assert (await client.delete(f"/uploads/{upload.id}", headers=auth_headers)).status_code == 204
    assert not path.exists()
    db_session.expunge_all()
    assert await db_session.get(CtrPrediction, cid) is None


# ---- job handler ------------------------------------------------------------------


async def _run_job(db, session_factory, upload, raises=None) -> Job:
    job = Job(upload_id=upload.id, tool_type="ctr")
    db.add(job)
    await db.commit()
    if raises is None:
        await process_job({"session_factory": session_factory}, job.id)
    else:
        with pytest.raises(raises):
            await process_job({"session_factory": session_factory}, job.id)
    await db.refresh(job)
    return job


async def _candidates(db: AsyncSession, upload: Upload) -> list[CtrPrediction]:
    db.expunge_all()
    return list(
        (await db.execute(select(CtrPrediction).where(CtrPrediction.upload_id == upload.id).order_by(CtrPrediction.id))).scalars()
    )


async def test_no_candidates_fails_with_a_message(db_session, session_factory: async_sessionmaker[AsyncSession]) -> None:
    upload = await _upload(db_session)
    job = await _run_job(db_session, session_factory, upload, raises=NoCandidatesError)
    assert job.status == "failed" and "add candidates" in job.error_message


async def test_title_only_candidates_score_without_models(
    db_session, session_factory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No thumbnail -> the detectors are never needed, even when missing.
    monkeypatch.setattr(get_settings(), "ctr_text_model_path", tmp_path / "missing.onnx")
    upload = await _upload(db_session)
    db_session.add_all([CtrPrediction(upload_id=upload.id, candidate_title="x" * 50),
                        CtrPrediction(upload_id=upload.id, candidate_title="x" * 80)])
    await db_session.commit()
    job = await _run_job(db_session, session_factory, upload)
    assert job.status == "succeeded", job.error_message
    rows = await _candidates(db_session, upload)
    assert [(r.predicted_score, r.partial) for r in rows] == [(100.0, True), (70.0, True)]
    version = (await db_session.execute(select(ModelVersion))).scalar_one()
    assert (version.model_name, version.version_label, version.is_live) == ("ctr", "heuristic-v0", True)
    assert {r.model_version_id for r in rows} == {version.id} and all(r.scored_at for r in rows)

    await _run_job(db_session, session_factory, upload)  # idempotent re-run, same version
    assert len((await db_session.execute(select(ModelVersion))).scalars().all()) == 1
    assert [r.predicted_score for r in await _candidates(db_session, upload)] == [100.0, 70.0]


async def test_missing_model_fails_fast_when_a_thumbnail_exists(
    db_session, session_factory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(get_settings(), "ctr_text_model_path", tmp_path / "missing.onnx")
    upload = await _upload(db_session)
    key = storage.save(storage.make_key("t.png"), _png(_blank(8, 8)))
    db_session.add(CtrPrediction(upload_id=upload.id, candidate_title="t", candidate_thumbnail_path=key))
    await db_session.commit()
    job = await _run_job(db_session, session_factory, upload, raises=DetectorModelMissingError)
    assert "CTR_TEXT_MODEL_PATH" in job.error_message


@needs_models
async def test_real_scoring_ranks_and_isolates_a_broken_thumbnail(db_session, session_factory) -> None:
    upload = await _upload(db_session)
    face = storage.save(storage.make_key("face.jpg"), (FIXTURES / "face.jpg").read_bytes())
    blank = storage.save(storage.make_key("blank.png"), _png(_blank()))
    db_session.add_all(
        [
            CtrPrediction(upload_id=upload.id, candidate_title="How we built it in a single year", candidate_thumbnail_path=face),
            CtrPrediction(upload_id=upload.id, candidate_title="How we built it in a single year", candidate_thumbnail_path=blank),
            CtrPrediction(upload_id=upload.id, candidate_title="Broken", candidate_thumbnail_path="gone.png"),
        ]
    )
    await db_session.commit()
    job = await _run_job(db_session, session_factory, upload)
    assert job.status == "succeeded", job.error_message
    with_face, without, broken = await _candidates(db_session, upload)
    assert with_face.predicted_score > without.predicted_score
    assert with_face.features["thumbnail"]["face_count"] >= 1 and not with_face.partial
    assert broken.predicted_score is None and "unreadable" in broken.features["error"]
