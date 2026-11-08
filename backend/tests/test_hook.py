"""Hook-strength scorer v0 (M14, docs/decisions/0020): the opening window,
language detection, each feature extractor (real spaCy ES/EN models), the
emotion model, the scorer shapes, the `hook` job handler and `ClipOut.hook`."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.hook import features, nlp, scorer
from app.hook.features import HookFeatures
from app.hook.window import Opening, detect_language, opening
from app.models.clip import Clip
from app.models.hook import HookScore
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.show_notes import ShowNotesDraft
from app.models.transcript import TranscriptSegment, TranscriptWord
from app.models.upload import Upload
from app.shorts.selection import TimedWord
from app.worker.handlers.hook import NoClipsError, episode_title
from app.worker.tasks import process_job


def _spacy_installed() -> bool:
    import spacy.util

    return all(spacy.util.is_package(m) for m in nlp.SPACY_MODELS.values())


def _emotion_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(get_settings().hook_emotion_model, "config.json"), str)
    except Exception:
        return False


needs_spacy = pytest.mark.skipif(not _spacy_installed(), reason="spaCy es/en small models not installed")
needs_emotion = pytest.mark.skipif(not _emotion_cached(), reason="xlm-emo-t not in the HF cache")


def _doc(text: str, lang: str):
    return nlp.spacy_model(lang)(text)


def _w(start: float, text: str) -> TimedWord:
    return TimedWord(start, start + 0.3, text)


# ---- window ------------------------------------------------------------------------


def test_opening_slices_start_window_and_clip_end() -> None:
    words = [_w(9.0, "antes"), _w(10.2, "Hola,"), _w(11.0, "¿qué"), _w(24.9, "tal?"), _w(25.0, "fuera"),
             TimedWord(None, None, "x")]
    o = opening(words, 10.0, 60.0)
    assert o.text == "Hola, ¿qué tal?" and o.word_count == 3 and o.length_s == 15.0
    assert o.first_word_offset == pytest.approx(0.2)
    short = opening(words, 10.0, 20.0)  # clip shorter than the window
    assert short.text == "Hola, ¿qué" and short.length_s == 10.0


def test_opening_empty_window() -> None:
    o = opening([_w(100.0, "lejos")], 0.0, 30.0)
    assert o == Opening(text="", word_count=0, first_word_offset=None, length_s=15.0)


def test_punctuation_only_tokens_are_not_words() -> None:
    o = opening([_w(0.1, "Bueno"), _w(0.4, "..."), _w(0.6, "vale")], 0.0, 30.0)
    assert o.word_count == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bueno, pues yo creo que esto es lo que hay que hacer porque ahora mismo no tenemos nada", ("es", False)),
        ("Well I think this is what we have to do because right now we have nothing", ("en", False)),
        ("Der Hund läuft schnell über die grüne Wiese", ("en", True)),
        ("", ("en", True)),
    ],
)
def test_detect_language(text: str, expected: tuple[str, bool]) -> None:
    assert detect_language(text) == expected


# ---- features ----------------------------------------------------------------------


def test_lead_silence_and_pace() -> None:
    o = Opening(text="a b c", word_count=30, first_word_offset=0.8, length_s=15.0)
    assert features.lead_silence(o) == 0.8 and features.words_per_second(o) == 2.0
    empty = Opening(text="", word_count=0, first_word_offset=None, length_s=12.0)
    assert features.lead_silence(empty) == 12.0 and features.words_per_second(empty) == 0.0


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [("¿Sabes qué pasó?", True), ("Sabes qué pasó?", True), ("¿Sabes qué pasó", True), ("Te cuento qué pasó.", False)],
)
def test_is_question(sentence: str, expected: bool) -> None:
    assert features.is_question(sentence) is expected


def test_punct_counts_spanish_pairs_once() -> None:
    assert features.punct_count("¿Qué? ¿En serio?", "?", "¿") == 2
    assert features.punct_count("¡Vaya! Increíble!", "!", "¡") == 2
    assert features.punct_count("Nada.", "!", "¡") == 0


def test_curiosity_matches_whole_phrases_only() -> None:
    assert features.curiosity("Pero resulta que nadie lo sabía", "es") == 2
    assert features.curiosity("La perorata siguió", "es") == 0  # "pero" inside a word
    assert features.curiosity("It turns out the reason why is simple, but", "en") == 3
    assert features.curiosity("A butterfly", "en") == 0


@needs_spacy
def test_unresolved_demonstratives_pronouns_not_determiners() -> None:
    assert features.unresolved_demonstratives(_doc("Esto es lo que nadie te cuenta.", "es")) == 1
    assert features.unresolved_demonstratives(_doc("Este libro es bueno.", "es")) == 0
    assert features.unresolved_demonstratives(_doc("This is what nobody tells you.", "en")) == 1
    assert features.unresolved_demonstratives(_doc("This book is good.", "en")) == 0


@needs_spacy
@pytest.mark.parametrize(
    ("text", "lang", "imperatives", "second_person"),
    [
        ("Mira, escucha bien.", "es", 2, 0),  # "escucha" is mis-tagged NOUN; the lexicon catches it
        ("¿Sabes lo que te digo?", "es", 0, 2),  # "Sabes" (Person=2), "te"
        ("Usted lo sabe.", "es", 0, 1),
        ("Listen to this.", "en", 1, 0),
        ("Do you know what your problem is?", "en", 0, 2),
        ("I know the answer.", "en", 0, 0),
    ],
)
def test_imperatives_and_second_person(text: str, lang: str, imperatives: int, second_person: int) -> None:
    assert features.imperatives_and_second_person(_doc(text, lang), lang) == (imperatives, second_person)


@needs_spacy
@pytest.mark.parametrize(
    ("text", "lang", "expected"),
    [
        ("Es lo más importante.", "es", 1),
        ("Esto es buenísimo.", "es", 1),
        ("Es el mejor del mundo.", "es", 1),
        ("Es más alto que yo.", "es", 0),  # comparative, not superlative
        ("It's the best and the most important thing.", "en", 2),
        ("It's better than that.", "en", 0),
    ],
)
def test_superlatives(text: str, lang: str, expected: int) -> None:
    assert features.superlatives(_doc(text, lang), lang) == expected


@needs_spacy
def test_numerics() -> None:
    assert features.numerics(_doc("Hay 3 formas de hacerlo en 2026.", "es")) == 2
    assert features.numerics(_doc("Here are three ways.", "en")) == 1
    assert features.numerics(_doc("Nothing here.", "en")) == 0


def test_syllables_and_readability() -> None:
    assert [features.syllables(w, "es") for w in ("Esto", "es", "buenísimo")] == [2, 1, 4]
    assert [features.syllables(w, "en") for w in ("the", "cat", "make", "table")] == [1, 1, 1, 2]
    assert features.readability(["Esto", "es", "buenísimo"], 1, "es") == round(206.84 - 60 * 7 / 3 - 1.02 * 3, 2)
    assert features.readability(["The", "cat", "sat"], 1, "en") == round(206.835 - 1.015 * 3 - 84.6, 2)
    assert features.readability([], 0, "en") is None


def test_title_overlap_ignores_stopwords() -> None:
    assert features.title_overlap("La inflación en España", "Inflación y salarios en España", "es") == round(2 / 3, 4)
    assert features.title_overlap("de la en el", "de la en el", "es") == 0.0
    assert features.title_overlap("algo", None, "es") == 0.0


@needs_spacy
def test_extract_end_to_end_spanish() -> None:
    text = "¿Sabes por qué esto es lo más importante? Mira, hay 3 razones. ¡Increíble!"
    o = Opening(text=text, word_count=13, first_word_offset=0.2, length_s=5.0)
    f = features.extract(o, _doc(text, "es"), "es", {"anger": 0.1, "fear": 0.1, "joy": 0.7, "sadness": 0.1},
                         "Las 3 razones más importantes")
    assert f.first_is_question and f.lead_silence_s == 0.2 and f.words_per_second == 2.6
    assert f.sentences == 3 and f.questions == 1 and f.exclamations == 1
    assert f.imperatives == 1 and f.second_person == 1 and f.superlatives == 1 and f.numerics == 1
    assert f.first_unresolved_demonstratives == 1 and f.emotion_max == 0.7 and f.title_overlap > 0


@needs_spacy
def test_extract_empty_window() -> None:
    o = Opening(text="", word_count=0, first_word_offset=None, length_s=15.0)
    f = features.extract(o, _doc("", "es"), "es", nlp.emotion(""), "Título")
    assert (f.words, f.sentences, f.readability, f.first_is_question) == (0, 0, None, False)
    assert f.emotion == {"anger": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0}


# ---- emotion -------------------------------------------------------------------------


@needs_emotion
@pytest.mark.parametrize(
    ("text", "label"),
    [("¡Qué alegría, esto es maravilloso, estoy feliz!", "joy"),
     ("Estoy furioso, esto es una vergüenza y me da rabia.", "anger")],
)
def test_real_emotion_model(text: str, label: str) -> None:
    probs = nlp.emotion(text)
    assert set(probs) == set(nlp.EMOTION_LABELS)
    assert max(probs, key=probs.get) == label and sum(probs.values()) == pytest.approx(1.0, abs=0.01)


# ---- scorer --------------------------------------------------------------------------


def _feats(**kw) -> HookFeatures:
    base = dict(
        lead_silence_s=0.0, words_per_second=3.0, first_is_question=False, first_curiosity=0,
        first_unresolved_demonstratives=0, second_person=0, imperatives=0, words=40, sentences=3,
        avg_sentence_words=13.3, readability=70.0, questions=0, exclamations=0, curiosity_connectives=0,
        superlatives=0, numerics=0, entity_density=0.0, emotion={}, emotion_max=0.0, title_overlap=0.0,
    )
    return HookFeatures(**(base | kw))


def test_weights_sum_to_one() -> None:
    assert sum(scorer.WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("silence", "expected"), [(0.0, 1.0), (0.3, 1.0), (1.15, 0.5), (2.0, 0.0), (9.0, 0.0)]
)
def test_start_shape(silence: float, expected: float) -> None:
    assert scorer.score(_feats(lead_silence_s=silence)).sub_scores["start"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("wps", "expected"),
    [(0.5, 0.0), (1.0, 0.0), (1.65, 0.5), (2.3, 1.0), (3.0, 1.0), (3.8, 1.0), (4.4, 0.8), (5.0, 0.6), (7.0, 0.6)],
)
def test_pace_shape(wps: float, expected: float) -> None:
    assert scorer.score(_feats(words_per_second=wps)).sub_scores["pace"] == pytest.approx(expected)


def test_framing_and_address() -> None:
    assert scorer.framing_score(_feats(first_is_question=True, first_curiosity=1)) == 1.0
    assert scorer.framing_score(_feats(first_curiosity=1)) == 0.7
    assert scorer.framing_score(_feats(first_unresolved_demonstratives=2)) == 0.7
    assert scorer.framing_score(_feats()) == 0.3
    assert [scorer.address_score(_feats(second_person=n)) for n in (0, 1, 2, 5)] == [0.3, 0.8, 1.0, 1.0]
    assert scorer.address_score(_feats(second_person=1, imperatives=1)) == 1.0


def test_weighted_sum() -> None:
    best = scorer.score(_feats(first_is_question=True, imperatives=2))
    assert best.score == 100.0
    s = scorer.score(_feats(lead_silence_s=1.15))  # start 0.5, pace 1, framing 0.3, address 0.3
    assert s.score == round((0.30 * 0.5 + 0.25 * 1 + 0.25 * 0.3 + 0.20 * 0.3) * 100, 1)


def test_zero_weight_features_leave_the_score_unchanged() -> None:
    a = scorer.score(_feats())
    b = scorer.score(_feats(words=3, sentences=9, readability=-20.0, questions=4, exclamations=5, superlatives=3,
                            numerics=7, entity_density=0.9, emotion={"joy": 1.0}, emotion_max=1.0,
                            title_overlap=1.0, curiosity_connectives=6))
    assert a == b


def test_empty_window_scores_without_error() -> None:
    s = scorer.score(_feats(lead_silence_s=15.0, words_per_second=0.0))
    assert s.sub_scores["start"] == 0.0 and s.sub_scores["pace"] == 0.0 and s.score == 13.5


# ---- handler -------------------------------------------------------------------------


async def _upload(db: AsyncSession, filename: str = "1X09_la-entrevista.mp4") -> Upload:
    upload = Upload(filename=filename, storage_path="k")
    db.add(upload)
    await db.commit()
    return upload


async def _transcript(db: AsyncSession, upload: Upload, sentences: list[tuple[float, str]]) -> None:
    """One segment per (start, text), words 0.3 s apart."""
    for i, (start, text) in enumerate(sentences):
        toks = text.split()
        seg = TranscriptSegment(upload_id=upload.id, segment_index=i, start_seconds=start,
                                end_seconds=start + 0.3 * len(toks), text=text)
        seg.words = [TranscriptWord(word_index=k, word=" " + t, start_seconds=start + 0.3 * k,
                                    end_seconds=start + 0.3 * k + 0.25) for k, t in enumerate(toks)]
        db.add(seg)
    await db.commit()


async def _clips(db: AsyncSession, upload: Upload, starts: list[float]) -> list[Clip]:
    rows = [
        Clip(upload_id=upload.id, rank=r, start_seconds=s, end_seconds=s + 30, score=0.5, peak_window_index=r,
             aspect_ratio="9:16", caption_burned_in=True, storage_path=f"c{upload.id}_{r}.mp4",
             crop_bounding_box_track=[], framing={})
        for r, s in enumerate(starts, start=1)
    ]
    db.add_all(rows)
    db.add(Job(upload_id=upload.id, tool_type="shorts", status="succeeded"))
    await db.commit()
    return rows


async def _run_job(db, session_factory, upload, raises=None) -> Job:
    job = Job(upload_id=upload.id, tool_type="hook")
    db.add(job)
    await db.commit()
    if raises is None:
        await process_job({"session_factory": session_factory}, job.id)
    else:
        with pytest.raises(raises):
            await process_job({"session_factory": session_factory}, job.id)
    await db.refresh(job)
    return job


@pytest.fixture
def fake_emotion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the 1.1 GB model in handler tests; the real one has its own test."""
    monkeypatch.setattr(nlp, "emotion_classifier", lambda: None)
    monkeypatch.setattr(nlp, "emotion", lambda text: {"anger": 0.0, "fear": 0.0, "joy": 1.0 if text else 0.0, "sadness": 0.0})


SPANISH = [
    # Clip 1 at 100 s: question first, direct address, starts at once.
    (100.0, "¿Sabes por qué nadie te cuenta esto sobre la inflación en España y los salarios?"),
    (104.0, "Mira, te lo explico ahora mismo porque es muy importante para todos nosotros."),
    # Clip 2 at 200 s: 1 s of dead air, flat statement.
    (201.0, "El martes fuimos a la tienda y compramos pan para la cena de la semana con mi hermano y luego volvimos a casa."),
    # Clip 3 at 300 s: nothing said in the window.
]


async def test_no_clips_fails_with_a_message(db_session, session_factory) -> None:
    upload = await _upload(db_session)
    job = await _run_job(db_session, session_factory, upload, raises=NoClipsError)
    assert job.status == "failed" and "run Shorts first" in job.error_message


async def test_missing_spacy_model_fails_clearly(db_session, session_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(nlp.SPACY_MODELS, "es", "xx_missing_model_sm")
    upload = await _upload(db_session)
    await _transcript(db_session, upload, SPANISH)
    await _clips(db_session, upload, [100.0])
    job = await _run_job(db_session, session_factory, upload, raises=nlp.HookModelMissingError)
    assert "xx_missing_model_sm" in job.error_message
    assert (await db_session.execute(select(HookScore))).first() is None


async def test_missing_emotion_model_fails_clearly(db_session, session_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nlp, "_emotion", None)
    monkeypatch.setattr(get_settings(), "hook_emotion_model", "/nonexistent/emotion-model")
    upload = await _upload(db_session)
    await _clips(db_session, upload, [100.0])
    job = await _run_job(db_session, session_factory, upload, raises=nlp.HookModelMissingError)
    assert "HOOK_EMOTION_MODEL" in job.error_message


@needs_spacy
async def test_handler_scores_every_clip_and_is_idempotent(db_session, session_factory, fake_emotion) -> None:
    upload = await _upload(db_session)
    await _transcript(db_session, upload, SPANISH)
    clips = await _clips(db_session, upload, [100.0, 200.0, 300.0])
    job = await _run_job(db_session, session_factory, upload)
    assert job.status == "succeeded", job.error_message

    db_session.expunge_all()
    rows = {h.clip_id: h for h in (await db_session.execute(select(HookScore))).scalars()}
    hook1, hook2, hook3 = (rows[c.id] for c in clips)
    assert hook1.score > hook2.score > hook3.score
    fv = hook1.feature_vector
    assert fv["language"] == "es" and fv["title_source"] == "filename" and fv["version"] == "heuristic-v0"
    assert fv["features"]["first_is_question"] and fv["sub_scores"]["framing"] == 1.0
    assert fv["weights"] == scorer.WEIGHTS
    assert hook2.feature_vector["features"]["lead_silence_s"] == pytest.approx(1.0)
    assert hook3.feature_vector["window"]["text"] == "" and hook3.feature_vector["sub_scores"]["pace"] == 0.0
    version = (await db_session.execute(select(ModelVersion))).scalar_one()
    assert (version.model_name, version.version_label, version.is_live) == ("hook", "heuristic-v0", True)
    assert {h.model_version_id for h in rows.values()} == {version.id} and all(h.scored_at for h in rows.values())

    await _run_job(db_session, session_factory, upload)  # re-run: upsert, same version
    db_session.expunge_all()
    again = list((await db_session.execute(select(HookScore))).scalars())
    assert len(again) == 3 and {h.id for h in again} == {h.id for h in rows.values()}
    assert len((await db_session.execute(select(ModelVersion))).scalars().all()) == 1


async def test_episode_title_prefers_show_notes(db_session) -> None:
    upload = await _upload(db_session, "1X09_la-entrevista_final.mp4")
    assert await episode_title(db_session, upload) == ("1X09 la entrevista final", "filename")
    db_session.add(ShowNotesDraft(upload_id=upload.id, title_suggestions=["La entrevista que lo cambió todo"],
                                  description="d", tags=[], summary="s", language="es", llm_model_used="m"))
    await db_session.commit()
    assert await episode_title(db_session, upload) == ("La entrevista que lo cambió todo", "show_notes")


# ---- API -----------------------------------------------------------------------------


async def test_clip_out_hook_null_then_ranked_with_ties(client, auth_headers, db_session) -> None:
    upload = await _upload(db_session)
    clips = await _clips(db_session, upload, [0.0, 50.0, 100.0])
    r = await client.get(f"/uploads/{upload.id}/clips", headers=auth_headers)
    assert [c["hook"] for c in r.json()] == [None, None, None]

    for clip, score in zip(clips[:3], (70.0, 85.0, 70.0)):
        db_session.add(HookScore(clip_id=clip.id, score=score, feature_vector={"sub_scores": {"start": 1.0}}))
    await db_session.commit()
    body = (await client.get(f"/uploads/{upload.id}/clips", headers=auth_headers)).json()
    assert [(c["hook"]["score"], c["hook"]["rank"]) for c in body] == [(70.0, 2), (85.0, 1), (70.0, 2)]
    assert body[0]["hook"]["features"] == {"sub_scores": {"start": 1.0}} and body[0]["hook"]["scored_at"]


async def test_deleting_clips_cascades_hook_scores(db_session) -> None:
    upload = await _upload(db_session)
    (clip,) = await _clips(db_session, upload, [0.0])
    db_session.add(HookScore(clip_id=clip.id, score=50.0, feature_vector={}))
    await db_session.commit()
    await db_session.delete(clip)
    await db_session.commit()
    assert (await db_session.execute(select(HookScore))).first() is None
