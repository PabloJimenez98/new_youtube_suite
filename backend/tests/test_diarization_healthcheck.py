"""Startup healthcheck (M4, docs/decisions/0008 §5): the real pipeline on
the bundled 2-speaker clip, plus the degraded path with a fake pipeline.

The real test is what catches the silent failure modes (token missing,
wrong license accepted, empty output) -- it needs HF_TOKEN in
backend/.env, like the running worker does.
"""

import json
from collections.abc import Iterator

import pytest

from app.diarization import pipeline
from app.diarization.assign import Turn

GROUND_TRUTH = json.loads(pipeline.HEALTHCHECK_CLIP.with_suffix(".json").read_text())


@pytest.fixture(autouse=True)
def _reset_health() -> Iterator[None]:
    yield
    pipeline.health.error = None
    pipeline.health.checked = False


def test_real_pipeline_finds_both_speakers_on_the_healthcheck_clip() -> None:
    turns = pipeline.diarize(pipeline.HEALTHCHECK_CLIP)

    assert turns, "empty diarization output -- check HF_TOKEN / model license"
    assert {t.speaker for t in turns} == {"SPEAKER_00", "SPEAKER_01"}
    # Relabelled by first appearance: the clip opens with speaker A.
    assert turns[0].speaker == "SPEAKER_00"
    # Each ground-truth turn's midpoint lands on the matching speaker.
    mapping = {"A": "SPEAKER_00", "B": "SPEAKER_01"}
    for truth in GROUND_TRUTH["turns"]:
        mid = (truth["start"] + truth["end"]) / 2
        covering = [t for t in turns if t.start <= mid <= t.end]
        assert covering and covering[0].speaker == mapping[truth["speaker"]], truth


async def test_startup_healthcheck_passes_with_the_real_pipeline() -> None:
    await pipeline.startup_healthcheck()

    assert pipeline.health.checked
    assert pipeline.health.error is None
    pipeline.ensure_available()  # doesn't raise


@pytest.mark.parametrize(
    ("fake", "expected"),
    [
        (lambda _p: [], "0 turns / 0 speaker(s)"),
        (lambda _p: [Turn(0.0, 37.0, "SPEAKER_00")], "1 turns / 1 speaker(s)"),
    ],
)
async def test_degraded_when_output_is_empty_or_single_speaker(
    monkeypatch: pytest.MonkeyPatch, fake, expected: str
) -> None:
    monkeypatch.setattr(pipeline, "diarize", fake)

    await pipeline.startup_healthcheck()  # must not raise

    assert pipeline.health.checked
    assert expected in pipeline.health.error
    with pytest.raises(pipeline.DiarizationUnavailableError, match="Diarization unavailable"):
        pipeline.ensure_available()


async def test_degraded_when_the_pipeline_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_p):
        raise pipeline.DiarizationUnavailableError("HF_TOKEN is not set")

    monkeypatch.setattr(pipeline, "diarize", _boom)

    await pipeline.startup_healthcheck()

    assert pipeline.health.error == "DiarizationUnavailableError: HF_TOKEN is not set"


# ---- GPU memory release (regression: CUDA OOM on the next transcription) ----


class _FakeAnnotation:
    def itertracks(self, yield_label: bool = False):
        return iter(())


class _FakeOutput:
    exclusive_speaker_diarization = _FakeAnnotation()


@pytest.mark.parametrize("fails", [False, True])
def test_diarize_releases_torch_cached_gpu_memory(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """After every run -- success or failure -- torch's cached allocator
    blocks must be handed back, or a following Whisper job OOMs on 3 GB."""
    released: list[bool] = []

    def _fake_pipeline(_audio):
        if fails:
            raise RuntimeError("inference blew up")
        return _FakeOutput()

    monkeypatch.setattr(pipeline, "get_pipeline", lambda: _fake_pipeline)
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pipeline.torch.cuda, "empty_cache", lambda: released.append(True))

    if fails:
        with pytest.raises(RuntimeError, match="inference blew up"):
            pipeline.diarize(pipeline.HEALTHCHECK_CLIP)
    else:
        assert pipeline.diarize(pipeline.HEALTHCHECK_CLIP) == []

    assert released == [True]
