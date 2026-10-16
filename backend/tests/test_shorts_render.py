"""Framing + render with real ffmpeg on synthetic video (docs/decisions/0013 §2, §4).

The face-detection tests skip when the YuNet model isn't downloaded
(see README, "Shorts").
"""

import subprocess
from pathlib import Path

import pytest

from app.config import get_settings
from app.shorts import captions, framing, render
from app.shorts.framing import Plan, Shot
from app.shorts.selection import TimedWord
from tests.fixtures import FACE_IMAGE

W, H, FPS = 1280, 720, 25.0
MODEL = get_settings().shorts_face_model_path
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_file(), reason="YuNet model not downloaded")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="module")
def testsrc(tmp_path_factory) -> Path:
    """8 s of 1280x720 testsrc + a sine tone."""
    out = tmp_path_factory.mktemp("src") / "testsrc.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=25:duration=8",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out))
    return out


@pytest.fixture(scope="module")
def face_then_pattern(tmp_path_factory) -> Path:
    """3 s of the face at x 900-1156 on grey, then a hard cut to 3 s of testsrc (no face)."""
    out = tmp_path_factory.mktemp("src") / "face.mp4"
    ffmpeg("-f", "lavfi", "-i", f"color=c=gray:size={W}x{H}:rate=25:duration=3", "-loop", "1", "-i", str(FACE_IMAGE),
           "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=25:duration=3",
           "-f", "lavfi", "-i", "sine=frequency=330:duration=6",
           "-filter_complex", "[0][1]overlay=900:200:shortest=1[a];[a][2]concat=n=2:v=1:a=0[v]",
           "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", str(out))
    return out


def static_plan(aspect: str, modes: list[tuple[float, float, str]], x: int = 400) -> Plan:
    crop_w, out_w, out_h = framing.geometry(W, H, aspect)
    shots = [Shot(s, e, m) for s, e, m in modes]
    track = [{"t": round(t / 5, 3), "x": x, "y": 0, "w": crop_w, "h": H}
             for s, e, m in modes if m == "crop" for t in range(int(s * 5), int(e * 5))]
    return Plan(W, H, crop_w, out_w, out_h, shots, track)


def test_filter_graph_single_run_and_mixed(tmp_path: Path) -> None:
    single = render.filter_graph(static_plan("9:16", [(0, 4, "crop")]), FPS, tmp_path, None)
    assert single.startswith("[0:v]sendcmd=f=") and single.endswith("setsar=1[vout]")
    assert "crop=w=404:h=720:x=400:y=0,scale=720:1280" in single
    mixed = render.filter_graph(static_plan("9:16", [(0, 2, "crop"), (2, 4, "fit")]), FPS, tmp_path,
                                tmp_path / "c.ass")
    assert mixed.startswith("[0:v]split=2[s0][s1]")
    assert "[s0]trim=start=0.0000:end=2.0000,setpts=PTS-STARTPTS,sendcmd" in mixed
    assert "[s1]trim=start=2.0000,setpts=PTS-STARTPTS,split[bg][fg]" in mixed
    assert mixed.endswith(f"concat=n=2:v=1:a=0,ass='{tmp_path / 'c.ass'}'[vout]")
    assert (tmp_path / "crop0.cmd").read_text().count("crop x") == 50


@pytest.mark.parametrize(
    "aspect,modes,size",
    [
        ("9:16", [(0, 4, "crop")], (720, 1280)),
        ("9:16", [(0, 4, "fit")], (720, 1280)),
        ("4:5", [(0, 2, "crop"), (2, 4, "fit")], (720, 900)),
        ("16:9", [(0, 4, "fit")], (1280, 720)),
    ],
)
def test_render_produces_a_valid_clip(testsrc: Path, tmp_path: Path, aspect, modes, size) -> None:
    p = static_plan(aspect, modes)
    if aspect == "16:9":
        p.crop_w = 0
    words = [TimedWord(2 + i * 0.4, 2.3 + i * 0.4, w) for i, w in enumerate("hola qué tal estás hoy".split())]
    ass = captions.build_ass(words, 2.0, 6.0, p.out_w, p.out_h)
    out = tmp_path / "clip.mp4"
    info = render.render(testsrc, 2.0, 6.0, p, FPS, ass, out)
    assert (info["width"], info["height"]) == size
    assert info["duration"] == pytest.approx(4.0, abs=0.5) and info["has_audio"]


def test_render_rejects_a_wrong_duration(testsrc: Path, tmp_path: Path) -> None:
    with pytest.raises(render.RenderError, match="expected"):
        render.render(testsrc, 6.0, 12.0, static_plan("9:16", [(0, 6, "crop")]), FPS, None, tmp_path / "c.mp4")


@needs_model
def test_yunet_finds_the_fixture_face() -> None:
    import cv2

    img = cv2.imread(str(FACE_IMAGE))
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (img.shape[1], img.shape[0]), framing.FACE_SCORE, 0.3, 50)
    _, faces = det.detect(img)
    assert faces is not None and len(faces) == 1


@needs_model
def test_analyse_and_render_follow_the_face_then_fit_after_the_cut(face_then_pattern: Path, tmp_path: Path) -> None:
    a = framing.analyse(face_then_pattern, 0.0, 6.0, Path(MODEL))
    assert a.fps == 25.0 and len(a.frame_times) == 150
    p = framing.plan(a, 6.0, "9:16")
    assert [s.mode for s in p.shots] == ["crop", "fit"]
    assert p.shots[0].end == pytest.approx(3.0, abs=0.05)
    crop = [q for q in p.track if q["w"] == 404]
    # The face is at x 900-1156 in source pixels (detected around 990-1035): the crop centres on it.
    assert all(800 <= q["x"] + 202 <= 1100 for q in crop)
    info = render.render(face_then_pattern, 0.0, 6.0, p, a.fps, None, tmp_path / "f.mp4")
    assert (info["width"], info["height"]) == (720, 1280)


def test_analyse_fails_fast_without_the_model(testsrc: Path, tmp_path: Path) -> None:
    with pytest.raises(framing.FaceModelMissingError):
        framing.analyse(testsrc, 0, 1, tmp_path / "missing.onnx")
