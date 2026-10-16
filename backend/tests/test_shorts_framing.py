"""Framing logic with fake detections (docs/decisions/0013 §2). No video."""

import pytest

from app.shorts import framing
from app.shorts.framing import (
    Analysis,
    Face,
    Sample,
    Shot,
    crop_commands,
    geometry,
    pick_face,
    plan,
    runs,
    shots_from_diffs,
    smooth_centres,
)

W, H = 1280, 720
FPS = 25.0


def face_at(cx: float, size: float = 100, score: float = 0.9) -> Face:
    return Face(cx - size / 2, 200, size, size, score)


def analysis(duration: float, cuts: list[float], faces_at) -> Analysis:
    """Frames at 25 fps, a cut distance of 0.4 at `cuts`, samples every 5 frames from `faces_at(t)`."""
    n = int(duration * FPS)
    times = [k / FPS for k in range(n)]
    diffs = [0.4 if any(abs(t - c) < 1e-6 for c in cuts) else 0.02 for t in times]
    samples = [Sample(t, tuple(faces_at(t))) for k, t in enumerate(times) if k % 5 == 0]
    return Analysis(W, H, FPS, times, diffs, samples)


@pytest.mark.parametrize(
    "aspect,expected",
    [("9:16", (404, 720, 1280)), ("1:1", (720, 720, 720)), ("4:5", (576, 720, 900)), ("16:9", (0, 1280, 720))],
)
def test_geometry_per_aspect(aspect: str, expected: tuple[int, int, int]) -> None:
    assert geometry(W, H, aspect) == expected


def test_geometry_caps_the_short_side() -> None:
    assert geometry(3840, 2160, "9:16") == (1216, 1080, 1920)


def test_cuts_split_shots() -> None:
    times = [k / FPS for k in range(100)]
    diffs = [0.4 if k in (25, 60) else 0.03 for k in range(100)]
    assert shots_from_diffs(times, diffs, 4.0) == [(0.0, 1.0), (1.0, 2.4), (2.4, 4.0)]


def test_pick_face_prefers_big_confident_faces_and_penalises_the_edge() -> None:
    small, big = face_at(300, 60), face_at(800, 120)
    assert pick_face((small, big), W, None) is big
    edge = Face(0, 200, 130, 130, 0.9)  # bigger, but touching the frame edge
    assert pick_face((edge, big), W, None) is big
    assert pick_face((), W, None) is None


def test_a_big_unsure_box_near_the_edge_loses_to_a_confident_face() -> None:
    # Episode 8 QA: a blurred foreground hand vs the speaker.
    hand = Face(12, 335, 206, 323, 0.72)
    speaker = Face(575, 124, 137, 179, 0.93)
    assert pick_face((hand, speaker), W, None) is speaker
    assert pick_face((hand, speaker), W, 100) is speaker


def test_pick_face_keeps_the_previous_subject_on_near_ties() -> None:
    a, b = face_at(300, 100, 0.9), face_at(900, 100, 0.88)  # weights within 20 %
    assert pick_face((a, b), W, 880) is b
    assert pick_face((a, b), W, None) is a


def test_smoothing_has_a_dead_zone_and_follows_real_moves() -> None:
    jitter = smooth_centres([500, 510, 495, 505, 500], 404)  # < 5 % of 404 px
    assert all(c == 500 for c in jitter)
    moved = smooth_centres([500, 700, 700, 700, 700, 700, 700, 700, 700, 700], 404)
    assert moved[0] == 500 and 500 < moved[1] < moved[2] < 700 and moved[-1] == pytest.approx(700, abs=10)
    assert smooth_centres([None, None], 404) == []
    assert smooth_centres([None, 600, None], 404) == [600, 600, 600]  # hold over gaps


def test_plan_crops_on_faces_resets_at_cuts_and_clamps() -> None:
    a = analysis(4.0, [2.0], lambda t: [face_at(300)] if t < 2 else [face_at(1270, 60)])
    p = plan(a, 4.0, "9:16")
    assert [(s.start, s.end, s.mode) for s in p.shots] == [(0.0, 2.0, "crop"), (2.0, 4.0, "crop")]
    xs = {q["t"]: q["x"] for q in p.track}
    assert xs[0.0] == 300 - 202 and xs[1.8] == 98  # centred, steady
    assert xs[2.0] == W - 404  # jumps at the cut and is clamped to the right edge
    assert all(q["w"] == 404 and q["h"] == H for q in p.track)


def test_plan_fits_shots_without_enough_faces() -> None:
    a = analysis(2.0, [], lambda t: [face_at(640)] if t < 0.5 else [])
    p = plan(a, 2.0, "9:16")
    assert [s.mode for s in p.shots] == ["fit"]
    assert all(q["w"] == W for q in p.track)


def test_wide_targets_never_crop() -> None:
    p = plan(analysis(1.0, [], lambda t: [face_at(640)]), 1.0, "16:9")
    assert p.crop_w == 0 and [s.mode for s in p.shots] == ["fit"]


def test_runs_merge_same_mode_shots() -> None:
    shots = [Shot(0, 1, "crop"), Shot(1, 2, "crop"), Shot(2, 3, "fit"), Shot(3, 4, "crop")]
    assert runs(shots) == [Shot(0, 2, "crop"), Shot(2, 3, "fit"), Shot(3, 4, "crop")]


def test_crop_commands_interpolate_per_frame_and_restart_per_run() -> None:
    p = framing.Plan(W, H, 404, 720, 1280, [Shot(0, 1, "fit"), Shot(1, 2, "crop")],
                     [{"t": 1.0, "x": 100, "y": 0, "w": 404, "h": H}, {"t": 1.2, "x": 200, "y": 0, "w": 404, "h": H}])
    lines = crop_commands(p, Shot(1, 2, "crop"), FPS).splitlines()
    assert len(lines) == 25
    assert lines[0] == "0.0000 crop x 100;"
    assert lines[1] == "0.0400 crop x 120;"
    assert lines[-1] == "0.9600 crop x 200;"
