"""One ffmpeg pass per clip: cut, frame, caption, encode (docs/decisions/0013 §4).

`filter_graph` is pure (and unit-tested); `render` runs ffmpeg and
`probe` checks the result with ffprobe.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from app.shorts.framing import Plan, Shot, crop_commands, runs

DURATION_TOLERANCE_S = 0.5
BLUR = "boxblur=20:2"


class RenderError(Exception):
    """ffmpeg failed or produced a file that doesn't check out."""


def _quote(path: Path) -> str:
    # Paths come from mkdtemp (no quotes or colons); single quotes still guard spaces.
    return "'" + str(path).replace("\\", "/").replace("'", "") + "'"


def branch_filter(p: Plan, run: Shot, cmd_file: Path | None, i: int = 0) -> str:
    """The filter chain for one run of same-mode shots, ending at the output size.

    Inner pad labels carry the run index `i`: filtergraph labels must be
    unique, and a clip can have several `fit` runs (season-1 run, 1x30).
    """
    if run.mode == "crop":
        x0 = next((q["x"] for q in p.track if run.start <= q["t"] < run.end), 0)
        return (
            f"sendcmd=f={_quote(cmd_file)},crop=w={p.crop_w}:h={p.height}:x={x0}:y=0,"
            f"scale={p.out_w}:{p.out_h},setsar=1"
        )
    if p.crop_w == 0:  # the target is as wide as the source, or wider: letterbox only if needed
        return (
            f"scale={p.out_w}:{p.out_h}:force_original_aspect_ratio=decrease,"
            f"pad={p.out_w}:{p.out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
    return (
        f"split[bg{i}][fg{i}];[bg{i}]scale={p.out_w}:{p.out_h}:force_original_aspect_ratio=increase,"
        f"crop={p.out_w}:{p.out_h},{BLUR}[b{i}];[fg{i}]scale={p.out_w}:-2[f{i}];"
        f"[b{i}][f{i}]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def filter_graph(p: Plan, fps: float, workdir: Path, ass_file: Path | None) -> str:
    """The `-filter_complex` graph from `[0:v]` to `[vout]`; writes one sendcmd file per crop run."""
    parts = runs(p.shots)
    chains = []
    for i, run in enumerate(parts):
        cmd_file = None
        if run.mode == "crop":
            cmd_file = workdir / f"crop{i}.cmd"
            cmd_file.write_text(crop_commands(p, run, fps))
        chains.append(branch_filter(p, run, cmd_file, i))

    subs = f",ass={_quote(ass_file)}" if ass_file else ""
    if len(parts) == 1:
        return f"[0:v]{chains[0]}{subs}[vout]"
    graph = [f"[0:v]split={len(parts)}" + "".join(f"[s{i}]" for i in range(len(parts)))]
    for i, (run, chain) in enumerate(zip(parts, chains, strict=True)):
        end = "" if i == len(parts) - 1 else f":end={run.end:.4f}"
        graph.append(f"[s{i}]trim=start={run.start:.4f}{end},setpts=PTS-STARTPTS,{chain}[r{i}]")
    graph.append("".join(f"[r{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0{subs}[vout]")
    return ";".join(graph)


def render(src: Path, start: float, end: float, p: Plan, fps: float, ass_text: str | None, out: Path) -> dict:
    """Encode [start, end) of `src` to `out`; returns the probe of the result."""
    with tempfile.TemporaryDirectory(prefix="yts-short-") as tmp:
        workdir = Path(tmp)
        ass_file = None
        if ass_text is not None:
            ass_file = workdir / "captions.ass"
            ass_file.write_text(ass_text, encoding="utf-8")
        graph = filter_graph(p, fps, workdir, ass_file)
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
            "-filter_complex", graph,
            "-map", "[vout]", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RenderError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")
    info = probe(out)
    expected = end - start
    if abs(info["duration"] - expected) > DURATION_TOLERANCE_S:
        raise RenderError(f"clip is {info['duration']:.2f}s, expected {expected:.2f}s")
    if (info["width"], info["height"]) != (p.out_w, p.out_h):
        raise RenderError(f"clip is {info['width']}x{info['height']}, expected {p.out_w}x{p.out_h}")
    if not info["has_audio"] and has_audio(src):
        raise RenderError("the clip lost the source's audio")
    return info


def has_audio(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def probe(path: Path) -> dict:
    """duration, width, height, has_audio of a rendered file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RenderError(f"ffprobe failed: {out.stderr.strip()[-300:]}")
    data = json.loads(out.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise RenderError("no video stream in the rendered clip")
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"]),
    }
