"""Audio extraction via the system `ffmpeg` binary."""

import asyncio
from pathlib import Path

# Whisper's expected input: 16kHz mono 16-bit PCM.
_FFMPEG_ARGS = ("-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le")
_STDERR_TAIL_CHARS = 1000


class AudioExtractionError(Exception):
    """ffmpeg exited non-zero. The message carries the tail of its stderr."""


async def extract_audio(source: Path, dest: Path) -> None:
    """Write `source`'s audio track to `dest` as a 16kHz mono WAV."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        *_FFMPEG_ARGS,
        str(dest),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace").strip()[-_STDERR_TAIL_CHARS:]
        raise AudioExtractionError(f"ffmpeg exited with {proc.returncode}: {tail}")
