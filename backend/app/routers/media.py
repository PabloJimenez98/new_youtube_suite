"""Range-capable file streaming shared by the upload and clip file endpoints.

Starlette's FileResponse doesn't implement Range requests in the version
pinned here, and Range support is what lets a <video> element seek
without downloading the whole file first -- so it is handled by hand
(docs/decisions/0005-video-preview-streaming.md).
"""

import re
from collections.abc import Iterator
from pathlib import Path

from fastapi import HTTPException, Request, status
from fastapi.responses import StreamingResponse

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 64 * 1024


def _iter_file_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    """Yield bytes [start, end] (inclusive) from `path`, in chunks."""
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def stream_file(path: Path, media_type: str, request: Request,
                headers: dict[str, str] | None = None) -> StreamingResponse:
    """The whole file, or the requested byte range as a 206."""
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    file_size = path.stat().st_size
    extra = headers or {}

    range_header = request.headers.get("range")
    if range_header:
        match = _RANGE_RE.match(range_header)
        if match:
            start_str, end_str = match.groups()
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            if start <= end:
                return StreamingResponse(
                    _iter_file_range(path, start, end),
                    status_code=status.HTTP_206_PARTIAL_CONTENT,
                    media_type=media_type,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(end - start + 1),
                        **extra,
                    },
                )

    return StreamingResponse(
        _iter_file_range(path, 0, file_size - 1),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size), **extra},
    )
