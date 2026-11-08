"""Link an upload to its YouTube video (docs/decisions/0022 §1).

Episodes downloaded from YouTube carry the video ID in their filename,
e.g. ``SHOW 1X09： Title [oSCkZ8yVz7M].mkv``. The ID is read from the last
``[...]`` group of exactly 11 YouTube-ID characters, stored in
``uploads.youtube_video_id`` and editable through ``PATCH /uploads/{id}``.
"""

import re

YOUTUBE_ID_PATTERN = r"[A-Za-z0-9_-]{11}"
_IN_FILENAME = re.compile(rf"\[({YOUTUBE_ID_PATTERN})\](?=[^\[]*$)")
_VALID = re.compile(rf"^{YOUTUBE_ID_PATTERN}$")

# Same rule in SQL, for the 0015 backfill: Postgres' substring() returns the
# first capture group; the trailing [^\[]*$ pins it to the last bracket group.
SQL_FILENAME_REGEX = r"\[([A-Za-z0-9_-]{11})\][^\[]*$"


def youtube_id_from_filename(filename: str) -> str | None:
    match = _IN_FILENAME.search(filename)
    return match.group(1) if match else None


def is_valid_youtube_id(value: str) -> bool:
    return bool(_VALID.match(value))
