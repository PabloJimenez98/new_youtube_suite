"""ISO-8601 durations as returned by `contentDetails.duration` (0015 §6).

Handles `PT#H#M#S`, `P#DT...` (streams longer than a day) and `P0D`/`PT0S`
(live and upcoming). Anything else raises `ValueError`.
"""

import re

_DURATION = re.compile(
    r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?"
)


def parse_duration(value: str) -> int:
    """Return the duration in whole seconds."""
    match = _DURATION.fullmatch(value)
    # "P" and "PT" alone match the pattern but carry no component.
    if not match or value in ("P", "") or value.endswith("T"):
        raise ValueError(f"not an ISO-8601 duration: {value!r}")
    parts = {k: int(v) for k, v in match.groupdict(default="0").items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]
