"""ID batching for `videos.list`, which takes at most 50 IDs per call."""

from collections.abc import Iterable

MAX_IDS_PER_CALL = 50


def batch_ids(ids: Iterable[str], size: int = MAX_IDS_PER_CALL) -> list[list[str]]:
    """Split into batches of `size`, dropping duplicates and keeping order."""
    unique = list(dict.fromkeys(ids))
    return [unique[i : i + size] for i in range(0, len(unique), size)]
