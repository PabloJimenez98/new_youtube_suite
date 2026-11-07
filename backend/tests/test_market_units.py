"""M10 pure-logic tests: durations, batching, due-video selection, RSS
parsing, `videos.list` mapping, channel references. No network, no DB."""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.market.batching import batch_ids
from app.market.durations import parse_duration
from app.market.rss import RssError, parse_feed
from app.market.selection import VideoState, is_due, select_due
from app.market.youtube_api import parse_channel_ref, snapshot_fields

FIXTURES = Path(__file__).parent / "fixtures" / "market"


# ---- durations ----


@pytest.mark.parametrize(
    "value, seconds",
    [
        ("PT15M33S", 933),
        ("PT1H2M3S", 3723),
        ("PT45S", 45),
        ("PT2H", 7200),
        ("P1DT2H3M", 93780),
        ("P0D", 0),
        ("PT0S", 0),
    ],
)
def test_parse_duration(value: str, seconds: int) -> None:
    assert parse_duration(value) == seconds


@pytest.mark.parametrize("value", ["15:33", "", "PT", "P", "P1DT", "PT1.5S", "1H"])
def test_parse_duration_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(value)


# ---- batching ----


@pytest.mark.parametrize("n, sizes", [(0, []), (1, [1]), (50, [50]), (51, [50, 1]), (120, [50, 50, 20])])
def test_batch_sizes(n: int, sizes: list[int]) -> None:
    assert [len(b) for b in batch_ids(f"v{i}" for i in range(n))] == sizes


def test_batch_dedupes_keeping_order() -> None:
    assert batch_ids(["c", "a", "c", "b", "a"]) == [["c", "a", "b"]]


# ---- selection ----

TODAY = date(2026, 11, 7)


def _video(published_days_ago: int, last_days_ago: int | None, **kw) -> VideoState:
    published = datetime.combine(TODAY - timedelta(days=published_days_ago), datetime.min.time(), timezone.utc)
    return VideoState(
        youtube_video_id=kw.get("vid", "v"),
        published_at=published,
        last_snapshot_date=None if last_days_ago is None else TODAY - timedelta(days=last_days_ago),
        unavailable=kw.get("unavailable", False),
        channel_active=kw.get("active", True),
    )


def _due(v: VideoState) -> bool:
    return is_due(v, TODAY, recent_window_days=90, older_interval_days=7)


def test_recent_without_snapshot_today_is_due() -> None:
    assert _due(_video(10, 1)) and _due(_video(10, None))


def test_recent_already_snapshotted_today_is_not_due() -> None:
    assert not _due(_video(10, 0))


def test_older_follows_the_weekly_interval() -> None:
    assert not _due(_video(400, 6))
    assert _due(_video(400, 7))
    assert _due(_video(400, None))
    assert not _due(_video(400, 0))


def test_unavailable_and_inactive_are_never_due() -> None:
    assert not _due(_video(10, None, unavailable=True))
    assert not _due(_video(10, None, active=False))


def test_ninety_day_edge() -> None:
    assert _due(_video(90, 1))  # still recent: daily
    assert not _due(_video(91, 1))  # older: weekly


def test_select_due_keeps_input_order() -> None:
    videos = [_video(5, 1, vid="a"), _video(400, 2, vid="b"), _video(400, 8, vid="c")]
    assert select_due(videos, TODAY, 90, 7) == ["a", "c"]


# ---- RSS ----


def test_parse_feed_fixture() -> None:
    entries = parse_feed((FIXTURES / "channel_feed.xml").read_bytes())
    assert [e.video_id for e in entries] == ["vid_new_0001", "vid_old_0001"]
    assert entries[0].title == "Episodio 120 | Con invitada"
    assert entries[0].published_at == datetime(2026, 11, 6, 17, 0, 12, tzinfo=timezone.utc)


def test_parse_empty_feed() -> None:
    assert parse_feed('<feed xmlns="http://www.w3.org/2005/Atom"><title>x</title></feed>') == []


def test_parse_invalid_feed_raises() -> None:
    with pytest.raises(RssError):
        parse_feed(b"<html><body>Error 404</body>")


# ---- videos.list mapping ----


@pytest.fixture
def items() -> dict[str, dict]:
    data = json.loads((FIXTURES / "videos_list.json").read_text(encoding="utf-8"))
    return {i["id"]: i for i in data["items"]}


def test_mapping_casts_counters_and_parses_fields(items: dict[str, dict]) -> None:
    row = snapshot_fields(items["vid_full_001"])
    assert row == {
        "youtube_video_id": "vid_full_001",
        "title": "Episodio 117 | Todo",
        "published_at": datetime(2026, 10, 1, 16, 0, 5, tzinfo=timezone.utc),
        "duration_seconds": 3723,
        "view_count": 123456,
        "like_count": 7890,
        "comment_count": 321,
        "category_id": "23",
        "tags": ["podcast", "humor"],
        "thumbnail_url": "https://i.ytimg.com/vi/vid_full_001/hqdefault.jpg",
    }


def test_mapping_hidden_counts_are_null_not_zero(items: dict[str, dict]) -> None:
    row = snapshot_fields(items["vid_hidden_01"])
    assert row["like_count"] is None and row["comment_count"] is None
    assert row["view_count"] == 999
    assert row["tags"] == []
    # No `high` thumbnail: falls back to the next size.
    assert row["thumbnail_url"].endswith("mqdefault.jpg")


def test_mapping_bad_duration_is_null_and_row_still_maps(items: dict[str, dict]) -> None:
    row = snapshot_fields(items["vid_bad_dur_1"])
    assert row["duration_seconds"] is None
    assert row["thumbnail_url"] is None
    assert row["title"] == "Weird duration"


# ---- channel references ----


@pytest.mark.parametrize(
    "ref, parsed",
    [
        ("@TheWildProject", ("handle", "@TheWildProject")),
        ("https://www.youtube.com/@lexfridman/videos", ("handle", "@lexfridman")),
        ("UCSHZKyawb77ixDdsGog4iWA", ("id", "UCSHZKyawb77ixDdsGog4iWA")),
        ("https://www.youtube.com/channel/UCSHZKyawb77ixDdsGog4iWA", ("id", "UCSHZKyawb77ixDdsGog4iWA")),
    ],
)
def test_parse_channel_ref(ref: str, parsed: tuple[str, str]) -> None:
    assert parse_channel_ref(ref) == parsed


@pytest.mark.parametrize("ref", ["", "@", "lexfridman", "UCshort"])
def test_parse_channel_ref_rejects(ref: str) -> None:
    with pytest.raises(ValueError):
        parse_channel_ref(ref)
