"""Channel upload feeds (`feeds/videos.xml?channel_id=...`): free, last ~15
uploads. The hourly "something new" trigger (0015 §1). Stdlib Atom parsing."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

import httpx

FEED_URL = "https://www.youtube.com/feeds/videos.xml"
_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}


class RssError(Exception):
    pass


@dataclass(frozen=True)
class RssEntry:
    video_id: str
    title: str
    published_at: datetime | None


def parse_feed(xml: bytes | str) -> list[RssEntry]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise RssError(f"invalid feed XML: {exc}") from exc
    entries = []
    for entry in root.findall("atom:entry", _NS):
        video_id = entry.findtext("yt:videoId", namespaces=_NS)
        if not video_id:
            continue
        published = entry.findtext("atom:published", namespaces=_NS)
        entries.append(
            RssEntry(
                video_id=video_id,
                title=entry.findtext("atom:title", default="", namespaces=_NS),
                published_at=datetime.fromisoformat(published) if published else None,
            )
        )
    return entries


async def fetch_feed(http: httpx.AsyncClient, channel_id: str) -> list[RssEntry]:
    try:
        response = await http.get(FEED_URL, params={"channel_id": channel_id}, timeout=20)
    except httpx.HTTPError as exc:
        raise RssError(f"feed request failed: {exc}") from exc
    if response.status_code != 200:
        raise RssError(f"feed returned HTTP {response.status_code}")
    return parse_feed(response.content)
