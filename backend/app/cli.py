"""One-off operational commands.

v1 is single-operator/not multi-tenant (see docs/decisions/v1-scope.md
and docs/decisions/0002-backend-skeleton.md), so there is deliberately
no public `/auth/register` endpoint. The one user account is created
via this script instead.

Usage:
    python -m app.cli create-user someone@example.com

    # Market-analysis pipeline (M10, docs/decisions/0015 §6)
    python -m app.cli market add <@handle | channel URL | UC... id>
    python -m app.cli market seed        # app/market/seed_channels.json
    python -m app.cli market list
    python -m app.cli market deactivate <market_channel id>
    python -m app.cli market pull        # the daily pull, now
    python -m app.cli market rss         # the RSS poll, now
"""

import argparse
import asyncio
import getpass
import json
import logging
from pathlib import Path

from sqlalchemy import func, select, update

from app.auth.security import hash_password
from app.database import async_session_factory
from app.models.user import User

SEED_FILE = Path(__file__).parent / "market" / "seed_channels.json"


async def create_user(email: str, password: str) -> None:
    async with async_session_factory() as db:
        user = User(email=email, password_hash=hash_password(password))
        db.add(user)
        await db.commit()
        print(f"Created user #{user.id} <{user.email}>")


# ---- market ----------------------------------------------------------------


async def _market_add(refs: list[str]) -> None:
    from app.config import get_settings
    from app.market.pull import ChannelNotFound, add_channel, backfill_channel
    from app.market.tasks import youtube_client
    from app.market.youtube_api import BudgetExhausted, QuotaExceeded, YouTubeApiError

    settings = get_settings()
    async with youtube_client(settings) as (yt, _http):
        for ref in refs:
            try:
                channel = await add_channel(async_session_factory, yt, ref)
            except (QuotaExceeded, BudgetExhausted) as exc:
                print(f"STOPPED: {exc}")
                break
            except (ChannelNotFound, ValueError, YouTubeApiError) as exc:
                print(f"SKIP {ref}: {exc}")
                continue
            if channel.backfill_completed_at is not None:
                # Re-running `seed`/`add` must not re-page a whole catalogue;
                # new uploads arrive through RSS and the daily pull.
                print(f"Already tracked: #{channel.id} {channel.channel_title} (backfilled)")
                continue
            print(f"Added #{channel.id} {channel.channel_title} ({channel.youtube_channel_id}); backfilling...")
            try:
                new = await backfill_channel(async_session_factory, yt, channel)
            except (QuotaExceeded, BudgetExhausted) as exc:
                print(f"STOPPED during backfill: {exc} (resumes on the next pull)")
                break
            except YouTubeApiError as exc:
                print(f"  backfill FAILED: {exc} (retried on the next pull)")
                continue
            print(f"  {new} new videos")
        print(f"units used: {yt.units_used}")
    print("Snapshots are taken by the daily pull (`market pull` to run it now).")


async def _market_list() -> None:
    from app.models.market import MarketChannel, MarketVideo

    async with async_session_factory() as db:
        rows = (
            await db.execute(
                select(MarketChannel, func.count(MarketVideo.id))
                .outerjoin(MarketVideo, MarketVideo.market_channel_id == MarketChannel.id)
                .group_by(MarketChannel.id)
                .order_by(MarketChannel.id)
            )
        ).all()
    for ch, n in rows:
        state = "active" if ch.active else "inactive"
        backfill = "backfilled" if ch.backfill_completed_at else "backfill pending"
        print(f"#{ch.id:<3} {ch.youtube_channel_id}  {state:<8} {backfill:<16} {n:>6} videos  {ch.channel_title}")


async def _market_deactivate(channel_id: int) -> None:
    from app.models.market import MarketChannel

    async with async_session_factory() as db:
        result = await db.execute(
            update(MarketChannel).where(MarketChannel.id == channel_id).values(active=False)
        )
        await db.commit()
    print("Deactivated (history kept)." if result.rowcount else f"No market channel #{channel_id}.")


async def _market_run(kind: str) -> None:
    from app.market.tasks import market_daily_pull, market_rss_poll

    # The task logs its own report line (and the skip) via `app.market`.
    await (market_daily_pull if kind == "pull" else market_rss_poll)({})


def _seed_refs() -> list[str]:
    entries = json.loads(SEED_FILE.read_text(encoding="utf-8"))["channels"]
    refs = []
    for entry in entries:
        if entry.get("ref"):
            refs.append(entry["ref"])
        else:
            print(f"SKIP {entry['name']}: no handle/ID in the seed file yet")
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    cu = sub.add_parser("create-user")
    cu.add_argument("email")
    market = sub.add_parser("market").add_subparsers(dest="market_command", required=True)
    market.add_parser("add").add_argument("refs", nargs="+")
    market.add_parser("seed")
    market.add_parser("list")
    market.add_parser("deactivate").add_argument("channel_id", type=int)
    market.add_parser("pull")
    market.add_parser("rss")
    args = parser.parse_args()

    if args.command == "create-user":
        password = getpass.getpass("Password: ")
        asyncio.run(create_user(args.email, password))
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    cmd = args.market_command
    if cmd == "add":
        asyncio.run(_market_add(args.refs))
    elif cmd == "seed":
        asyncio.run(_market_add(_seed_refs()))
    elif cmd == "list":
        asyncio.run(_market_list())
    elif cmd == "deactivate":
        asyncio.run(_market_deactivate(args.channel_id))
    else:
        asyncio.run(_market_run(cmd))


if __name__ == "__main__":
    main()
