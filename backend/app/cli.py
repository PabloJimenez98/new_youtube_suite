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

    # Connected-channel analytics (M11, docs/decisions/0017 §6). Connecting
    # happens in the app (Analytics page), not here.
    python -m app.cli analytics status   # connection status, no tokens
    python -m app.cli analytics pull     # the daily analytics pull, now

    # Reporting API: real CTR/impressions (M12, docs/decisions/0018 §4)
    python -m app.cli reporting status   # job, report counts, reach rows
    python -m app.cli reporting poll     # the daily reporting poll, now

    # Relevance BiLSTM (M15, docs/decisions/0024 §5). Stop the worker first.
    python -m app.cli relevance pseudo-label [--upload ID] [--force]
    python -m app.cli relevance train --label bilstm-e5s-YYYYMMDD-i1
    python -m app.cli relevance eval <label>        # offline metrics + gate
    python -m app.cli relevance ab-sheet <label>    # blind top-5 A/B for the owner
    python -m app.cli relevance ab-score <label>    # after filling the sheet
    python -m app.cli relevance status <label>
    python -m app.cli models list [--name relevance]
    python -m app.cli models set-live relevance <label>   # also the rollback
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


# ---- analytics -------------------------------------------------------------


async def _analytics_status() -> None:
    from app.config import get_settings
    from app.models.analytics import AnalyticsRetentionPoint, AnalyticsSnapshot, AnalyticsVideo
    from app.youtube_oauth.tokens import get_credential

    if not get_settings().youtube_oauth_configured:
        print("not_configured: set GOOGLE_OAUTH_CLIENT_ID/SECRET and OAUTH_TOKEN_ENCRYPTION_KEY (README)")
        return
    row = await get_credential(async_session_factory)
    if row is None:
        print("disconnected: connect a channel from the Analytics page")
        return
    print(f"{row.status}: {row.connected_channel_title} ({row.connected_channel_id})")
    print(f"  scopes: {', '.join(row.scopes)}")
    print(f"  access token expires {row.token_expiry:%Y-%m-%d %H:%M} UTC, last refresh {row.last_refresh_at}")
    if row.last_error:
        print(f"  last error: {row.last_error}")
    async with async_session_factory() as db:
        for model in (AnalyticsVideo, AnalyticsSnapshot, AnalyticsRetentionPoint):
            n = await db.scalar(select(func.count()).select_from(model).where(model.channel_id == row.connected_channel_id))
            print(f"  {model.__tablename__}: {n} rows")


async def _analytics_pull() -> None:
    from app.analytics.tasks import analytics_daily_pull

    # The task logs its own report line (and any skip) via `app.analytics`.
    await analytics_daily_pull({})


# ---- reporting -------------------------------------------------------------


async def _reporting_status() -> None:
    from app.reporting.status import reach_summary
    from app.youtube_oauth.tokens import get_credential

    row = await get_credential(async_session_factory)
    if row is None:
        print("disconnected: connect a channel from the Analytics page")
        return
    summary = await reach_summary(async_session_factory, row.connected_channel_id)
    if summary is None:
        print(f"no reach job yet for {row.connected_channel_id}: run `python -m app.cli reporting poll`")
        return
    print(f"job {summary.youtube_job_id}: {summary.job_status}, created {summary.job_created_at:%Y-%m-%d %H:%M} UTC")
    print(f"  last poll: {summary.last_polled_at}")
    print(
        f"  reports: {summary.reports_downloaded} downloaded, {summary.reports_pending} pending, "
        f"{summary.reports_failed} failed; latest day {summary.latest_report_date}"
    )
    print(f"  analytics_reach_snapshots: {summary.reach_rows} rows")
    if summary.last_error:
        print(f"  last error: {summary.last_error}")


async def _reporting_poll() -> None:
    from app.reporting.tasks import reporting_daily_poll

    # The task logs its own report line (and any skip) via `app.reporting`.
    await reporting_daily_poll({})


# ---- insights (precomputed Analytics, 0023) -----------------------------------------


async def _insights_refresh(force: bool) -> None:
    from app.insights.store import refresh_insights

    report = await refresh_insights(async_session_factory, force=force)
    print(f"{len(report.computed)} computed, {len(report.skipped)} unchanged in {report.seconds:.1f}s")


async def _insights_status() -> None:
    from app.models.insight import InsightResult

    async with async_session_factory() as db:
        rows = (await db.execute(select(InsightResult).order_by(InsightResult.name, InsightResult.format))).scalars().all()
    if not rows:
        print("no precomputed results yet: run `python -m app.cli insights refresh`")
    for r in rows:
        print(f"{r.name:<9} {r.format:<5} {r.computed_at:%Y-%m-%d %H:%M} UTC  {r.duration_ms:>6} ms  {r.source_key}")


# ---- relevance BiLSTM + model versions (M15, 0024 §5) ----------------------------


async def _relevance(args: argparse.Namespace) -> None:
    from app.relevance import retrain
    from app.relevance.training import TrainConfig

    cmd = args.relevance_command
    try:
        if cmd == "pseudo-label":
            r = await retrain.pseudo_label(async_session_factory, args.upload or None, args.force)
            print(f"teacher {r['teacher']}: {r['labelled']} uploads labelled, {r['skipped']} skipped")
        elif cmd == "train":
            from app.relevance.bilstm import resolve_device
            from app.config import get_settings

            cfg = TrainConfig(
                hidden=args.hidden, layers=args.layers, patience=args.patience, combine_mode=args.combine,
                lambdas=tuple(float(x) for x in args.lambdas.split(",")),
                seeds=tuple(range(args.seeds)),
                device=resolve_device(get_settings().relevance_device),
            )
            encoder = None
            if args.encoder_dir:
                encoder = retrain.EncoderSpec(Path(args.encoder_dir), args.encoder_name, args.encoder_prefix)
            v = await retrain.train_version(async_session_factory, args.label, teacher_label=args.teacher,
                                            cfg=cfg, encoder=encoder, cache_dir=retrain.bilstm.default_cache_dir())
            t = v.metrics["train"]
            print(f"trained {v.version_label} (not live): lambda {t['lambda']}, seed {t['seed']}, "
                  f"val spearman {t['val_spearman']:.4f}; episodes {t['episodes']}; bundle {v.artifact_path}")
            print(f"next: python -m app.cli relevance eval {v.version_label}")
        elif cmd == "eval":
            r = await retrain.eval_version(async_session_factory, args.label,
                                           cache_dir=retrain.bilstm.default_cache_dir())
            print(f"test episodes: {r['episodes']} (teacher {r['teacher']})")
            for method, m in r["summary"].items():
                print(f"  {method:<9} " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))
            g = r["gate"]
            print(f"offline gate: {'PASSED' if g['passed'] else 'FAILED'} "
                  f"(BiLSTM {g['bilstm_spearman']:.3f} vs heuristic {g['heuristic_spearman']:.3f})")
        elif cmd == "ab-sheet":
            sheet, key = await retrain.write_ab_sheet(async_session_factory, args.label,
                                                      Path(args.out) if args.out else None,
                                                      cache_dir=retrain.bilstm.default_cache_dir())
            print(f"sheet: {sheet}\nkey (don't open before answering): {key}")
        elif cmd == "ab-score":
            r = await retrain.score_ab_sheet(async_session_factory, args.label, Path(args.out) if args.out else None)
            print(f"candidate: {r['wins']} wins, {r['ties']} ties, {r['losses']} losses -> "
                  f"A/B {'PASSED' if r['passed'] else 'FAILED'}")
        elif cmd == "status":
            s = await retrain.promotion_status(async_session_factory, args.label)
            print(f"{s['label']}: offline gate {s['offline_gate']}, A/B {s['ab']} -> "
                  f"{'promotable' if s['promotable'] else 'not promotable'}; live is {s['live']}")
    except (retrain.RetrainError, LookupError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


async def _models(args: argparse.Namespace) -> None:
    from app.model_registry import VersionNotFoundError, list_versions, set_live
    from app.models.model_version import ModelVersion

    async with async_session_factory() as db:
        if args.models_command == "list":
            for v in await list_versions(db, args.name):
                m = v.metrics or {}
                gate = (m.get("test") or {}).get("gate", {}).get("passed")
                ab = (m.get("ab") or {}).get("passed")
                extra = "" if v.artifact_path is None else f"  gate={gate} ab={ab}"
                print(f"{v.model_name:<9} {'LIVE' if v.is_live else '    '} {v.version_label}{extra}")
            return
        target = await db.scalar(select(ModelVersion).where(ModelVersion.model_name == args.name,
                                                            ModelVersion.version_label == args.label))
        if target is not None and target.artifact_path is not None and not args.force and args.name == "relevance":
            from app.relevance.retrain import promotion_status

            s = await promotion_status(async_session_factory, args.label)
            if not s["promotable"]:
                raise SystemExit(
                    f"REFUSED: {args.label} is not promotable (offline gate {s['offline_gate']}, A/B {s['ab']}); "
                    "see docs/decisions/0024 §7, or pass --force"
                )
        try:
            switch = await set_live(db, args.name, args.label)
        except (VersionNotFoundError, FileNotFoundError) as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        except Exception as exc:  # a broken bundle
            raise SystemExit(f"ERROR: {exc}") from exc
    if switch.changed:
        print(f"{switch.model_name}: live {switch.previous} -> {switch.current}")
        print(f"rollback: python -m app.cli models set-live {switch.model_name} {switch.previous}")
    else:
        print(f"{switch.model_name}: {switch.current} is already live")


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
    analytics = sub.add_parser("analytics").add_subparsers(dest="analytics_command", required=True)
    analytics.add_parser("status")
    analytics.add_parser("pull")
    reporting = sub.add_parser("reporting").add_subparsers(dest="reporting_command", required=True)
    reporting.add_parser("status")
    reporting.add_parser("poll")
    insights = sub.add_parser("insights").add_subparsers(dest="insights_command", required=True)
    insights.add_parser("status")
    insights.add_parser("refresh").add_argument("--force", action="store_true", help="recompute even if the sources are unchanged")
    relevance = sub.add_parser("relevance").add_subparsers(dest="relevance_command", required=True)
    pl = relevance.add_parser("pseudo-label", help="run the teacher on uploads it hasn't labelled")
    pl.add_argument("--upload", type=int, action="append", help="only these upload ids (repeatable)")
    pl.add_argument("--force", action="store_true", help="relabel uploads that already have labels")
    tr = relevance.add_parser("train", help="train a BiLSTM version (registered not live)")
    tr.add_argument("--label", required=True)
    tr.add_argument("--teacher", help="teacher version label (default: the one with the most labelled uploads)")
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--layers", type=int, default=1)
    tr.add_argument("--lambdas", default="0,0.1,0.5")
    tr.add_argument("--seeds", type=int, default=3)
    tr.add_argument("--patience", type=int, default=40)
    tr.add_argument("--combine", choices=["heads", "exact_heuristic"], default="heads",
                    help="exact_heuristic: compute the heuristic exactly, BiLSTM for the LLM part (0024 §9)")
    tr.add_argument("--encoder-dir", help="default: the chapters bundle's encoder/")
    tr.add_argument("--encoder-name", default="intfloat/multilingual-e5-small")
    tr.add_argument("--encoder-prefix", default="passage: ")
    for name in ("eval", "ab-sheet", "ab-score", "status"):
        p = relevance.add_parser(name)
        p.add_argument("label")
        if name in ("ab-sheet", "ab-score"):
            p.add_argument("--out", help="sheet directory (default: <storage>/../m15-ab/<label>)")
    models = sub.add_parser("models").add_subparsers(dest="models_command", required=True)
    models.add_parser("list").add_argument("--name", choices=["relevance", "ctr", "hook"])
    sl = models.add_parser("set-live", help="switch the live version (also the rollback)")
    sl.add_argument("name", choices=["relevance", "ctr", "hook"])
    sl.add_argument("label")
    sl.add_argument("--force", action="store_true", help="skip the promotion check (0024 §7)")
    args = parser.parse_args()

    if args.command == "create-user":
        password = getpass.getpass("Password: ")
        asyncio.run(create_user(args.email, password))
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    if args.command == "analytics":
        asyncio.run(_analytics_status() if args.analytics_command == "status" else _analytics_pull())
        return
    if args.command == "insights":
        asyncio.run(_insights_status() if args.insights_command == "status" else _insights_refresh(args.force))
        return
    if args.command == "relevance":
        asyncio.run(_relevance(args))
        return
    if args.command == "models":
        asyncio.run(_models(args))
        return
    if args.command == "reporting":
        asyncio.run(_reporting_status() if args.reporting_command == "status" else _reporting_poll())
        return
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
