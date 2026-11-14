"""M15's offline pipeline, driven by the CLI (docs/decisions/0024 §5).

    pseudo-label -> train -> eval -> ab-sheet -> (owner) -> ab-score -> models set-live

Runs outside the worker. GPU steps (the teacher's LLM, the encoder, training)
need the worker stopped, as for the chapter models (0009).
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.chapter import Chapter
from app.models.job import Job
from app.models.model_version import ModelVersion
from app.models.relevance import RelevancePseudoLabel
from app.models.transcript import TranscriptSegment
from app.models.upload import Upload
from app.relevance import bilstm
from app.relevance.dataset import (
    Episode,
    load_episodes,
    load_or_make_split,
    partition,
    resolve_teacher,
    split_hash,
)
from app.relevance.evaluation import AbEpisode, EpisodeEval, ab_sheet, evaluate, score_ab
from app.relevance.training import TrainConfig, train_grid

log = logging.getLogger("app.relevance")


class RetrainError(Exception):
    """A user-facing failure of an M15 command."""


def split_path() -> Path:
    return get_settings().relevance_model_root / "split.json"


# ---- pseudo-labels ---------------------------------------------------------------


async def pseudo_label(session_factory: async_sessionmaker[AsyncSession], upload_ids: list[int] | None = None,
                       force: bool = False) -> dict:
    """Run the configured teacher on transcribed uploads it hasn't labelled yet; write only the store."""
    from app.relevance import pipeline
    from app.worker.handlers.relevance import (
        get_or_create_version,
        score_with_teacher,
        words_from_segments,
        write_pseudo_labels,
    )

    label = pipeline.get_scorer().version_label  # fails fast without an LLM bundle
    done = skipped = 0
    async with session_factory() as db:
        transcribed = select(Job.upload_id).where(Job.tool_type == "transcribe", Job.status == "succeeded")
        q = select(Upload.id).where(Upload.id.in_(transcribed)).order_by(Upload.id)
        if upload_ids:
            q = q.where(Upload.id.in_(upload_ids))
        candidates = list((await db.execute(q)).scalars())
        teacher = await db.scalar(
            select(ModelVersion).where(ModelVersion.model_name == "relevance", ModelVersion.version_label == label)
        )
        labelled: set[int] = set()
        if teacher is not None and not force:
            labelled = set(
                (await db.execute(
                    select(RelevancePseudoLabel.upload_id).where(RelevancePseudoLabel.teacher_version_id == teacher.id)
                    .distinct()
                )).scalars()
            )
    for upload_id in candidates:
        if upload_id in labelled:
            skipped += 1
            continue
        async with session_factory() as db:
            upload = await db.get(Upload, upload_id)
            segments = list(
                (await db.execute(
                    select(TranscriptSegment).where(TranscriptSegment.upload_id == upload_id)
                    .order_by(TranscriptSegment.segment_index).options(selectinload(TranscriptSegment.words))
                )).scalars()
            )
            if not segments:
                skipped += 1
                continue
            chapters = [(c.start_seconds, c.title) for c in (
                await db.execute(select(Chapter).where(Chapter.upload_id == upload_id).order_by(Chapter.order_index))
            ).scalars()]
            duration = upload.duration_seconds or max(s.end_seconds for s in segments)
            got_label, description, result = await score_with_teacher(words_from_segments(segments), duration, chapters)
            version = await get_or_create_version(db, got_label, description)
            await write_pseudo_labels(db, upload_id, version, result.scores)
            await db.commit()
            done += 1
            log.info("pseudo-label: upload %s, %s windows (%s)", upload_id, len(result.scores), got_label)
    return {"teacher": label, "labelled": done, "skipped": skipped}


# ---- embeddings -----------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSpec:
    dir: Path
    name: str
    prefix: str
    max_length: int = bilstm.ENCODER_MAX_LENGTH


def default_encoder_spec() -> EncoderSpec:
    """The chapters bundle's frozen e5 encoder (0024 §1)."""
    bundle = get_settings().chapters_model_dir
    if bundle is None or not (Path(bundle) / "encoder").is_dir():
        raise RetrainError("No encoder: set CHAPTERS_MODEL_DIR to the chapters bundle, or pass --encoder-dir")
    boundary = json.loads((Path(bundle) / "manifest.json").read_text()).get("boundary", {})
    return EncoderSpec(Path(bundle) / "encoder", boundary.get("encoder", "intfloat/multilingual-e5-small"),
                       boundary.get("encoder_prefix", "passage: "))


def default_lid_path() -> Path:
    """The chapters bundle's fastText LID, the one the teacher uses (0012 §2)."""
    bundle = get_settings().chapters_model_dir
    manifest = Path(bundle) / "manifest.json" if bundle else None
    lid = json.loads(manifest.read_text()).get("titles", {}).get("lid") if manifest and manifest.is_file() else None
    if not lid or not (Path(bundle) / lid["file"]).is_file():
        raise RetrainError("exact_heuristic needs the LID model: set CHAPTERS_MODEL_DIR to the chapters bundle")
    return Path(bundle) / lid["file"]


def embed_episodes(episodes: list[Episode], spec: EncoderSpec, encode: bilstm.Encoder | None = None,
                   cache_dir: Path | None = None) -> None:
    """Fill `Episode.emb`, through the embedding cache."""
    if encode is None:
        encode = bilstm.SentenceEncoder(spec.dir, spec.prefix, spec.max_length,
                                        bilstm.resolve_device(get_settings().relevance_device))
    for e in episodes:
        key = bilstm.embedding_cache_key(spec.name, spec.prefix, spec.max_length, e.texts)
        e.emb = bilstm.cached_embeddings(cache_dir, key, e.texts, encode)


# ---- train --------------------------------------------------------------------


async def _episodes_and_split(session_factory, teacher_label: str | None):
    async with session_factory() as db:
        teacher = await resolve_teacher(db, teacher_label)
    episodes, skipped = await load_episodes(session_factory, teacher)
    for upload_id, reason in skipped:
        log.warning("skip upload %s: %s", upload_id, reason)
    split, created = load_or_make_split(split_path(), [(e.upload_id, e.language) for e in episodes])
    if created:
        log.info("new split written to %s", split_path())
    return teacher, episodes, partition(episodes, split), split


async def train_version(session_factory: async_sessionmaker[AsyncSession], label: str, *,
                        teacher_label: str | None = None, cfg: TrainConfig | None = None,
                        encoder: EncoderSpec | None = None, encode: bilstm.Encoder | None = None,
                        cache_dir: Path | None = None, lid_src_path: Path | None = None) -> ModelVersion:
    """Train, write the bundle, register a non-live `model_versions` row."""
    settings = get_settings()
    cfg = cfg or TrainConfig(device=bilstm.resolve_device(settings.relevance_device))
    async with session_factory() as db:
        if await db.scalar(select(ModelVersion.id).where(ModelVersion.model_name == "relevance",
                                                         ModelVersion.version_label == label)):
            raise RetrainError(f"Relevance version {label!r} already exists -- pick a new label")
    bundle_dir = settings.relevance_model_root / label
    if bundle_dir.exists():
        raise RetrainError(f"{bundle_dir} already exists -- pick a new label")
    teacher, episodes, parts, split = await _episodes_and_split(session_factory, teacher_label)
    if not parts["train"]:
        raise RetrainError("No training episodes")
    encoder = encoder or default_encoder_spec()
    embed_episodes(parts["train"] + parts["val"], encoder, encode, cache_dir)
    outcome = train_grid(parts["train"], parts["val"], cfg)

    train_mean = float(np.mean(np.concatenate([e.target for e in parts["train"]])))
    training = {
        "teacher": teacher.version_label,
        "split": split_hash(split),
        "episodes": {k: len(parts[k]) for k in ("train", "val", "test")},
        "train_target_mean": train_mean,
        **outcome.summary(),
    }
    config = bilstm.BundleConfig(label=label, encoder_name=encoder.name, encoder_prefix=encoder.prefix,
                                 encoder_max_length=encoder.max_length, model_config=outcome.model.config,
                                 combine_mode=cfg.combine_mode, training=training)
    metrics = {"train": training}
    lid_src = None
    if cfg.combine_mode == "exact_heuristic":
        lid_src = lid_src_path or default_lid_path()
    bilstm.save_bundle(bundle_dir, config, outcome.model.cpu(), metrics, encoder.dir, lid_src)

    async with session_factory() as db:
        version = ModelVersion(
            model_name="relevance",
            version_label=label,
            architecture_notes=(
                f"Frozen {encoder.name} window embeddings -> BiLSTM(hidden {cfg.hidden}, {cfg.layers} layer) -> "
                f"two heads (heuristic, LLM), combined {config.combine_heuristic}/{config.combine_llm} "
                f"(mode {config.combine_mode}); "
                f"Huber + lambda {outcome.lam} smoothness. docs/decisions/0024-relevance-bilstm-m15.md"
            ),
            training_data_reference={"kind": "distilled", "encoder": encoder.name, **{
                k: training[k] for k in ("teacher", "split", "episodes", "lambda", "seed")},
                "combine_mode": cfg.combine_mode},
            metrics=metrics,
            artifact_path=str(bundle_dir),
            is_live=False,
            trained_at=datetime.now(UTC),
        )
        db.add(version)
        await db.commit()
    return version


# ---- eval / A/B -------------------------------------------------------------------


async def _trained_version(db: AsyncSession, label: str) -> ModelVersion:
    version = await db.scalar(select(ModelVersion).where(ModelVersion.model_name == "relevance",
                                                         ModelVersion.version_label == label))
    if version is None or version.artifact_path is None:
        raise RetrainError(f"No trained relevance version {label!r}")
    return version


async def _predictions(session_factory, label: str, which: str, encode=None, cache_dir=None):
    async with session_factory() as db:
        version = await _trained_version(db, label)
    config, model = bilstm.load_model(Path(version.artifact_path))
    teacher_label = (version.training_data_reference or {}).get("teacher")
    teacher, episodes, parts, split = await _episodes_and_split(session_factory, teacher_label)
    spec = EncoderSpec(Path(version.artifact_path) / "encoder", config.encoder_name, config.encoder_prefix,
                       config.encoder_max_length)
    chosen = parts[which]
    embed_episodes(chosen, spec, encode, cache_dir)
    preds = {}
    for e in chosen:
        h, l = bilstm.predict(model, e.emb)  # noqa: E741
        h_used, s = bilstm.combine_scores(config.combine_mode, h, l, e.heuristic,
                                          config.combine_heuristic, config.combine_llm)
        preds[e.upload_id] = (h_used, l, s)
    return version, teacher, config, chosen, preds


async def eval_version(session_factory: async_sessionmaker[AsyncSession], label: str, *, encode=None,
                       cache_dir=None) -> dict:
    """Offline metrics on the frozen test split; stored in the bundle and `model_versions.metrics`."""
    version, teacher, config, episodes, preds = await _predictions(session_factory, label, "test", encode, cache_dir)
    if not episodes:
        raise RetrainError("The test split has no labelled episodes")
    constant = (config.training or {}).get("train_target_mean", 0.5)
    report = evaluate(
        [EpisodeEval(e.upload_id, e.target, e.heuristic, e.llm, e.starts, *preds[e.upload_id]) for e in episodes],
        constant,
    )
    report["teacher"] = teacher.version_label
    report["evaluated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    async with session_factory() as db:
        row = await _trained_version(db, label)
        row.metrics = {**(row.metrics or {}), "test": report}
        await db.commit()
    metrics_file = Path(version.artifact_path) / "metrics.json"
    stored = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
    metrics_file.write_text(json.dumps({**stored, "test": report}, indent=1))
    return report


def default_ab_dir(label: str) -> Path:
    """Outside the repo, next to the uploads (0024 §5)."""
    return get_settings().storage_base_path.resolve().parent / "m15-ab" / label


async def write_ab_sheet(session_factory: async_sessionmaker[AsyncSession], label: str, out_dir: Path | None = None,
                         *, seed: int = 15, encode=None, cache_dir=None) -> tuple[Path, Path]:
    version, teacher, config, episodes, preds = await _predictions(session_factory, label, "ab", encode, cache_dir)
    if not episodes:
        raise RetrainError("The A/B split has no labelled episodes")
    ab = [AbEpisode(e.upload_id, e.name, list(e.starts), e.texts, list(e.target), list(preds[e.upload_id][2]))
          for e in episodes]
    sheet, key = ab_sheet(ab, label, teacher.version_label, seed)
    key["order"] = [e.upload_id for e in episodes]
    out_dir = out_dir or default_ab_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path, key_path = out_dir / "ab_sheet.md", out_dir / "ab_key.json"
    sheet_path.write_text(sheet, encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
    return sheet_path, key_path


ANSWER_ROW = re.compile(r"^\|\s*(\d+):.*\|\s*(A|B|tie)\s*\|\s*$", re.IGNORECASE)


def parse_answers(sheet: str, key: dict) -> dict[str, str]:
    """The filled answer table -> {upload_id: 'A' | 'B' | 'tie'}; every episode must be answered."""
    order = key["order"]
    answers = {}
    for line in sheet.splitlines():
        if m := ANSWER_ROW.match(line.strip()):
            answers[str(order[int(m.group(1)) - 1])] = m.group(2)
    missing = [str(i) for i in order if str(i) not in answers]
    if missing:
        raise RetrainError(f"Unanswered episodes in the sheet: {len(missing)} of {len(order)}")
    return answers


async def score_ab_sheet(session_factory: async_sessionmaker[AsyncSession], label: str,
                         out_dir: Path | None = None) -> dict:
    out_dir = out_dir or default_ab_dir(label)
    key = json.loads((out_dir / "ab_key.json").read_text(encoding="utf-8"))
    if key["candidate"] != label:
        raise RetrainError(f"The key in {out_dir} is for {key['candidate']!r}, not {label!r}")
    result = score_ab(key, parse_answers((out_dir / "ab_sheet.md").read_text(encoding="utf-8"), key))
    result["scored_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    async with session_factory() as db:
        row = await _trained_version(db, label)
        row.metrics = {**(row.metrics or {}), "ab": result}
        await db.commit()
    return result


async def promotion_status(session_factory: async_sessionmaker[AsyncSession], label: str) -> dict:
    """Both halves of the 0024 §7 rule, from the stored metrics."""
    async with session_factory() as db:
        row = await _trained_version(db, label)
        live = await db.scalar(select(ModelVersion.version_label).where(
            ModelVersion.model_name == "relevance", ModelVersion.is_live))
        n_labels = await db.scalar(select(func.count()).select_from(RelevancePseudoLabel))
    m = row.metrics or {}
    gate = (m.get("test") or {}).get("gate", {}).get("passed")
    ab = (m.get("ab") or {}).get("passed")
    return {"label": label, "live": live, "offline_gate": gate, "ab": ab,
            "promotable": bool(gate) and bool(ab), "pseudo_labels": n_labels}
