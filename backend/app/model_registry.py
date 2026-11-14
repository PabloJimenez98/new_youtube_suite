"""Which version of each custom model is live (docs/decisions/0024 §5).

`set_live` is the live-switch and the rollback: switching back to the
previous label is the rollback. It runs in one transaction: the old row is
unset before the new one is set, so the partial unique index (one live row
per model, 0012 §5) never sees two. Existing scores are not recomputed.
"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_version import MODEL_NAMES, ModelVersion


class VersionNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class LiveSwitch:
    model_name: str
    previous: str | None
    current: str

    @property
    def changed(self) -> bool:
        return self.previous != self.current


async def set_live(db: AsyncSession, model_name: str, label: str) -> LiveSwitch:
    if model_name not in MODEL_NAMES:
        raise VersionNotFoundError(f"Unknown model {model_name!r} (one of {', '.join(MODEL_NAMES)})")
    target = await db.scalar(
        select(ModelVersion).where(ModelVersion.model_name == model_name, ModelVersion.version_label == label)
    )
    if target is None:
        raise VersionNotFoundError(f"No {model_name} version {label!r}")
    if target.artifact_path is not None:
        # A trained model must be able to serve before it goes live.
        if model_name == "relevance":
            from app.relevance.bilstm import check_bundle

            check_bundle(Path(target.artifact_path))
        elif not Path(target.artifact_path).exists():
            raise FileNotFoundError(f"{label!r}: artifact missing at {target.artifact_path}")
    previous = await db.scalar(
        select(ModelVersion).where(ModelVersion.model_name == model_name, ModelVersion.is_live)
    )
    if previous is not None and previous.id == target.id:
        return LiveSwitch(model_name, label, label)
    await db.execute(
        update(ModelVersion).where(ModelVersion.model_name == model_name, ModelVersion.is_live).values(is_live=False)
    )
    await db.flush()
    target.is_live = True
    await db.commit()
    return LiveSwitch(model_name, previous.version_label if previous else None, label)


async def list_versions(db: AsyncSession, model_name: str | None = None) -> list[ModelVersion]:
    q = select(ModelVersion).order_by(ModelVersion.model_name, ModelVersion.id)
    if model_name is not None:
        q = q.where(ModelVersion.model_name == model_name)
    return list((await db.execute(q)).scalars())
