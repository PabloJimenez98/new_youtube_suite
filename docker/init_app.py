"""One-off initialisation of the examiner stack (docker/compose.yml, service `init`).

Idempotent: safe to run on every `docker compose up`.

1. Apply the database migrations.
2. Create the operator account (OPERATOR_EMAIL / OPERATOR_PASSWORD) if it does not exist.
3. Register the trained relevance BiLSTM (`bilstm-e5s-20261114-i2`) and make it live, exactly as
   `app.cli relevance train` + `models set-live` did in M15 (docs/decisions/0024). Its stored
   training and test metrics come from the bundle's metrics.json.
4. Choose the default LLM: Qwen when its GGUF is installed (GPU mode), else our own model.
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BACKEND = Path("/app/backend")
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from sqlalchemy import select  # noqa: E402

from app.auth.security import hash_password  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.database import async_session_factory  # noqa: E402
from app.llm import registry  # noqa: E402
from app.model_registry import set_live  # noqa: E402
from app.models import ModelVersion, User  # noqa: E402
from app.settings_store import set_llm_default  # noqa: E402

RELEVANCE_LABEL = "bilstm-e5s-20261114-i2"


def log(msg: str) -> None:
    print(f"[init] {msg}", flush=True)


async def ensure_user(email: str, password: str) -> None:
    async with async_session_factory() as db:
        if await db.scalar(select(User.id).where(User.email == email)):
            log(f"operator account <{email}> exists")
            return
        db.add(User(email=email, password_hash=hash_password(password)))
        await db.commit()
        log(f"created operator account <{email}>")


async def ensure_relevance() -> None:
    bundle = (BACKEND / get_settings().relevance_model_root / RELEVANCE_LABEL).resolve()
    if not (bundle / "weights.pt").is_file():
        log(f"relevance bundle missing at {bundle}: the zero-shot teacher stays in use "
            "(run fetch_models.py first)")
        return
    async with async_session_factory() as db:
        row = await db.scalar(select(ModelVersion).where(ModelVersion.model_name == "relevance",
                                                         ModelVersion.version_label == RELEVANCE_LABEL))
        if row is None:
            config = json.loads((bundle / "config.json").read_text())
            metrics = json.loads((bundle / "metrics.json").read_text())
            training = config.get("training", {})
            db.add(ModelVersion(
                model_name="relevance",
                version_label=RELEVANCE_LABEL,
                architecture_notes=(
                    f"Frozen {config['encoder']['name']} window embeddings -> BiLSTM "
                    f"{config['model_config']} -> heads, combined {config['combine']}. "
                    "Installed from the v1.0 release. docs/decisions/0024-relevance-bilstm-m15.md"),
                training_data_reference={"kind": "distilled", "encoder": config["encoder"]["name"],
                                         **{k: training.get(k) for k in ("teacher", "split", "episodes", "lambda", "seed")},
                                         "combine_mode": config["combine"]["mode"]},
                metrics=metrics,
                artifact_path=str(bundle),
                is_live=False,
                trained_at=datetime.fromisoformat("2026-11-14T11:58:00+00:00"),
            ))
            await db.commit()
            log(f"registered relevance version {RELEVANCE_LABEL}")
        switch = await set_live(db, "relevance", RELEVANCE_LABEL)
        log(f"live relevance version: {switch.current}")


async def ensure_llm_default() -> None:
    """The default stays "qwen", the deployed path, in both modes.

    Without the Qwen GGUF (CPU mode), "qwen" still selects the deployed models where they need no
    LLM: chapter titles fall back to the fine-tuned mT5, and relevance uses the live BiLSTM. Show
    notes need an LLM, so in CPU mode their run is started with "Our model" (DELIVERY.md).
    """
    try:
        qwen = registry.resolve(registry.QWEN).path.is_file()
    except registry.LlmUnavailableError:
        qwen = False
    async with async_session_factory() as db:
        await set_llm_default(db, registry.QWEN)
    log(f"default LLM: qwen ({'installed' if qwen else 'not installed: mT5 titles, BiLSTM relevance'}); "
        f"own model {'installed' if registry.own_available() else 'missing'}")


async def main() -> None:
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    log("migrations applied")
    await ensure_user(os.environ.get("OPERATOR_EMAIL", "examiner@example.com"),
                      os.environ.get("OPERATOR_PASSWORD", "examiner"))
    await ensure_relevance()
    await ensure_llm_default()
    log(f"done (mode: {os.environ.get('YTS_MODE', 'cpu')})")


if __name__ == "__main__":
    asyncio.run(main())
