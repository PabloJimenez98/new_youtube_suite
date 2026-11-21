"""Platform settings in `app_settings` (M16, docs/decisions/0025 §11)."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import registry
from app.models.app_setting import AppSetting

LLM_DEFAULT = "llm_default"

# Tools whose LLM can be chosen per run (job params `llm`).
LLM_TOOLS = ("chapters", "shownotes", "relevance")


async def get_llm_default(db: AsyncSession) -> str:
    row = await db.get(AppSetting, LLM_DEFAULT)
    value = row.value if row is not None else None
    return value if value in registry.CHOICES else registry.QWEN


async def set_llm_default(db: AsyncSession, value: str) -> None:
    if value not in registry.CHOICES:
        raise ValueError(f"llm_default must be one of {registry.CHOICES}")
    row = await db.get(AppSetting, LLM_DEFAULT)
    if row is None:
        db.add(AppSetting(key=LLM_DEFAULT, value=value))
    else:
        row.value = value
    await db.commit()


def job_llm(params: dict | None) -> str:
    """The LLM a job runs with: its `llm` param (filled in at creation), else Qwen (jobs from before M16)."""
    value = (params or {}).get("llm")
    return value if value in registry.CHOICES else registry.QWEN
