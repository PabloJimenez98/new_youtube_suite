"""Platform settings API (M16): which LLM the chapter-titles, show-notes and relevance tools use
by default. See docs/decisions/0025-own-llm-m16.md §11."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_current_user
from app.database import get_db
from app.llm import registry
from app.models.user import User
from app.settings_store import get_llm_default, set_llm_default

router = APIRouter(prefix="/settings", tags=["settings"])


class LlmSettingsOut(BaseModel):
    default: Literal["qwen", "own"]
    own_available: bool
    own_label: str | None
    qwen_label: str | None


class LlmSettingsIn(BaseModel):
    default: Literal["qwen", "own"]


async def _out(db: AsyncSession) -> LlmSettingsOut:
    try:
        qwen = registry.resolve(registry.QWEN).label
    except registry.LlmUnavailableError:
        qwen = None
    return LlmSettingsOut(default=await get_llm_default(db), own_available=registry.own_available(),
                          own_label=registry.own_label(), qwen_label=qwen)


@router.get("/llm", response_model=LlmSettingsOut)
async def get_llm_settings(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    return await _out(db)


@router.put("/llm", response_model=LlmSettingsOut)
async def put_llm_settings(body: LlmSettingsIn, db: AsyncSession = Depends(get_db),
                           _user: User = Depends(get_current_user)):
    if body.default == registry.OWN and not registry.own_available():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Our own LLM isn't installed on this machine")
    await set_llm_default(db, body.default)
    return await _out(db)
