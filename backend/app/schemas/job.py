from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.models.job import JOB_TOOL_TYPES

# Derived from the model's tuple (not hand-duplicated) so the two
# can't drift apart.
JobToolType = Literal[JOB_TOOL_TYPES]  # type: ignore[valid-type]


class JobCreate(BaseModel):
    tool_type: JobToolType


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    upload_id: int
    tool_type: str
    status: str
    progress_pct: int | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    created_at: datetime
