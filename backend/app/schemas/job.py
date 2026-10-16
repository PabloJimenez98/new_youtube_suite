from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.models.job import JOB_TOOL_TYPES

# Derived from the model's tuple (not hand-duplicated) so the two
# can't drift apart.
JobToolType = Literal[JOB_TOOL_TYPES]  # type: ignore[valid-type]


class ShortsParams(BaseModel):
    """Options for a `shorts` run (docs/decisions/0013 §5)."""

    model_config = ConfigDict(extra="forbid")

    aspect_ratio: Literal["9:16", "1:1", "4:5", "16:9"] = "9:16"
    captions: bool = True


# Tools that take per-run params, and their schema. Every other tool takes none.
PARAMS_BY_TOOL: dict[str, type[BaseModel]] = {"shorts": ShortsParams}


class JobCreate(BaseModel):
    tool_type: JobToolType
    params: dict | None = None

    @model_validator(mode="after")
    def _validate_params(self) -> "JobCreate":
        schema = PARAMS_BY_TOOL.get(self.tool_type)
        if schema is None:
            if self.params:
                raise ValueError(f"'{self.tool_type}' jobs take no params")
            self.params = None
        else:
            # Stored with defaults filled in, so a re-run reproduces the run.
            self.params = schema.model_validate(self.params or {}).model_dump()
        return self


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
    params: dict | None = None
    created_at: datetime
