"""Pydantic request bodies for the control endpoints.

These are validation boundaries, not scheduling logic: a request
either names a scenario id, a speed the UI is allowed to pick, a
YES/NO answer, a login username, or the four plain fields a GPU
request needs - FastAPI/pydantic reject anything else (wrong type,
wrong literal, missing field) before it ever reaches
`SimulationSession`.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from api.config import ALLOWED_SPEEDS, MAX_USER_GPU_REQUEST, MIN_USER_GPU_REQUEST


class SetSpeedRequest(BaseModel):
    speed: float = Field(..., description=f"One of {ALLOWED_SPEEDS}")


class StepRequest(BaseModel):
    minutes: Optional[float] = Field(None, gt=0, description="Simulated minutes to advance; defaults to one base tick")


class PromptResponseRequest(BaseModel):
    gpu_id: str
    response: Literal["YES", "NO"]


class LoginRequest(BaseModel):
    username: str


class GPURequestBody(BaseModel):
    workload: str = Field(..., min_length=1, max_length=64)
    gpu_count: int = Field(1, ge=MIN_USER_GPU_REQUEST, le=MAX_USER_GPU_REQUEST)
    estimated_minutes: float = Field(..., gt=0, le=10_000)
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class ManualAssignRequest(BaseModel):
    """Admin-only demo/test control (Phase 10) - place a user directly
    onto a currently-free GPU to establish a starting arrangement
    before a demonstration begins. ``display_name`` lets the admin
    seed a user the engine has never seen before (mirrors
    `GPURequestBody`'s auto-registration via the login account's own
    display name for a real request)."""

    gpu_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1, max_length=64)
