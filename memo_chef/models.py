from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class StageUpdate(BaseModel):
    key: str
    label: str
    percent: int = Field(ge=0, le=100)
    detail: str = ""


class RunRequest(BaseModel):
    memo_path: str
    proforma_path: str
    output_dir: str
    api_key: str
    config_path: str
    run_id: str
    property_name: str | None = None
    schedule_path: str | None = None
    market_data_path: str | None = None
    dry_run: bool = False
    skip_validation: bool = False
    resume_from_checkpoint: bool = True

    @property
    def memo_name(self) -> str:
        return Path(self.memo_path).name

    @property
    def proforma_name(self) -> str:
        return Path(self.proforma_path).name


class RunWarning(BaseModel):
    stage: str
    message: str


class StageRecord(BaseModel):
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    detail: str = ""


class RunManifest(BaseModel):
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "running"
    memo_name: str
    proforma_name: str
    property_name: str | None = None
    dry_run: bool = False
    skip_validation: bool = False
    outputs: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[RunWarning] = Field(default_factory=list)
    stages: dict[str, StageRecord] = Field(default_factory=dict)


class RunResult(BaseModel):
    manifest: RunManifest
    memo_path: str
    log_path: str
    manifest_path: str
    memo_bytes: bytes
    log_bytes: bytes
    manifest_bytes: bytes
    changes: list[dict] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    missed: list[dict] = Field(default_factory=list)
    log_lines: list[str] = Field(default_factory=list)
