from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class SourceDirective(BaseModel):
    """User instruction attached to a specific uploaded source."""

    source_id: str  # e.g. "proforma:Unit Mix", "comp:hub-lexington", "supplemental"
    source_type: str  # proforma_tab | supplemental | comp_url | market_data | schedule
    directive: str  # free-text user instruction
    scope: str = "both"  # "mapping" | "slide_generation" | "both"


class CompUrl(BaseModel):
    url: str
    label: str = ""
    guidance: str = ""


class UnitMixEntry(BaseModel):
    unit_type: str
    beds: int | None = None
    baths: int | None = None
    sf: int | None = None
    rent: float | None = None
    rent_per_sf: float | None = None


class CompProperty(BaseModel):
    name: str
    address: str | None = None
    distance_mi: float | None = None
    unit_mix: list[UnitMixEntry] = Field(default_factory=list)
    total_units: int | None = None
    occupancy_pct: float | None = None
    year_built: int | None = None
    concessions: str | None = None
    source: str  # "url", "realpage", "csv", "manual"
    source_detail: str = ""


class CompSlideRequest(BaseModel):
    subject_property: CompProperty
    comps: list[CompProperty]
    sort_by: str = "distance"
    max_comps: int = 6
    include_narrative: bool = True


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
    property_rename_to: str | None = None
    schedule_path: str | None = None
    market_data_path: str | None = None
    supplemental_path: str | None = None
    supplemental_type: str | None = None  # "pdf", "url", "excel", "csv"
    supplemental_brief: str | None = None
    comp_urls: list[CompUrl] = Field(default_factory=list)
    comp_csv_path: str | None = None
    comp_manual_entries: list[dict] | None = None
    auto_generate_comp_slide: bool = False
    comp_max_comps: int = 6
    comp_sort_by: str = "distance"
    source_directives: list[SourceDirective] = Field(default_factory=list)
    dry_run: bool = False
    skip_validation: bool = False
    use_batch_api: bool = False
    resume_from_checkpoint: bool = True
    config_override_path: str | None = None

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
    property_rename_to: str | None = None
    dry_run: bool = False
    skip_validation: bool = False
    config_profile: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    accuracy: dict | None = None
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
    unvalidated_pages: list[int] = Field(default_factory=list)
    log_lines: list[str] = Field(default_factory=list)
