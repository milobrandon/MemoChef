from __future__ import annotations

import logging as _logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_log = _logging.getLogger(__name__)


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
    # College House SQL comp/market performance pull (any filter enables it)
    college_house_institution: str | None = None
    college_house_ipeds: int | None = None
    college_house_properties: list[str] = Field(default_factory=list)
    college_house_base_variant_only: bool = False
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


class SlideContent(BaseModel):
    """Content specification for a single slide to generate."""
    title: str
    section: str  # target memo section name
    insert_after_slide: int  # 1-based slide number
    content_type: str  # "table_and_narrative", "chart", "table", "narrative_only"
    source_refs: list[str] = Field(default_factory=list)  # e.g. ["supplemental:demographics.pdf"]
    visual_type: str | None = None  # "bar_chart", "line_chart", "pie_chart", "table"
    visual_data: dict = Field(default_factory=dict)  # categories, series, title
    narrative: str = ""
    rationale: str = ""


class SlidePlan(BaseModel):
    """Claude-generated plan for slides to create."""
    slides_to_generate: list[SlideContent] = Field(default_factory=list)


class DeckProfile(BaseModel):
    """Profile of an existing memo deck's structure and style."""
    sections: list[dict] = Field(default_factory=list)  # from detect_memo_sections
    total_slides: int = 0
    has_charts: bool = False
    has_tables: bool = False
    slide_layouts_used: list[str] = Field(default_factory=list)
    visual_types_present: list[str] = Field(default_factory=list)  # "chart", "table", "image"
    # Dominant formatting extracted from existing slides
    title_font_name: str | None = None
    title_font_size_pt: float | None = None
    body_font_name: str | None = None
    body_font_size_pt: float | None = None


# ── Market Data Update Schema ─────────────────────────────────────────────────

class ChartSeriesUpdate(BaseModel):
    name: str
    new_values: list[float | int | None]
    old_values: list[float | int | None] = Field(default_factory=list)


class ChartSeriesAdd(BaseModel):
    name: str
    values: list[float | int | None]


class MarketChartUpdate(BaseModel):
    type: Literal["chart_update"] = "chart_update"
    page: int
    chart_name: str | None = None
    chart_title: str | None = None
    series: list[ChartSeriesUpdate] = Field(default_factory=list)
    categories: list[str] | None = None
    add_series: list[ChartSeriesAdd] = Field(default_factory=list)
    remove_series: list[str] = Field(default_factory=list)
    source: str
    reasoning: str
    confidence: str = "high"  # "high" | "medium" | "low"


class MarketNarrativeUpdate(BaseModel):
    type: Literal["narrative_update"] = "narrative_update"
    page: int
    old_text: str
    new_text: str
    source: str
    reasoning: str
    confidence: str = "high"


class MarketTableCellUpdate(BaseModel):
    row: int  # 0-based, matching python-pptx table indexing
    col: int  # 0-based, matching python-pptx table indexing
    old_value: str
    new_value: str


class MarketTableUpdate(BaseModel):
    type: Literal["table_update"] = "table_update"
    page: int
    slide_table: str
    updates: list[MarketTableCellUpdate]
    source: str
    reasoning: str
    confidence: str = "high"


class MarketDataUpdateSet(BaseModel):
    market_data_updates: list[dict] = Field(default_factory=list)
    unmatched_memo_metrics: list[str] = Field(default_factory=list)
    unmatched_workbook_tabs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def chart_updates(self) -> list[MarketChartUpdate]:
        results = []
        for u in self.market_data_updates:
            if u.get("type") != "chart_update":
                continue
            try:
                results.append(MarketChartUpdate(**u))
            except Exception as exc:
                _log.warning("Skipping malformed chart_update: %s", exc)
        return results

    def narrative_updates(self) -> list[MarketNarrativeUpdate]:
        results = []
        for u in self.market_data_updates:
            if u.get("type") != "narrative_update":
                continue
            try:
                results.append(MarketNarrativeUpdate(**u))
            except Exception as exc:
                _log.warning("Skipping malformed narrative_update: %s", exc)
        return results

    def table_updates(self) -> list[MarketTableUpdate]:
        results = []
        for u in self.market_data_updates:
            if u.get("type") != "table_update":
                continue
            try:
                results.append(MarketTableUpdate(**u))
            except Exception as exc:
                _log.warning("Skipping malformed table_update: %s", exc)
        return results
