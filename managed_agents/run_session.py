#!/usr/bin/env python3
"""
Session lifecycle helpers for Memo Chef managed agent.

Handles: file upload -> session creation -> event streaming -> output retrieval.
Uses the raw HTTP API client since the SDK doesn't yet support managed agents.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from managed_agents.api_client import (
    create_session as api_create_session,
    download_file as api_download_file,
    list_files as api_list_files,
    send_user_message as api_send_message,
    stream_events as api_stream_events,
    upload_file as api_upload_file,
)
from managed_agents.config import AGENT_ID, ENVIRONMENT_ID, FIREFLIES_API_KEY
from managed_agents.examples_cache import resolve_examples


def upload_file(file_path: Path) -> str:
    """Upload a file via the Files API and return the file ID."""
    return api_upload_file(file_path)


def upload_example_memos() -> list[dict]:
    """Return resource dicts for every example IC memo.

    Examples are uploaded once (via `bootstrap_examples.py` or
    lazily on first session) and cached in `.examples.json`. This
    function reuses cached file_ids whenever possible — only files
    whose on-disk sha256 changed, or whose cached file_id has
    expired server-side, get re-uploaded.

    Returns the same resource shape the API expects in
    `create_session(resources=...)`.
    """
    return resolve_examples()


def upload_fireflies_config(
    *,
    lookback_days: int = 90,
    search_terms: list[str] | None = None,
    api_key_override: str | None = None,
) -> dict | None:
    """Create and upload a Fireflies config JSON for the agent.

    Returns a resource dict for session creation, or None if no key is configured.
    api_key_override takes precedence over the FIREFLIES_API_KEY env var.
    """
    effective_key = api_key_override or FIREFLIES_API_KEY
    if not effective_key:
        return None

    import json
    import tempfile

    config = {
        "api_key": effective_key,
        "lookback_days": lookback_days,
        "search_terms": search_terms or [],
    }

    config_path = Path(tempfile.mktemp(suffix=".json", prefix="fireflies_"))
    config_path.write_text(json.dumps(config, indent=2))

    file_id = upload_file(config_path)
    config_path.unlink(missing_ok=True)

    return {
        "type": "file",
        "file_id": file_id,
        "mount_path": "/mnt/session/uploads/fireflies_config.json",
    }


def create_session(
    *,
    uploaded_resources: list[dict],
    title: str = "Memo Chef Run",
) -> str:
    """Create a new session referencing the agent, environment, and uploaded files."""
    session = api_create_session(
        agent_id=AGENT_ID,
        environment_id=ENVIRONMENT_ID,
        title=title,
        resources=uploaded_resources,
    )
    return session["id"]


def send_message(session_id: str, message: str) -> None:
    """Send a user message event to the session."""
    api_send_message(session_id, message)


def stream_events(session_id: str) -> Generator[dict, None, None]:
    """Stream SSE events from the session, yielding parsed event dicts.

    Each yielded dict has at minimum a "type" key. Additional keys depend
    on the event type (text, name, error, stop_reason, etc.).
    """
    yield from api_stream_events(session_id)


def get_output_files(session_id: str) -> list[dict]:
    """List files scoped to the session (i.e. files the agent created)."""
    files = api_list_files(scope_id=session_id)
    return [
        {
            "id": f["id"],
            "filename": f.get("filename", "unknown"),
            "size_bytes": f.get("size_bytes", 0),
            "downloadable": f.get("downloadable", True),
        }
        for f in files
    ]


def download_file_to(file_id: str, dest: Path) -> Path:
    """Download a file from the Files API to a local path."""
    return api_download_file(file_id, dest)


def build_user_message(
    *,
    proforma_filename: str | None,
    memo_filename: str,
    supplemental_filenames: list[str] | None = None,
    instructions: str = "",
    meeting_lookback_days: int | None = None,
    property_name: str | None = None,
    market_tabs: list[str] | None = None,
    generate_market_slides: bool = False,
    college_house_extract_filename: str | None = None,
    college_house_institution: str | None = None,
) -> str:
    """Build the initial user message that kicks off the agent run.

    If proforma_filename is None, the run is narrative-only: the agent
    should update narrative sections (entitlement, DD, design, schedule,
    market context) using supplemental files and meeting transcripts, and
    leave proforma-driven tables/financials untouched.
    """
    if proforma_filename is None:
        parts = [
            "Please update the IC memo's narrative sections ONLY. "
            "Do NOT modify any financial tables, returns, budgets, or "
            "proforma-driven numbers — no proforma is provided for this run.",
            "",
            f"**Memo template**: `/mnt/session/uploads/{memo_filename}`",
        ]
    else:
        parts = [
            "Please update the IC memo using the proforma data.",
            "",
            f"**Proforma**: `/mnt/session/uploads/{proforma_filename}`",
            f"**Memo template**: `/mnt/session/uploads/{memo_filename}`",
        ]

    if supplemental_filenames:
        parts.append("")
        parts.append("**Supplemental files**:")
        for name in supplemental_filenames:
            parts.append(f"- `/mnt/session/uploads/{name}`")

    if meeting_lookback_days:
        parts.extend([
            "",
            f"**Meeting transcripts**: A Fireflies config is mounted at "
            f"`/mnt/session/uploads/fireflies_config.json`. Search the last "
            f"**{meeting_lookback_days} days** of meetings for due diligence, "
            f"entitlements, design, and schedule updates relevant to this deal. "
            f"Use transcript insights to enrich narrative sections of the memo.",
        ])

    if property_name:
        parts.extend([
            "",
            f"**Property**: {property_name}. This run is scoped to this "
            f"property only. When transcripts, documents, or supplemental "
            f"files cover multiple properties, apply updates ONLY in the "
            f"context of {property_name}; do not attribute content from "
            f"other properties.",
        ])

    if market_tabs:
        tab_list = ", ".join(f"`{t}`" for t in market_tabs)
        parts.extend([
            "",
            f"**Market analysis workbook — approved tabs**: {tab_list}. "
            f"These are the ONLY tabs from the market workbook you should "
            f"read. Do NOT open or reference any other tabs (back-end data "
            f"like raw IPEDS, RealPage dumps, manual data tables, or raw "
            f"supply/demand are out of scope even if they appear in the "
            f"file). If a tab you need isn't in this list, surface it as "
            f"a changelog warning — don't read it anyway.",
        ])

    if college_house_extract_filename:
        scope = (
            f"for **{college_house_institution}**" if college_house_institution
            else "for the subject market"
        )
        parts.extend([
            "",
            f"**College House comp & market performance**: a live extract "
            f"from Subtext's College House research database {scope} is "
            f"mounted at `/mnt/session/uploads/{college_house_extract_filename}`. "
            f"It has two sheets: `Comp Performance Summary` (latest month per "
            f"property — total beds, prelease %, occupancy %, rate per bed, "
            f"rate per SF, and YoY rent growth, bed-weighted across bedroom "
            f"types) and `Monthly Raw Data` (full monthly time series by "
            f"property and bedroom count, with numerator/denominator columns "
            f"and derived PreleasePct / OccupancyPct / RatePerBed / "
            f"RatePerSF). Comp rent growth MUST use the leasing-cycle-average "
            f"convention: average rents from September through the current "
            f"month vs the same window one year prior — the summary sheet's "
            f"YoY Rent Growth column is already computed this way; do not "
            f"substitute single-month or calendar-year comparisons. Use it "
            f"as the authoritative CURRENT source for comp-table performance "
            f"columns (prelease, occupancy, market rents) and for market "
            f"performance narrative/charts — it is fresher than any static "
            f"market workbook tabs covering the same metrics. Percentages "
            f"are decimals (0.93 = 93%). Treat it like market data: apply "
            f"the market-workbook skill's plausibility bounds and "
            f"cross-slide propagation rules.",
        ])

    if generate_market_slides:
        parts.extend([
            "",
            "**Generate new market research slides**: ON. If the memo's "
            "market research section is thinner than the approved workbook "
            "tabs support, you may insert new slides into that section "
            "using the house style from `/mnt/examples/`. Insert slides "
            "adjacent to related existing content; do not append to the "
            "end of the deck. Narrative on new slides should cite specific "
            "years and deltas (e.g. \"rent growth accelerated from 1.6% "
            "in 2021 to 17.8% in 2023\").",
        ])
    elif market_tabs:
        parts.extend([
            "",
            "**Generate new market research slides**: OFF. Update existing "
            "market research slides only. Do NOT insert new slides even if "
            "the workbook has data the memo doesn't currently show.",
        ])

    if instructions:
        parts.append("")
        parts.append(f"**Additional instructions**: {instructions}")

    # Example memos are only loaded for full-proforma runs; reference them
    # only when the caller actually mounted them.
    if proforma_filename is not None:
        parts.extend([
            "",
            "Example IC memos (for house style reference) are mounted at `/mnt/examples/`.",
        ])

    parts.extend([
        "",
        "Write the updated memo to `/mnt/session/uploads/output.pptx` and a changelog "
        "to `/mnt/session/uploads/changelog.md`.",
    ])

    return "\n".join(parts)
