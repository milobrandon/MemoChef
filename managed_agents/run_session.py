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
from managed_agents.config import AGENT_ID, ENVIRONMENT_ID, EXAMPLES_DIR


def upload_file(file_path: Path) -> str:
    """Upload a file via the Files API and return the file ID."""
    return api_upload_file(file_path)


def upload_example_memos() -> list[dict]:
    """Upload all example IC memos from the examples/ directory.

    Returns resource dicts suitable for session creation.
    """
    resources = []
    if not EXAMPLES_DIR.exists():
        return resources

    for path in sorted(EXAMPLES_DIR.glob("*.pptx")):
        file_id = upload_file(path)
        resources.append({
            "type": "file",
            "file_id": file_id,
            "mount_path": f"/mnt/examples/{path.name}",
        })
    return resources


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
    proforma_filename: str,
    memo_filename: str,
    supplemental_filenames: list[str] | None = None,
    instructions: str = "",
) -> str:
    """Build the initial user message that kicks off the agent run."""
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

    if instructions:
        parts.append("")
        parts.append(f"**Additional instructions**: {instructions}")

    parts.extend([
        "",
        "Example IC memos (for house style reference) are mounted at `/mnt/examples/`.",
        "",
        "Write the updated memo to `/mnt/session/uploads/output.pptx` and a changelog "
        "to `/mnt/session/uploads/changelog.md`.",
    ])

    return "\n".join(parts)
