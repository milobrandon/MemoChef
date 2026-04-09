#!/usr/bin/env python3
"""
Session lifecycle helpers for Memo Chef managed agent.

Handles: file upload → session creation → event streaming → output retrieval.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from managed_agents.config import ANTHROPIC_API_KEY, AGENT_ID, ENVIRONMENT_ID, EXAMPLES_DIR


@dataclass
class SessionRun:
    """Tracks a single agent run."""

    session_id: str = ""
    file_ids: list[str] = field(default_factory=list)
    output_file_ids: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def upload_file(client: anthropic.Anthropic, file_path: Path) -> str:
    """Upload a file via the Files API and return the file ID."""
    with open(file_path, "rb") as f:
        result = client.beta.files.upload(file=f)
    return result.id


def upload_example_memos(client: anthropic.Anthropic) -> list[dict]:
    """Upload all example IC memos from the examples/ directory.

    Returns resource dicts suitable for session creation.
    """
    resources = []
    if not EXAMPLES_DIR.exists():
        return resources

    for path in sorted(EXAMPLES_DIR.glob("*.pptx")):
        file_id = upload_file(client, path)
        resources.append({
            "type": "file",
            "file_id": file_id,
            "mount_path": f"/mnt/examples/{path.name}",
        })
    return resources


def create_session(
    client: anthropic.Anthropic,
    *,
    uploaded_resources: list[dict],
    title: str = "Memo Chef Run",
) -> str:
    """Create a new session referencing the agent, environment, and uploaded files."""
    session = client.beta.sessions.create(
        agent=AGENT_ID,
        environment_id=ENVIRONMENT_ID,
        title=title,
        resources=uploaded_resources,
    )
    return session.id


def send_message(
    client: anthropic.Anthropic,
    session_id: str,
    message: str,
) -> None:
    """Send a user message event to the session."""
    client.beta.sessions.events.send(
        session_id,
        events=[
            {
                "type": "user.message",
                "content": [{"type": "text", "text": message}],
            },
        ],
    )


def stream_events(
    client: anthropic.Anthropic,
    session_id: str,
) -> Generator[dict, None, None]:
    """Stream SSE events from the session, yielding parsed event dicts.

    Each yielded dict has at minimum a "type" key. Additional keys depend
    on the event type:
    - agent.message: "text" (concatenated text blocks)
    - agent.tool_use: "name", "input"
    - agent.tool_result: "content"
    - session.status_idle: "stop_reason"
    - session.error: "error"
    """
    with client.beta.sessions.events.stream(session_id) as stream:
        for event in stream:
            evt = {"type": event.type}

            if event.type == "agent.message":
                texts = []
                if hasattr(event, "content"):
                    for block in event.content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                evt["text"] = "".join(texts)

            elif event.type == "agent.tool_use":
                evt["name"] = getattr(event, "name", "")
                evt["input"] = getattr(event, "input", {})

            elif event.type == "agent.tool_result":
                evt["content"] = getattr(event, "content", "")

            elif event.type == "session.status_idle":
                evt["stop_reason"] = getattr(event, "stop_reason", "")

            elif event.type == "session.error":
                err = getattr(event, "error", {})
                evt["error"] = err if isinstance(err, dict) else str(err)

            yield evt

            if event.type in ("session.status_idle", "session.status_terminated"):
                break


def get_output_files(
    client: anthropic.Anthropic,
    session_id: str,
) -> list[dict]:
    """List files scoped to the session (i.e. files the agent created)."""
    result = client.beta.files.list(scope_id=session_id)
    return [
        {
            "id": f.id,
            "filename": f.filename,
            "size_bytes": f.size_bytes,
            "downloadable": getattr(f, "downloadable", True),
        }
        for f in result.data
    ]


def download_file(client: anthropic.Anthropic, file_id: str, dest: Path) -> Path:
    """Download a file from the Files API to a local path."""
    response = client.beta.files.download(file_id)
    # The SDK returns a BinaryAPIResponse; read bytes from it
    dest.write_bytes(response.read())
    return dest


def build_user_message(
    *,
    proforma_filename: str,
    memo_filename: str,
    supplemental_filenames: list[str] | None = None,
    instructions: str = "",
) -> str:
    """Build the initial user message that kicks off the agent run."""
    parts = [
        f"Please update the IC memo using the proforma data.",
        f"",
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
        "Write the updated memo to `/mnt/session/uploads/output.pptx` and a changelog to `/mnt/session/uploads/changelog.md`.",
    ])

    return "\n".join(parts)
