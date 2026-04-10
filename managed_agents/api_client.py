"""
Raw HTTP client for the Managed Agents API.

The Anthropic Python SDK (v0.79) doesn't yet have beta.agents/environments/
sessions namespaces, so we use httpx directly for those endpoints while
falling back to the SDK for the Files API (which IS supported via beta.files).
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx

from managed_agents.config import ANTHROPIC_API_KEY

BASE_URL = "https://api.anthropic.com"
BETA_HEADER = "managed-agents-2026-04-01"
FILES_BETA = "files-api-2025-04-14"

_HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "anthropic-beta": BETA_HEADER,
    "content-type": "application/json",
}


def _headers(extra_betas: list[str] | None = None) -> dict[str, str]:
    """Build request headers, optionally appending extra beta flags."""
    h = dict(_HEADERS)
    if not h["x-api-key"]:
        h["x-api-key"] = ANTHROPIC_API_KEY  # reload in case env changed
    if extra_betas:
        h["anthropic-beta"] = ",".join([BETA_HEADER] + extra_betas)
    return h


def _check(resp: httpx.Response) -> dict:
    """Raise on HTTP errors and return parsed JSON."""
    if resp.status_code >= 400:
        raise RuntimeError(
            f"API error {resp.status_code}: {resp.text}"
        )
    return resp.json()


# ── Agents ──────────────────────────────────────────────────────────

def create_agent(
    *,
    name: str,
    model: str,
    system: str,
    tools: list[dict],
    description: str = "",
) -> dict:
    """POST /v1/agents — create a managed agent."""
    body: dict[str, Any] = {
        "name": name,
        "model": model,
        "system": system,
        "tools": tools,
    }
    if description:
        body["description"] = description
    with httpx.Client(timeout=30) as c:
        resp = c.post(f"{BASE_URL}/v1/agents", headers=_headers(), json=body)
    return _check(resp)


def get_agent(agent_id: str) -> dict:
    """GET /v1/agents/{id}"""
    with httpx.Client(timeout=30) as c:
        resp = c.get(f"{BASE_URL}/v1/agents/{agent_id}", headers=_headers())
    return _check(resp)


# ── Environments ────────────────────────────────────────────────────

def create_environment(
    *,
    name: str,
    config: dict,
) -> dict:
    """POST /v1/environments — create a cloud environment."""
    with httpx.Client(timeout=30) as c:
        resp = c.post(
            f"{BASE_URL}/v1/environments",
            headers=_headers(),
            json={"name": name, "config": config},
        )
    return _check(resp)


def get_environment(env_id: str) -> dict:
    """GET /v1/environments/{id}"""
    with httpx.Client(timeout=30) as c:
        resp = c.get(f"{BASE_URL}/v1/environments/{env_id}", headers=_headers())
    return _check(resp)


# ── Files ───────────────────────────────────────────────────────────

def upload_file(file_path: Path) -> str:
    """Upload a file via POST /v1/files and return the file ID."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY or "",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": f"{BETA_HEADER},{FILES_BETA}",
    }
    with httpx.Client(timeout=120) as c:
        with open(file_path, "rb") as f:
            resp = c.post(
                f"{BASE_URL}/v1/files",
                headers=headers,
                files={"file": (file_path.name, f)},
            )
    data = _check(resp)
    return data["id"]


def list_files(scope_id: str | None = None) -> list[dict]:
    """GET /v1/files — list files, optionally filtered by scope."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY or "",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": f"{BETA_HEADER},{FILES_BETA}",
    }
    params = {}
    if scope_id:
        params["scope_id"] = scope_id
    with httpx.Client(timeout=30) as c:
        resp = c.get(f"{BASE_URL}/v1/files", headers=headers, params=params)
    data = _check(resp)
    return data.get("data", [])


def download_file(file_id: str, dest: Path) -> Path:
    """GET /v1/files/{id}/content — download file bytes."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY or "",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": f"{BETA_HEADER},{FILES_BETA}",
    }
    with httpx.Client(timeout=120) as c:
        resp = c.get(f"{BASE_URL}/v1/files/{file_id}/content", headers=headers)
    if resp.status_code >= 400:
        raise RuntimeError(f"Download error {resp.status_code}: {resp.text}")
    dest.write_bytes(resp.content)
    return dest


# ── Sessions ────────────────────────────────────────────────────────

def create_session(
    *,
    agent_id: str,
    environment_id: str,
    title: str = "",
    resources: list[dict] | None = None,
) -> dict:
    """POST /v1/sessions — create a new session."""
    body: dict[str, Any] = {
        "agent": agent_id,
        "environment_id": environment_id,
    }
    if title:
        body["title"] = title
    if resources:
        body["resources"] = resources
    with httpx.Client(timeout=60) as c:
        resp = c.post(f"{BASE_URL}/v1/sessions", headers=_headers(), json=body)
    return _check(resp)


def send_events(session_id: str, events: list[dict]) -> dict:
    """POST /v1/sessions/{id}/events — send user events."""
    with httpx.Client(timeout=30) as c:
        resp = c.post(
            f"{BASE_URL}/v1/sessions/{session_id}/events",
            headers=_headers(),
            json={"events": events},
        )
    return _check(resp)


def send_user_message(session_id: str, text: str) -> dict:
    """Send a user.message event with text content."""
    return send_events(session_id, [
        {
            "type": "user.message",
            "content": [{"type": "text", "text": text}],
        },
    ])


def stream_events(session_id: str) -> Generator[dict, None, None]:
    """GET /v1/sessions/{id}/stream — SSE stream of events.

    Yields parsed event dicts. Stops on session.status_idle or terminated.
    """
    headers = dict(_headers())
    headers["Accept"] = "text/event-stream"
    # Remove content-type for GET
    headers.pop("content-type", None)

    with httpx.Client(timeout=None) as c:
        with c.stream(
            "GET",
            f"{BASE_URL}/v1/sessions/{session_id}/stream",
            headers=headers,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"Stream error {resp.status_code}")

            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")

                    if not line.startswith("data: "):
                        continue

                    json_str = line[6:]
                    if not json_str:
                        continue

                    try:
                        event = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    yield event

                    if event.get("type") in (
                        "session.status_idle",
                        "session.status_terminated",
                    ):
                        return


def get_session(session_id: str) -> dict:
    """GET /v1/sessions/{id}"""
    with httpx.Client(timeout=30) as c:
        resp = c.get(f"{BASE_URL}/v1/sessions/{session_id}", headers=_headers())
    return _check(resp)
