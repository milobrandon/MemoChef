"""Unit tests for the SKILLS constant and skills wiring in api_client.

These tests don't hit the network. The live end-to-end test lives in
test_managed_agents_live.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from managed_agents import api_client
from managed_agents.skills import SKILLS


def test_skills_constant_shape():
    assert isinstance(SKILLS, list)
    assert len(SKILLS) >= 1
    for entry in SKILLS:
        assert set(entry.keys()) >= {"type", "skill_id"}
        assert entry["type"] in {"anthropic", "custom"}
        assert isinstance(entry["skill_id"], str) and entry["skill_id"]


def test_skills_includes_xlsx_and_pptx():
    ids = {s["skill_id"] for s in SKILLS if s["type"] == "anthropic"}
    assert "xlsx" in ids
    assert "pptx" in ids


class _FakeResp:
    def __init__(self, body: dict):
        self.status_code = 200
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


class _FakeClient:
    """Captures the JSON body of the first POST without hitting the network."""

    captured: dict | None = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None, **kwargs):
        type(self).captured = json
        return _FakeResp({"id": "agent_test", "version": 1})

    def get(self, url, headers=None, **kwargs):
        return _FakeResp({"version": 1})


def test_create_agent_passes_skills_in_body():
    _FakeClient.captured = None
    with patch.object(api_client.httpx, "Client", _FakeClient):
        api_client.create_agent(
            name="x",
            model="claude-sonnet-4-6",
            system="sys",
            tools=[{"type": "agent_toolset_20260401"}],
            skills=SKILLS,
        )
    assert _FakeClient.captured is not None
    assert _FakeClient.captured.get("skills") == SKILLS


def test_create_agent_omits_skills_when_none():
    _FakeClient.captured = None
    with patch.object(api_client.httpx, "Client", _FakeClient):
        api_client.create_agent(
            name="x",
            model="claude-sonnet-4-6",
            system="sys",
            tools=[{"type": "agent_toolset_20260401"}],
        )
    assert _FakeClient.captured is not None
    assert "skills" not in _FakeClient.captured


def test_update_agent_passes_skills_in_body():
    _FakeClient.captured = None
    with patch.object(api_client.httpx, "Client", _FakeClient):
        api_client.update_agent("agent_test", skills=SKILLS, version=1)
    assert _FakeClient.captured is not None
    assert _FakeClient.captured.get("skills") == SKILLS
    assert _FakeClient.captured.get("version") == 1
