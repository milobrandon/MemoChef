"""Unit tests for the custom-skills migration.

Covers:
  - SKILL.md frontmatter validity per the Agent Skills spec.
  - Manifest ↔ filesystem consistency.
  - Skills cache load/save round-trip.
  - build_skills_list() merges Anthropic + custom entries from the cache.
  - api_client._skill_files_payload preserves the common-root contract.
  - api_client.create_skill / create_skill_version use the right beta
    header, endpoint, and multipart shape — mocked, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from managed_agents import api_client, skill_manifest, skills as skills_module
from managed_agents.skill_manifest import SKILL_SPECS, SKILLS_CONTENT_DIR

_NAME_RE = re.compile(r"^[a-z0-9-]+$")
_RESERVED_WORDS = {"anthropic", "claude"}


def _parse_frontmatter(skill_md: Path) -> dict[str, str]:
    """Minimal YAML-frontmatter parser (key: value lines between --- markers).

    SKILL.md frontmatter is always two simple keys, so we don't pull in
    a full YAML lib for the test.
    """
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{skill_md}: missing opening --- marker")
    body = text[4:]
    end = body.find("\n---\n")
    if end < 0:
        raise ValueError(f"{skill_md}: missing closing --- marker")
    block = body[:end]
    fm: dict[str, str] = {}
    for raw in block.splitlines():
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError(f"{skill_md}: malformed frontmatter line: {raw!r}")
        key, _, val = raw.partition(":")
        fm[key.strip()] = val.strip()
    return fm


# ── Frontmatter lint ───────────────────────────────────────────────

@pytest.mark.parametrize("spec", SKILL_SPECS, ids=lambda s: s.name)
def test_skill_md_exists(spec):
    assert (spec.source_dir / "SKILL.md").is_file()


@pytest.mark.parametrize("spec", SKILL_SPECS, ids=lambda s: s.name)
def test_skill_frontmatter_valid(spec):
    fm = _parse_frontmatter(spec.source_dir / "SKILL.md")
    assert "name" in fm and "description" in fm

    name = fm["name"]
    assert name == spec.name, f"frontmatter name {name!r} != dir {spec.name!r}"
    assert len(name) <= 64
    assert _NAME_RE.match(name), f"name {name!r} must be lowercase-hyphen-numeric"
    assert not any(w in name for w in _RESERVED_WORDS)
    assert "<" not in name and ">" not in name

    desc = fm["description"]
    assert 1 <= len(desc) <= 1024
    assert "<" not in desc and ">" not in desc


# ── Manifest ↔ filesystem ──────────────────────────────────────────

def test_every_content_dir_has_a_spec():
    manifest_names = {s.name for s in SKILL_SPECS}
    on_disk = {p.name for p in SKILLS_CONTENT_DIR.iterdir() if p.is_dir()}
    extras = on_disk - manifest_names
    assert not extras, f"skill dirs without manifest entries: {extras}"


# ── Cache I/O ──────────────────────────────────────────────────────

def test_cache_round_trip(tmp_path, monkeypatch):
    cache_file = tmp_path / "skills.json"
    monkeypatch.setattr(skill_manifest, "SKILLS_CACHE_FILE", cache_file)

    assert skill_manifest.load_cache() == {}

    skill_manifest.save_cache({"memo-table-updates": "skill_abc"})
    loaded = skill_manifest.load_cache()
    assert loaded == {"memo-table-updates": "skill_abc"}
    # round-trips through JSON
    assert json.loads(cache_file.read_text()) == loaded


def test_cache_handles_corrupt_file(tmp_path, monkeypatch):
    cache_file = tmp_path / "skills.json"
    cache_file.write_text("{not valid json")
    monkeypatch.setattr(skill_manifest, "SKILLS_CACHE_FILE", cache_file)
    assert skill_manifest.load_cache() == {}


# ── build_skills_list() ────────────────────────────────────────────

def test_build_skills_list_anthropic_only_when_cache_empty(monkeypatch):
    monkeypatch.setattr(skills_module, "load_cache", lambda: {})
    out = skills_module.build_skills_list()
    assert {e["skill_id"] for e in out} == {"xlsx", "pptx"}
    assert all(e["type"] == "anthropic" for e in out)


def test_build_skills_list_merges_custom_entries(monkeypatch):
    fake_cache = {
        "memo-table-updates": "skill_one",
        "layout-integrity": "skill_two",
    }
    monkeypatch.setattr(skills_module, "load_cache", lambda: fake_cache)
    out = skills_module.build_skills_list()

    anthropic = [e for e in out if e["type"] == "anthropic"]
    custom = [e for e in out if e["type"] == "custom"]
    assert {e["skill_id"] for e in anthropic} == {"xlsx", "pptx"}
    assert {e["skill_id"] for e in custom} == {"skill_one", "skill_two"}
    for e in custom:
        assert e["version"] == "latest"


def test_build_skills_list_skips_uncached(monkeypatch):
    monkeypatch.setattr(skills_module, "load_cache", lambda: {"memo-table-updates": "skill_abc"})
    out = skills_module.build_skills_list()
    custom = [e for e in out if e["type"] == "custom"]
    assert len(custom) == 1
    assert custom[0]["skill_id"] == "skill_abc"


# ── _skill_files_payload ───────────────────────────────────────────

def test_skill_files_payload_uses_dir_basename_as_root(tmp_path):
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: x\n---\nhi\n")
    (skill_dir / "extra.md").write_text("more")

    files = api_client._skill_files_payload(skill_dir)
    upload_names = sorted(name for _, (name, _bytes) in files)
    assert upload_names == ["test-skill/SKILL.md", "test-skill/extra.md"]


def test_skill_files_payload_requires_skill_md(tmp_path):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "other.md").write_text("nope")
    with pytest.raises(FileNotFoundError, match="SKILL.md"):
        api_client._skill_files_payload(skill_dir)


def test_skill_files_payload_requires_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        api_client._skill_files_payload(tmp_path / "missing")


# ── api_client skill HTTP shape ────────────────────────────────────

class _FakeSkillResp:
    def __init__(self, body: dict, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


class _FakeSkillClient:
    """Captures the URL/headers/files of skill API calls."""

    captured: dict | None = None
    response_body: dict = {"id": "skill_abc", "latest_version": "v1"}

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, data=None, files=None, **kwargs):
        type(self).captured = {
            "url": url,
            "headers": dict(headers or {}),
            "data": dict(data or {}),
            "files": list(files or []),
        }
        return _FakeSkillResp(type(self).response_body)


def test_create_skill_posts_multipart_with_skills_beta(tmp_path):
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: x\n---\n")
    _FakeSkillClient.captured = None
    _FakeSkillClient.response_body = {"id": "skill_demo", "latest_version": "1"}

    with patch.object(api_client.httpx, "Client", _FakeSkillClient):
        result = api_client.create_skill(display_title="Demo", skill_dir=skill_dir)

    assert result == {"id": "skill_demo", "latest_version": "1"}
    cap = _FakeSkillClient.captured
    assert cap is not None
    assert cap["url"].endswith("/v1/skills")
    assert cap["headers"]["anthropic-beta"] == "skills-2025-10-02"
    assert cap["data"] == {"display_title": "Demo"}
    upload_names = sorted(name for _, (name, _b) in cap["files"])
    assert upload_names == ["demo-skill/SKILL.md"]


def test_create_skill_version_targets_versions_endpoint(tmp_path):
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: x\n---\n")
    _FakeSkillClient.captured = None
    _FakeSkillClient.response_body = {"skill_id": "skill_xyz", "version": "9"}

    with patch.object(api_client.httpx, "Client", _FakeSkillClient):
        result = api_client.create_skill_version("skill_xyz", skill_dir=skill_dir)

    assert result == {"skill_id": "skill_xyz", "version": "9"}
    cap = _FakeSkillClient.captured
    assert cap["url"].endswith("/v1/skills/skill_xyz/versions")
    assert cap["headers"]["anthropic-beta"] == "skills-2025-10-02"
    # version-bump does not need display_title
    assert cap["data"] == {}
