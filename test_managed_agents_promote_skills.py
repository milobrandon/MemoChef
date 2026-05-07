"""Unit tests for the close-out promotion workflow.

Covers:
  - Parsing pending_skill_updates.md (well-formed, malformed, empty).
  - append_to_skill_md: inserts under existing heading, creates heading
    if missing, dedupes on rerun.
  - Decisions ↔ promotions: only approved/edited/reassigned promote.
  - apply_decisions in dry-run does not mutate disk and does not call
    create_skill_version / update_agent.
  - apply_decisions live-run patches the skill, calls create_skill_version
    once per modified skill, calls update_agent once.
  - Audit log written; already_processed gates re-runs.
"""

from __future__ import annotations

import json

import pytest

from managed_agents import promote_skills
from managed_agents.promote_skills import (
    Decision,
    Entry,
    apply_decisions,
    append_to_skill_md,
    parse_entries,
    write_audit_log,
)
from managed_agents.skill_manifest import SKILL_SPECS

VALID_NAME = SKILL_SPECS[0].name  # any real manifest name


# ── Parsing ────────────────────────────────────────────────────────

def test_parse_two_well_formed_entries():
    text = """\
# Pending Skill Updates

## Entry 1
**target_skill:** memo-table-updates

**Rule:** Always reapply rPr after writing subtotal cells.

**Why:** During Limestone Q2 the user flagged black-on-dark Total rows.

**How to apply:** Run the font-color regression check on subtotal rows specifically.

## Entry 2
**target_skill:** layout-integrity

**Rule:** Continuation slides must inherit the original section banner.

**Why:** Continuation slides without banners read as orphan slides.

**How to apply:** When splitting a slide, copy the banner shape onto the new slide.
"""
    entries, warnings = parse_entries(text)
    assert warnings == []
    assert len(entries) == 2

    assert entries[0].target_skill == "memo-table-updates"
    assert entries[0].rule.startswith("Always reapply rPr")
    assert entries[1].target_skill == "layout-integrity"
    assert entries[1].how_to_apply.startswith("When splitting a slide")


def test_parse_skips_entries_missing_required_fields():
    text = """\
## Entry 1
**target_skill:** memo-table-updates

**Rule:** A rule with no Why or How.

## Entry 2
**target_skill:** layout-integrity

**Rule:** Complete rule.

**Why:** Complete why.

**How to apply:** Complete how.
"""
    entries, warnings = parse_entries(text)
    assert len(entries) == 1
    assert entries[0].index == 2
    assert any("entry 1" in w for w in warnings)


def test_parse_no_entries_returns_warning():
    entries, warnings = parse_entries("# Pending Skill Updates\n\n(none this run)\n")
    assert entries == []
    assert warnings == ["no '## Entry N' headings found"]


def test_parse_handles_multiline_field_values():
    text = """\
## Entry 1
**target_skill:** memo-changelog

**Rule:** First sentence of rule.
Continuation of rule on the next line.

**Why:** First sentence.
Continuation.

**How to apply:** Trigger on X.
"""
    entries, _ = parse_entries(text)
    assert len(entries) == 1
    assert "Continuation of rule" in entries[0].rule
    assert "Continuation." in entries[0].why


# ── append_to_skill_md ─────────────────────────────────────────────

def _make_entry(rule: str = "test rule", target: str = VALID_NAME) -> Entry:
    return Entry(
        index=1,
        target_skill=target,
        rule=rule,
        why="why",
        how_to_apply="how",
        raw="",
    )


def test_append_creates_learned_rules_section_when_missing(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\nname: x\ndescription: y\n---\n\nbody\n")

    append_to_skill_md(skill_md, _make_entry("alpha"))
    out = skill_md.read_text()
    assert "## Learned Rules" in out
    assert "**Rule:** alpha" in out
    assert out.count("## Learned Rules") == 1


def test_append_reuses_existing_section(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: x\ndescription: y\n---\n\nbody\n\n## Learned Rules\n\n"
        "- **Rule:** existing\n  **Why:** w\n  **How to apply:** h\n"
    )

    append_to_skill_md(skill_md, _make_entry("alpha"))
    out = skill_md.read_text()
    assert out.count("## Learned Rules") == 1
    assert "**Rule:** existing" in out
    assert "**Rule:** alpha" in out


def test_append_dedupes_identical_rule(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\nname: x\ndescription: y\n---\n\nbody\n")

    append_to_skill_md(skill_md, _make_entry("dup-rule"))
    append_to_skill_md(skill_md, _make_entry("dup-rule"))
    out = skill_md.read_text()
    assert out.count("**Rule:** dup-rule") == 1


# ── apply_decisions ────────────────────────────────────────────────

def _decision(decision: str, target: str = VALID_NAME, rule: str = "r") -> Decision:
    return Decision(
        entry_index=1,
        decision=decision,
        target_skill=target,
        rule=rule,
        why="w",
        how_to_apply="h",
    )


def test_apply_decisions_dry_run_no_mutation_no_api(monkeypatch, tmp_path):
    """Dry-run must not write to disk or call any skills API endpoint."""
    target_dir = tmp_path / VALID_NAME
    target_dir.mkdir()
    skill_md = target_dir / "SKILL.md"
    original_text = "---\nname: x\ndescription: y\n---\n"
    skill_md.write_text(original_text)

    spec_patch = type(
        "S", (), {"name": VALID_NAME, "source_dir": target_dir, "display_title": "X"},
    )
    monkeypatch.setattr(promote_skills, "SKILL_SPECS", (spec_patch,))
    monkeypatch.setattr(promote_skills, "load_cache", lambda: {VALID_NAME: "skill_abc"})

    called = {"create_version": 0, "update_agent": 0}

    def _fail_create(*a, **kw):
        called["create_version"] += 1
        raise AssertionError("create_skill_version called in dry-run")

    def _fail_update(*a, **kw):
        called["update_agent"] += 1
        raise AssertionError("update_agent called in dry-run")

    monkeypatch.setattr(promote_skills, "create_skill_version", _fail_create)
    monkeypatch.setattr(promote_skills, "update_agent", _fail_update)

    versions = apply_decisions([_decision("approved", rule="dry-rule")], dry_run=True)
    assert versions == {VALID_NAME: "(dry-run)"}
    assert skill_md.read_text() == original_text
    assert called == {"create_version": 0, "update_agent": 0}


def test_apply_decisions_live_run_publishes_once_per_skill(monkeypatch, tmp_path):
    """Live run mutates SKILL.md and calls create_skill_version once per
    modified skill, regardless of how many entries went to that skill."""
    target_dir = tmp_path / VALID_NAME
    target_dir.mkdir()
    (target_dir / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n\nbody\n")

    spec_patch = type(
        "S", (), {"name": VALID_NAME, "source_dir": target_dir, "display_title": "X"},
    )
    monkeypatch.setattr(promote_skills, "SKILL_SPECS", (spec_patch,))
    monkeypatch.setattr(promote_skills, "load_cache", lambda: {VALID_NAME: "skill_abc"})
    monkeypatch.setattr(promote_skills, "AGENT_ID", "agent_test")

    create_calls: list[tuple] = []
    monkeypatch.setattr(
        promote_skills, "create_skill_version",
        lambda skill_id, *, skill_dir: create_calls.append((skill_id, skill_dir))
        or {"version": "9999"},
    )

    update_calls: list[tuple] = []
    monkeypatch.setattr(
        promote_skills, "update_agent",
        lambda agent_id, **kw: update_calls.append((agent_id, kw)) or {},
    )
    monkeypatch.setattr(
        promote_skills, "build_skills_list",
        lambda: [{"type": "anthropic", "skill_id": "xlsx"}],
    )

    versions = apply_decisions([
        _decision("approved", rule="r1"),
        _decision("edited", rule="r2"),
        _decision("rejected", rule="r-no"),
        _decision("reassigned", rule="r3"),
    ])

    assert versions == {VALID_NAME: "9999"}
    assert len(create_calls) == 1  # one version bump despite 3 promoted entries
    assert create_calls[0][0] == "skill_abc"
    assert len(update_calls) == 1
    assert update_calls[0][0] == "agent_test"

    body = (target_dir / "SKILL.md").read_text()
    assert "**Rule:** r1" in body
    assert "**Rule:** r2" in body
    assert "**Rule:** r3" in body
    assert "**Rule:** r-no" not in body


def test_apply_decisions_skips_skill_with_no_cached_id(monkeypatch, tmp_path):
    """Without a cached skill_id, we still mutate SKILL.md but skip the
    publish — the user is expected to run sync_skills.py to bootstrap."""
    target_dir = tmp_path / VALID_NAME
    target_dir.mkdir()
    (target_dir / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n")

    spec_patch = type(
        "S", (), {"name": VALID_NAME, "source_dir": target_dir, "display_title": "X"},
    )
    monkeypatch.setattr(promote_skills, "SKILL_SPECS", (spec_patch,))
    monkeypatch.setattr(promote_skills, "load_cache", lambda: {})  # empty cache

    def _fail(*_a, **_kw):
        pytest.fail("API called for a skill with no cached ID")

    monkeypatch.setattr(promote_skills, "create_skill_version", _fail)
    monkeypatch.setattr(promote_skills, "update_agent", _fail)

    versions = apply_decisions([_decision("approved", rule="orphan")])
    assert versions == {}
    assert "**Rule:** orphan" in (target_dir / "SKILL.md").read_text()


# ── Audit log + reprocess gating ───────────────────────────────────

def test_audit_log_round_trip(tmp_path):
    src = tmp_path / "pending_skill_updates.md"
    src.write_text("# Pending\n")

    decisions = [_decision("approved"), _decision("rejected", rule="x")]
    audit = write_audit_log(src, decisions, {VALID_NAME: "v1"}, dry_run=False)
    assert audit == src.with_suffix(".promoted.json")

    payload = json.loads(audit.read_text())
    assert payload["source_file"] == str(src)
    assert payload["dry_run"] is False
    assert payload["new_versions"] == {VALID_NAME: "v1"}
    assert {d["decision"] for d in payload["decisions"]} == {"approved", "rejected"}


def test_already_processed_detects_existing_audit(tmp_path):
    src = tmp_path / "pending_skill_updates.md"
    src.write_text("# x\n")
    assert not promote_skills.already_processed(src)
    src.with_suffix(".promoted.json").write_text("{}")
    assert promote_skills.already_processed(src)
