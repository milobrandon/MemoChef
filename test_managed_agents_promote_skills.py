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
    assert warnings == ["no 'Entry N' headings found"]


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


def test_parse_normalizes_capitalized_and_spaced_keys():
    """The agent will sometimes write `**Target Skill:**` or
    `**How to apply:**` — those must resolve to the canonical
    underscore form so the entry isn't dropped as missing-fields."""
    text = """\
## Entry 1
**Target Skill:** memo-table-updates

**Rule:** capitalized key still parses.

**Why:** the model emitted Title Case headings by accident.

**How To Apply:** when keys come back capitalized.
"""
    entries, warnings = parse_entries(text)
    assert warnings == []
    assert len(entries) == 1
    assert entries[0].target_skill == "memo-table-updates"
    assert entries[0].rule.startswith("capitalized key")
    assert entries[0].how_to_apply.startswith("when keys come back")


def test_parse_handles_colon_outside_bold():
    text = """\
## Entry 1
**target_skill**: memo-table-updates

**rule**: colon outside the bold span.

**why**: parser must accept either form.

**how_to_apply**: trigger when X.
"""
    entries, warnings = parse_entries(text)
    assert warnings == []
    assert len(entries) == 1
    assert entries[0].rule == "colon outside the bold span."


def test_parse_continuation_line_with_inline_bold_is_not_a_new_key():
    """A bold token in a continuation line (no colon) must not be
    misclassified as a new key — that would split the value across
    two fields and corrupt the entry."""
    text = """\
## Entry 1
**target_skill:** memo-table-updates

**rule:** the **bold** word matters, and the next sentence continues here.

**why:** because parsing must preserve inline emphasis.

**how_to_apply:** never.
"""
    entries, warnings = parse_entries(text)
    assert warnings == []
    assert len(entries) == 1
    assert "bold" in entries[0].rule
    assert "matters" in entries[0].rule
    assert "next sentence continues" in entries[0].rule


def test_parse_accepts_h3_entry_heading():
    """If the agent drops to ### Entry by mistake, recover the entry
    rather than silently lose it."""
    text = """\
### Entry 1
**target_skill:** memo-table-updates

**rule:** h3 heading still gets parsed.

**why:** the agent occasionally uses the wrong heading depth.

**how_to_apply:** never.
"""
    entries, warnings = parse_entries(text)
    assert warnings == []
    assert len(entries) == 1
    assert entries[0].rule.startswith("h3 heading")


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


def test_append_does_not_skip_superset_rule(tmp_path):
    """A new rule whose text is a superset of an existing rule must
    NOT be silently skipped as a duplicate."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\nname: x\ndescription: y\n---\n")

    append_to_skill_md(skill_md, _make_entry("always check colors"))
    append_to_skill_md(skill_md, _make_entry("always check colors before saving"))

    out = skill_md.read_text()
    assert "- **Rule:** always check colors\n" in out
    assert "- **Rule:** always check colors before saving\n" in out


def test_append_lets_edited_rule_land_alongside_original(tmp_path):
    """An edited rule won't string-match the original, so it must
    append even when the un-edited form is already present."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\nname: x\ndescription: y\n---\n")

    original = _make_entry("be careful with subtotals")
    append_to_skill_md(skill_md, original)
    edited = _make_entry("be careful with subtotal rows on dark-theme decks")
    append_to_skill_md(skill_md, edited)

    out = skill_md.read_text()
    assert out.count("- **Rule:**") == 2


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


# ── Partial-failure / atomicity ────────────────────────────────────

def test_main_writes_audit_log_even_when_publish_fails(monkeypatch, tmp_path):
    """If create_skill_version raises mid-loop, the audit log must
    still record what was published so the operator can recover."""
    src = tmp_path / "pending.md"
    src.write_text("""\
## Entry 1
**target_skill:** memo-table-updates
**rule:** alpha
**why:** w
**how_to_apply:** h

## Entry 2
**target_skill:** layout-integrity
**rule:** bravo
**why:** w
**how_to_apply:** h
""")

    target_a = tmp_path / "memo-table-updates"
    target_b = tmp_path / "layout-integrity"
    for d in (target_a, target_b):
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n")

    spec_a = type("S", (), {
        "name": "memo-table-updates",
        "source_dir": target_a,
        "display_title": "A",
    })
    spec_b = type("S", (), {
        "name": "layout-integrity",
        "source_dir": target_b,
        "display_title": "B",
    })
    monkeypatch.setattr(promote_skills, "SKILL_SPECS", (spec_a, spec_b))
    monkeypatch.setattr(
        promote_skills, "load_cache",
        lambda: {"memo-table-updates": "skill_a", "layout-integrity": "skill_b"},
    )
    monkeypatch.setattr(promote_skills, "AGENT_ID", "agent_test")
    monkeypatch.setattr(promote_skills, "build_skills_list", lambda: [])

    # First skill publishes fine; second raises.
    calls: list[str] = []

    def _create_version(skill_id, *, skill_dir):
        calls.append(skill_id)
        if skill_id == "skill_a":
            return {"version": "v100"}
        raise RuntimeError("network blew up on second skill")

    monkeypatch.setattr(promote_skills, "create_skill_version", _create_version)
    monkeypatch.setattr(promote_skills, "update_agent", lambda *a, **kw: None)

    # Drive promote_skills.main without interactive input by stubbing
    # review_entries to auto-approve everything.
    def _auto_approve(entries, *, interactive=True):
        return [
            promote_skills.Decision(
                entry_index=e.index,
                decision="approved",
                target_skill=e.target_skill,
                rule=e.rule,
                why=e.why,
                how_to_apply=e.how_to_apply,
            )
            for e in entries
        ]
    monkeypatch.setattr(promote_skills, "review_entries", _auto_approve)

    rc = promote_skills.main([str(src)])
    assert rc == 3, "should signal partial failure with non-zero exit"

    audit = src.with_suffix(".promoted.json")
    assert audit.is_file(), "audit log MUST be written even on partial failure"
    payload = json.loads(audit.read_text())
    assert payload["error"]["type"] == "RuntimeError"
    # First skill made it to a published version; second did not.
    assert payload["new_versions"].get("memo-table-updates") == "v100"
    assert "layout-integrity" not in payload["new_versions"]
    assert calls == ["skill_a", "skill_b"]


# ── Quit deferral with identical entries ───────────────────────────

def test_review_entries_quit_defers_correctly_with_duplicate_entries(monkeypatch):
    """Two fully value-identical entries must NOT alias when one is
    quit on. Earlier behavior used entries.index(entry), which returns
    the index of the first equal element — quitting on the second
    would have deferred starting at position 0, double-counting the
    first.

    To trigger the bug realistically, both entries share the same
    `index` field (which would happen if the agent emitted
    `## Entry 1` twice in the same file).
    """
    e1 = Entry(index=1, target_skill=VALID_NAME, rule="dup", why="w", how_to_apply="h", raw="")
    e2 = Entry(index=1, target_skill=VALID_NAME, rule="dup", why="w", how_to_apply="h", raw="")
    e3 = Entry(index=2, target_skill=VALID_NAME, rule="other", why="w", how_to_apply="h", raw="")
    assert e1 == e2  # dataclass equality is by value; this is the trap

    # Approve first, quit on second; third must still be deferred (not
    # the first, despite being value-equal to the second).
    answers = iter(["a", "q"])
    monkeypatch.setattr(promote_skills, "_prompt_choice", lambda: next(answers))

    decisions = promote_skills.review_entries([e1, e2, e3])
    assert len(decisions) == 3
    assert decisions[0].decision == "approved"
    assert decisions[1].decision == "deferred"
    assert decisions[2].decision == "deferred"
