#!/usr/bin/env python3
"""
Promote agent-proposed skill updates from a session's pending file
into published skill versions.

The Memo Chef agent writes `pending_skill_updates.md` to its session
output whenever it learns something generalizable that belongs in a
custom skill's body. This CLI walks every entry in that file, asks
the human reviewer to approve / reject / edit / reassign, and on
approval:
  1. Appends the entry to the target SKILL.md under "## Learned Rules".
  2. Publishes a new version of that skill via the skills API.
  3. Re-references the agent so it picks the new version up.

An audit log (pending_skill_updates.promoted.json) is written next to
the source file. Re-running against an already-processed file is a
no-op unless `--reprocess` is passed.

Usage:
    python -m managed_agents.promote_skills <path-to-pending_skill_updates.md>
    python -m managed_agents.promote_skills <path> --dry-run
    python -m managed_agents.promote_skills <path> --reprocess
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from managed_agents.api_client import create_skill_version, update_agent
from managed_agents.config import AGENT_ID
from managed_agents.skill_manifest import SKILL_SPECS, load_cache
from managed_agents.skills import build_skills_list

VALID_SKILL_NAMES: tuple[str, ...] = tuple(s.name for s in SKILL_SPECS)
LEARNED_HEADING = "## Learned Rules"
DECISIONS = ("approved", "rejected", "edited", "reassigned", "deferred")


@dataclass
class Entry:
    """One proposed skill addition parsed from the markdown file."""

    index: int
    target_skill: str
    rule: str
    why: str
    how_to_apply: str
    raw: str  # original markdown body, for audit


@dataclass
class Decision:
    entry_index: int
    decision: str
    target_skill: str
    rule: str
    why: str
    how_to_apply: str
    edited_from: dict | None = None
    promoted_skill_id: str | None = None
    new_version: str | None = None
    notes: str = ""


# ── Parsing ────────────────────────────────────────────────────────

def parse_entries(text: str) -> tuple[list[Entry], list[str]]:
    """Split the pending file into Entry objects.

    Returns (entries, warnings). Warnings list any entry that is missing
    a required field — those entries are skipped, not silently dropped.
    """
    entries: list[Entry] = []
    warnings: list[str] = []

    # Accept any heading depth (## or ###) — the agent occasionally drops
    # to h3, and we'd rather rescue the entry than silently lose it.
    entry_heading = re.compile(r"^#+\s*Entry\s+(\d+)\b", re.IGNORECASE)

    blocks: list[tuple[int, list[str]]] = []
    current: list[str] | None = None
    current_idx = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        match = entry_heading.match(stripped)
        if match:
            if current is not None:
                blocks.append((current_idx, current))
            current = []
            current_idx = int(match.group(1))
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append((current_idx, current))

    if not blocks:
        warnings.append("no 'Entry N' headings found")
        return entries, warnings

    for idx, body_lines in blocks:
        raw = "\n".join(body_lines).strip()
        fields = _extract_bold_fields(body_lines)
        target = fields.get("target_skill", "").strip().strip("`").strip()
        rule = fields.get("rule", "").strip()
        why = fields.get("why", "").strip()
        how = fields.get("how_to_apply", "").strip()

        missing = [
            label for label, val in (
                ("target_skill", target),
                ("Rule", rule),
                ("Why", why),
                ("How to apply", how),
            )
            if not val
        ]
        if missing:
            warnings.append(f"entry {idx}: missing fields {missing}; skipping")
            continue

        entries.append(Entry(
            index=idx,
            target_skill=target,
            rule=rule,
            why=why,
            how_to_apply=how,
            raw=raw,
        ))

    return entries, warnings


def _normalize_field_key(raw: str) -> str:
    """Canonicalize a field key.

    Whitespace collapses to a single underscore so `Target Skill`,
    `target skill`, and `target_skill` all map to `target_skill`. Case
    is dropped. Trailing colons stripped.
    """
    cleaned = raw.strip().rstrip(":").strip()
    return re.sub(r"[\s_]+", "_", cleaned).lower()


def _is_field_key_line(stripped: str) -> tuple[str, str] | None:
    """Return (key, value) if `stripped` looks like a `**Key:** value`
    line, else None.

    Requires an explicit colon either inside the bold span (`**Key:**`)
    or immediately after the closing `**` (`**Key**:`). Without a colon,
    a leading `**bold**` token in a continuation line would falsely look
    like a new key — that's the bug this guards against.
    """
    if not stripped.startswith("**"):
        return None
    close = stripped.find("**", 2)
    if close < 0:
        return None
    key_block = stripped[2:close]
    after = stripped[close + 2:]

    if key_block.rstrip().endswith(":"):
        value_start = after.lstrip()
    elif after.lstrip().startswith(":"):
        value_start = after.lstrip().lstrip(":").lstrip()
    else:
        # No colon → not a key declaration, just inline bold text.
        return None

    return _normalize_field_key(key_block), value_start.strip()


def _extract_bold_fields(lines: list[str]) -> dict[str, str]:
    """Parse `**Key:** value` lines into {normalized_key: value}.

    Tolerates multi-line values (continuation lines append to the most
    recent key until a blank line or the next key line). Key names are
    normalized so `Target Skill`, `target skill`, and `target_skill`
    all resolve to the same field.
    """
    out: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in lines:
        stripped = line.strip()
        kv = _is_field_key_line(stripped)
        if kv is not None:
            key, value = kv
            current_key = key
            out.setdefault(key, [])
            if value:
                out[key].append(value)
            continue
        if not stripped:
            current_key = None
            continue
        if current_key is not None:
            out[current_key].append(stripped)
    return {k: " ".join(v).strip() for k, v in out.items()}


# ── SKILL.md mutation ──────────────────────────────────────────────

def _format_entry_for_skill(entry: Entry) -> str:
    return (
        f"- **Rule:** {entry.rule}\n"
        f"  **Why:** {entry.why}\n"
        f"  **How to apply:** {entry.how_to_apply}\n"
    )


def append_to_skill_md(skill_md: Path, entry: Entry) -> None:
    """Append a learned rule under '## Learned Rules', creating the
    section if absent. Idempotent across re-runs (won't add duplicates).

    Dedupe uses the formatted rule line `- **Rule:** {text}\n`, not free-
    text containment. That keeps a *superset* rule from being silently
    skipped because the existing rule is a substring of it, and lets an
    edited rule land cleanly even when the original is already present.
    """
    text = skill_md.read_text(encoding="utf-8")
    block = _format_entry_for_skill(entry)

    rule_line = f"- **Rule:** {entry.rule}\n"
    if rule_line in text:
        # Exact same rule line already present — skip silent duplicates
        # so re-runs don't bloat the file.
        return

    if LEARNED_HEADING in text:
        # Append below the existing section. Keep one trailing newline
        # at end-of-file.
        if not text.endswith("\n"):
            text += "\n"
        text += block
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += f"\n{LEARNED_HEADING}\n\n{block}"

    skill_md.write_text(text, encoding="utf-8")


# ── Interactive review ─────────────────────────────────────────────

def _print_entry(entry: Entry, total: int) -> None:
    print()
    print("─" * 70)
    print(f"Entry {entry.index} of {total} — target: {entry.target_skill}")
    print("─" * 70)
    print(f"  Rule:           {entry.rule}")
    print(f"  Why:            {entry.why}")
    print(f"  How to apply:   {entry.how_to_apply}")
    print()


def _prompt_choice() -> str:
    while True:
        choice = input(
            "[a]pprove  [r]eject  [e]dit  [s] reassign target  "
            "[d]efer  [q]uit > "
        ).strip().lower()
        if choice in {"a", "r", "e", "s", "d", "q"}:
            return choice
        print("  please enter one of: a, r, e, s, d, q")


def _prompt_nonempty(label: str, default: str) -> str:
    val = input(f"  {label} [{default}]: ").strip()
    return val or default


def _prompt_target_skill(default: str) -> str:
    print("  Available skills:")
    for name in VALID_SKILL_NAMES:
        marker = " (current)" if name == default else ""
        print(f"    - {name}{marker}")
    while True:
        val = input(f"  target_skill [{default}]: ").strip() or default
        if val in VALID_SKILL_NAMES:
            return val
        print(f"  '{val}' is not in the manifest. Pick one from the list above.")


def review_entries(
    entries: list[Entry],
    *,
    interactive: bool = True,
) -> list[Decision]:
    """Walk each entry; return a Decision per entry."""
    decisions: list[Decision] = []
    total = len(entries)
    for pos, entry in enumerate(entries):
        _print_entry(entry, total)

        if entry.target_skill not in VALID_SKILL_NAMES:
            print(
                f"  WARNING: target_skill {entry.target_skill!r} is not in "
                f"the manifest; you must reassign or reject this entry."
            )

        if not interactive:
            decisions.append(Decision(
                entry_index=entry.index,
                decision="deferred",
                target_skill=entry.target_skill,
                rule=entry.rule,
                why=entry.why,
                how_to_apply=entry.how_to_apply,
                notes="dry-run: no decision taken",
            ))
            continue

        choice = _prompt_choice()
        if choice == "q":
            print("  quitting; remaining entries left as 'deferred'")
            # Defer the current entry plus every entry after it. Use the
            # loop position rather than entries.index(entry) — list.index
            # finds the first equal entry, and dataclasses use value-based
            # equality, so two identical proposals would alias.
            for e in entries[pos:]:
                decisions.append(Decision(
                    entry_index=e.index,
                    decision="deferred",
                    target_skill=e.target_skill,
                    rule=e.rule,
                    why=e.why,
                    how_to_apply=e.how_to_apply,
                ))
            return decisions

        if choice == "r":
            decisions.append(Decision(
                entry_index=entry.index,
                decision="rejected",
                target_skill=entry.target_skill,
                rule=entry.rule,
                why=entry.why,
                how_to_apply=entry.how_to_apply,
            ))
            continue

        if choice == "d":
            decisions.append(Decision(
                entry_index=entry.index,
                decision="deferred",
                target_skill=entry.target_skill,
                rule=entry.rule,
                why=entry.why,
                how_to_apply=entry.how_to_apply,
            ))
            continue

        target = entry.target_skill
        rule = entry.rule
        why = entry.why
        how = entry.how_to_apply
        edited_from: dict | None = None

        if choice == "s":
            target = _prompt_target_skill(target)
            edited_from = {"target_skill": entry.target_skill}

        if choice == "e":
            rule = _prompt_nonempty("Rule", entry.rule)
            why = _prompt_nonempty("Why", entry.why)
            how = _prompt_nonempty("How to apply", entry.how_to_apply)
            target = _prompt_target_skill(target)
            if (rule, why, how, target) != (
                entry.rule, entry.why, entry.how_to_apply, entry.target_skill,
            ):
                edited_from = {
                    "rule": entry.rule,
                    "why": entry.why,
                    "how_to_apply": entry.how_to_apply,
                    "target_skill": entry.target_skill,
                }

        if target not in VALID_SKILL_NAMES:
            print(f"  ERROR: {target!r} is not a valid skill; defaulting to reject")
            decisions.append(Decision(
                entry_index=entry.index,
                decision="rejected",
                target_skill=entry.target_skill,
                rule=entry.rule,
                why=entry.why,
                how_to_apply=entry.how_to_apply,
                notes=f"invalid target after edit: {target!r}",
            ))
            continue

        decision_label = "approved"
        if choice == "s":
            decision_label = "reassigned"
        elif choice == "e":
            decision_label = "edited"

        decisions.append(Decision(
            entry_index=entry.index,
            decision=decision_label,
            target_skill=target,
            rule=rule,
            why=why,
            how_to_apply=how,
            edited_from=edited_from,
        ))

    return decisions


# ── Publishing ─────────────────────────────────────────────────────

def _is_promotion(decision: Decision) -> bool:
    return decision.decision in {"approved", "edited", "reassigned"}


def apply_decisions(
    decisions: list[Decision],
    *,
    dry_run: bool = False,
) -> dict[str, str]:
    """Append approved entries to their target SKILL.md, then publish a
    new version per modified skill. Returns {skill_name: new_version}.
    """
    cache = load_cache()
    by_skill: dict[str, list[Decision]] = {}
    for d in decisions:
        if _is_promotion(d):
            by_skill.setdefault(d.target_skill, []).append(d)

    new_versions: dict[str, str] = {}
    for skill_name, items in by_skill.items():
        spec = next((s for s in SKILL_SPECS if s.name == skill_name), None)
        if spec is None:
            print(f"  ERROR: no manifest entry for {skill_name!r}; skipping")
            continue
        skill_md = spec.source_dir / "SKILL.md"
        if not skill_md.is_file():
            print(f"  ERROR: SKILL.md missing under {spec.source_dir}; skipping")
            continue

        for d in items:
            entry = Entry(
                index=d.entry_index,
                target_skill=d.target_skill,
                rule=d.rule,
                why=d.why,
                how_to_apply=d.how_to_apply,
                raw="",
            )
            if dry_run:
                print(f"  [dry-run] would append entry {d.entry_index} to "
                      f"{skill_md.relative_to(spec.source_dir.parent.parent)}")
            else:
                append_to_skill_md(skill_md, entry)

        skill_id = cache.get(skill_name)
        if not skill_id:
            print(
                f"  WARNING: {skill_name} has no cached skill_id — run "
                f"sync_skills.py first to create it. Local SKILL.md was "
                f"updated but no new version was published.",
                file=sys.stderr,
            )
            continue

        if dry_run:
            print(f"  [dry-run] would publish new version of {skill_name} ({skill_id})")
            new_versions[skill_name] = "(dry-run)"
            continue

        result = create_skill_version(skill_id, skill_dir=spec.source_dir)
        version = result.get("version", "?")
        new_versions[skill_name] = version
        for d in items:
            d.promoted_skill_id = skill_id
            d.new_version = version
        print(f"  [published] {skill_name}: version {version}")

    if new_versions and not dry_run and AGENT_ID:
        skills = build_skills_list()
        update_agent(AGENT_ID, skills=skills)
        print(f"  [agent updated] {AGENT_ID}: now references the new versions at 'latest'")

    return new_versions


# ── Audit log ──────────────────────────────────────────────────────

def write_audit_log(
    source_file: Path,
    decisions: list[Decision],
    new_versions: dict[str, str],
    *,
    dry_run: bool,
    error: BaseException | None = None,
) -> Path:
    audit_path = source_file.with_suffix(".promoted.json")
    payload: dict = {
        "source_file": str(source_file),
        "processed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "dry_run": dry_run,
        "decisions": [asdict(d) for d in decisions],
        "new_versions": new_versions,
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    audit_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return audit_path


def already_processed(source_file: Path) -> bool:
    return source_file.with_suffix(".promoted.json").exists()


# ── CLI entry point ────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path,
        help="Path to pending_skill_updates.md downloaded from a session.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and show decisions without modifying SKILL.md or hitting the API.",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Process even if a .promoted.json audit log already exists.",
    )
    args = parser.parse_args(argv)

    source: Path = args.source
    if not source.is_file():
        print(f"ERROR: {source} not found", file=sys.stderr)
        return 1

    if already_processed(source) and not args.reprocess:
        print(
            f"{source.with_suffix('.promoted.json').name} already exists. "
            f"Use --reprocess to walk the file again.",
            file=sys.stderr,
        )
        return 1

    text = source.read_text(encoding="utf-8")
    entries, warnings = parse_entries(text)
    for w in warnings:
        print(f"  [parse warning] {w}", file=sys.stderr)

    if not entries:
        print("No valid entries found.")
        return 0

    print(f"Loaded {len(entries)} pending entr"
          f"{'y' if len(entries) == 1 else 'ies'} from {source.name}")

    decisions = review_entries(entries, interactive=not args.dry_run)

    summary: dict[str, int] = {d: 0 for d in DECISIONS}
    for d in decisions:
        summary[d.decision] = summary.get(d.decision, 0) + 1
    print()
    print("Summary:")
    for k in DECISIONS:
        if summary.get(k):
            print(f"  {k}: {summary[k]}")

    # Always write the audit log, even if apply_decisions raises partway
    # through publishing. Otherwise a network error mid-loop would leave
    # some skills bumped server-side with no record of which ones.
    new_versions: dict[str, str] = {}
    apply_error: BaseException | None = None
    try:
        new_versions = apply_decisions(decisions, dry_run=args.dry_run)
    except BaseException as exc:
        apply_error = exc
        # Surface published-version state captured on Decision objects
        # before the failure, so the audit reflects partial progress.
        for d in decisions:
            if d.new_version and d.target_skill not in new_versions:
                new_versions[d.target_skill] = d.new_version

    audit = write_audit_log(
        source, decisions, new_versions,
        dry_run=args.dry_run, error=apply_error,
    )
    print(f"Audit log: {audit}")
    if apply_error is not None:
        print(
            f"ERROR: apply_decisions failed partway through: {apply_error!r}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
