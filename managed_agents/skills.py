"""
Skills attached to the Memo Chef agent.

This module composes two sources:

1. **Pre-built Anthropic skills** — maintained by Anthropic; we attach
   xlsx and pptx so the agent can read/write Office files using the
   org's gold-standard helpers.

2. **Custom Memo Chef skills** — authored under
   `managed_agents/skills_content/`, uploaded to the org via
   `sync_skills.py`, and referenced here by the skill_id stored in
   `managed_agents/.skills.json`.

`build_skills_list()` returns the merged list to pass into
`create_agent`/`update_agent`. The legacy `SKILLS` constant is kept so
existing callers (and the test suite) continue to work; it is computed
from the same source of truth.

Docs: https://platform.claude.com/docs/en/managed-agents/skills
"""

from __future__ import annotations

from managed_agents.skill_manifest import SKILL_SPECS, load_cache

# Pre-built Anthropic skills. Keep this list short — every skill costs
# ~100 tokens of metadata per session even when unused.
_ANTHROPIC_SKILLS: list[dict] = [
    {"type": "anthropic", "skill_id": "xlsx"},
    {"type": "anthropic", "skill_id": "pptx"},
]


def _custom_skill_entries() -> list[dict]:
    """Return Managed Agents API entries for every cached custom skill.

    A skill in the manifest with no cached skill_id is silently skipped
    — that just means `sync_skills.py` hasn't been run yet for it.
    """
    cache = load_cache()
    entries: list[dict] = []
    for spec in SKILL_SPECS:
        skill_id = cache.get(spec.name)
        if not skill_id:
            continue
        entries.append({
            "type": "custom",
            "skill_id": skill_id,
            "version": "latest",
        })
    return entries


def build_skills_list() -> list[dict]:
    """Merged Anthropic + custom skills, suitable for create_agent/update_agent."""
    return list(_ANTHROPIC_SKILLS) + _custom_skill_entries()


# Backwards-compatible constant. Recomputed at import time from the
# manifest + cache, so callers that read `SKILLS` (e.g. setup_agent.py,
# update_skills.py, the test suite) get the merged list.
SKILLS: list[dict] = build_skills_list()
