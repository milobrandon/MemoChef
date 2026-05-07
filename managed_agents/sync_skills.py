#!/usr/bin/env python3
"""
Push the local Memo Chef custom skills to Anthropic.

For each skill in `skill_manifest.SKILL_SPECS`:
  - If no skill_id is cached for it, POST /v1/skills (create) and cache
    the assigned skill_id.
  - Otherwise POST /v1/skills/{id}/versions to publish a new version.

After syncing all skills, the agent's skills field is rewritten to
reference both the Anthropic pre-built skills (xlsx, pptx) and every
custom skill at `version: "latest"`.

Usage:
    python -m managed_agents.sync_skills
"""

from __future__ import annotations

import sys

from managed_agents.api_client import (
    create_skill,
    create_skill_version,
    list_skills,
    update_agent,
)
from managed_agents.config import AGENT_ID
from managed_agents.skill_manifest import SKILL_SPECS, load_cache, save_cache
from managed_agents.skills import build_skills_list


def _reconcile_cache_with_org(cache: dict[str, str]) -> dict[str, str]:
    """Match cached skill names against skills already in the org by
    display_title.

    A previous sync_skills run could have created a skill server-side and
    crashed before persisting the ID locally — without this reconcile pass
    the next run would create a duplicate. We hydrate cache entries from
    the org listing wherever the display_title matches a manifest entry.
    """
    title_to_name = {spec.display_title: spec.name for spec in SKILL_SPECS}
    try:
        org_skills = list_skills()
    except Exception as exc:
        print(f"  [warn] could not list skills for reconcile: {exc}", file=sys.stderr)
        return cache

    rehydrated = 0
    for entry in org_skills:
        title = entry.get("display_title")
        skill_id = entry.get("id")
        if not title or not skill_id:
            continue
        name = title_to_name.get(title)
        if not name or cache.get(name) == skill_id:
            continue
        if cache.get(name) and cache[name] != skill_id:
            # Local cache claims a different ID than what the org reports.
            # Don't auto-overwrite — surface the conflict and let a human
            # decide which one to keep.
            print(
                f"  [conflict] {name}: cache={cache[name]} but org has "
                f"{skill_id} for display_title {title!r}. Keeping cached value.",
                file=sys.stderr,
            )
            continue
        cache[name] = skill_id
        rehydrated += 1

    if rehydrated:
        save_cache(cache)
        print(f"  [reconciled] hydrated {rehydrated} skill ID(s) from the org")
    return cache


def _sync_one(name: str, display_title: str, source_dir, cache: dict[str, str]) -> str:
    """Create or version-bump a single skill. Returns the skill_id."""
    existing_id = cache.get(name)
    if existing_id:
        result = create_skill_version(existing_id, skill_dir=source_dir)
        version = result.get("version", "?")
        print(f"  [bumped]  {name}: {existing_id} → version {version}")
        return existing_id

    result = create_skill(display_title=display_title, skill_dir=source_dir)
    skill_id = result["id"]
    version = result.get("latest_version", "?")
    cache[name] = skill_id
    save_cache(cache)
    print(f"  [created] {name}: {skill_id} (version {version})")
    return skill_id


def main() -> int:
    cache = load_cache()
    print(f"Syncing {len(SKILL_SPECS)} custom skills...")
    cache = _reconcile_cache_with_org(cache)

    for spec in SKILL_SPECS:
        if not spec.source_dir.is_dir():
            print(f"  [skip]    {spec.name}: source dir missing", file=sys.stderr)
            continue
        _sync_one(spec.name, spec.display_title, spec.source_dir, cache)

    if not AGENT_ID:
        print(
            "\nNo MANAGED_AGENT_ID configured — skills uploaded but not "
            "attached to any agent. Run `python -m managed_agents.setup_agent` "
            "to provision one.",
        )
        return 0

    print(f"\nUpdating agent {AGENT_ID} skills list...")
    skills = build_skills_list()
    print(f"Skills: {[s.get('skill_id') for s in skills]}")
    result = update_agent(AGENT_ID, skills=skills)
    version = result.get("version", "?")
    print(f"Agent updated. version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
