#!/usr/bin/env python3
"""
Create (or update) the Memo Chef managed agent.

Usage:
    python -m managed_agents.setup_agent
    python -m managed_agents.setup_agent --allow-missing-skills

By default, refuses to create the agent if any custom skill in the
manifest has no cached skill_id (i.e. `sync_skills.py` hasn't been run
yet). This prevents silently provisioning an agent whose system prompt
references seven skills it isn't actually bound to. Pass
`--allow-missing-skills` to override (e.g. for a deliberate xlsx/pptx-only
deploy).

Saves the agent ID to managed_agents/.env for reuse across sessions.
"""

from __future__ import annotations

import argparse
import sys

from managed_agents.api_client import create_agent
from managed_agents.config import AGENT_ID, save_ids
from managed_agents.skill_manifest import SKILL_SPECS, load_cache
from managed_agents.skills import build_skills_list
from managed_agents.system_prompt import SYSTEM_PROMPT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-skills",
        action="store_true",
        help="Provision the agent even if some custom skills are not yet uploaded.",
    )
    args = parser.parse_args(argv)

    if AGENT_ID:
        print(f"Agent already provisioned: {AGENT_ID}")
        print("Delete MANAGED_AGENT_ID from managed_agents/.env to re-create.")
        return 0

    cache = load_cache()
    missing = [s.name for s in SKILL_SPECS if s.name not in cache]
    if missing:
        bullets = "\n  - ".join(missing)
        if not args.allow_missing_skills:
            print(
                "ERROR: the following custom skills have no cached skill_id "
                "yet:\n  - "
                + bullets
                + "\n\nRun `python -m managed_agents.sync_skills` first to "
                "upload them, or re-run this command with "
                "--allow-missing-skills to proceed anyway.",
                file=sys.stderr,
            )
            return 2
        print(
            "WARNING: proceeding without these skills attached "
            "(--allow-missing-skills set):\n  - " + bullets
        )

    skills = build_skills_list()

    agent = create_agent(
        name="Memo Chef",
        description="Autonomous IC memo updater for Subtext student-housing deals.",
        model="claude-sonnet-4-6",
        system=SYSTEM_PROMPT,
        tools=[{"type": "agent_toolset_20260401"}],
        skills=skills,
    )

    agent_id = agent["id"]
    version = agent.get("version", "?")
    print(f"Agent created: {agent_id} (version {version})")

    save_ids(agent_id=agent_id)
    print("Saved agent ID to managed_agents/.env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
