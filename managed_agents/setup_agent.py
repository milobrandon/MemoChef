#!/usr/bin/env python3
"""
Create (or update) the Memo Chef managed agent.

Usage:
    python -m managed_agents.setup_agent

Saves the agent ID to managed_agents/.env for reuse across sessions.
"""

from __future__ import annotations

from managed_agents.api_client import create_agent
from managed_agents.config import AGENT_ID, save_ids
from managed_agents.skill_manifest import SKILL_SPECS, load_cache
from managed_agents.skills import build_skills_list
from managed_agents.system_prompt import SYSTEM_PROMPT


def main() -> None:
    if AGENT_ID:
        print(f"Agent already provisioned: {AGENT_ID}")
        print("Delete MANAGED_AGENT_ID from managed_agents/.env to re-create.")
        return

    cache = load_cache()
    missing = [s.name for s in SKILL_SPECS if s.name not in cache]
    if missing:
        print(
            "WARNING: the following custom skills have no cached skill_id "
            "yet — run `python -m managed_agents.sync_skills` first to "
            "upload them, otherwise they will not be attached:\n  - "
            + "\n  - ".join(missing)
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


if __name__ == "__main__":
    main()
