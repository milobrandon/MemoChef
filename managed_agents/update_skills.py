#!/usr/bin/env python3
"""
Push the current SKILLS list to the already-provisioned managed agent.

setup_agent.py only attaches skills when create_agent runs once. After
editing skills.py, the remote agent keeps its old skill set until
someone POSTs the new list. Run this script to do that.

Usage:
    python -m managed_agents.update_skills
"""

from __future__ import annotations

import sys

from managed_agents.api_client import update_agent
from managed_agents.config import AGENT_ID
from managed_agents.skills import build_skills_list


def main() -> int:
    if not AGENT_ID:
        print(
            "No MANAGED_AGENT_ID configured. Run `python -m managed_agents.setup_agent` "
            "first, or set MANAGED_AGENT_ID in managed_agents/.env.",
            file=sys.stderr,
        )
        return 1

    skills = build_skills_list()
    print(f"Updating agent {AGENT_ID}")
    print(f"Skills: {[s['skill_id'] for s in skills]}")

    result = update_agent(AGENT_ID, skills=skills)

    version = result.get("version", "?")
    print(f"Agent updated. version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
