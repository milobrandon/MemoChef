#!/usr/bin/env python3
"""
Push the current SYSTEM_PROMPT to the already-provisioned managed agent.

setup_agent.py only calls create_agent once and stores MANAGED_AGENT_ID.
After editing system_prompt.py, the remote agent keeps its old prompt
until someone PUTs the new one. Run this script to do that.

Usage:
    python -m managed_agents.update_prompt
"""

from __future__ import annotations

import sys

from managed_agents.api_client import update_agent
from managed_agents.config import AGENT_ID
from managed_agents.system_prompt import SYSTEM_PROMPT


def main() -> int:
    if not AGENT_ID:
        print(
            "No MANAGED_AGENT_ID configured. Run `python -m managed_agents.setup_agent` "
            "first, or set MANAGED_AGENT_ID in managed_agents/.env.",
            file=sys.stderr,
        )
        return 1

    print(f"Updating agent {AGENT_ID}")
    print(f"New system prompt length: {len(SYSTEM_PROMPT):,} chars")

    result = update_agent(AGENT_ID, system=SYSTEM_PROMPT)

    version = result.get("version", "?")
    print(f"Agent updated. version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
