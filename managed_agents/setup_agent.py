#!/usr/bin/env python3
"""
Create (or update) the Memo Chef managed agent.

Usage:
    python -m managed_agents.setup_agent

Saves the agent ID to managed_agents/.env for reuse across sessions.
"""

from __future__ import annotations

import anthropic

from managed_agents.config import ANTHROPIC_API_KEY, AGENT_ID, save_ids
from managed_agents.system_prompt import SYSTEM_PROMPT


def create_agent() -> str:
    """Create the Memo Chef agent and return its ID."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    agent = client.beta.agents.create(
        name="Memo Chef",
        description="Autonomous IC memo updater for Subtext student-housing deals.",
        model="claude-sonnet-4-6",
        system=SYSTEM_PROMPT,
        tools=[
            {"type": "agent_toolset_20260401"},
        ],
    )

    print(f"Agent created: {agent.id} (version {agent.version})")
    return agent.id


def main() -> None:
    if AGENT_ID:
        print(f"Agent already provisioned: {AGENT_ID}")
        print("Delete managed_agents/.env entry to re-create.")
        return

    agent_id = create_agent()
    save_ids(agent_id=agent_id)
    print(f"Saved agent ID to managed_agents/.env")


if __name__ == "__main__":
    main()
