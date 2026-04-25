#!/usr/bin/env python3
"""
Create a new managed environment with the current config and switch to it.

Managed-agent environments are immutable: once created, neither the
networking config nor the package list can be edited. To roll out a
config change you have to provision a fresh environment and point
MANAGED_ENVIRONMENT_ID at it.

This script:
  1. Creates a new environment using ENVIRONMENT_CONFIG.
  2. Prints the OLD and NEW IDs so you can archive the old one later.
  3. Updates MANAGED_ENVIRONMENT_ID in managed_agents/.env (atomic).

Existing in-flight sessions on the old environment keep running —
archive happens separately via the API once nothing references it.

Usage:
    python -m managed_agents.recreate_environment
"""

from __future__ import annotations

import sys

from managed_agents.api_client import create_environment
from managed_agents.config import ENVIRONMENT_ID, save_ids
from managed_agents.environment_config import ENVIRONMENT_CONFIG, ENVIRONMENT_NAME


def main() -> int:
    old_id = ENVIRONMENT_ID
    print(f"Old environment ID: {old_id or '(none)'}")
    print(f"Creating new environment with name '{ENVIRONMENT_NAME}'...")

    # Append a short suffix so the name is unique within the workspace.
    # Anthropic requires names be unique per org+workspace; reusing the
    # exact same name as an existing env would 4xx.
    import datetime
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    name = f"{ENVIRONMENT_NAME}-{stamp}"

    env = create_environment(name=name, config=ENVIRONMENT_CONFIG)
    new_id = env["id"]

    print(f"New environment ID: {new_id}")
    print(f"  name:       {name}")
    print(f"  networking: {ENVIRONMENT_CONFIG['networking']}")

    save_ids(environment_id=new_id)
    print("Updated MANAGED_ENVIRONMENT_ID in managed_agents/.env")

    if old_id:
        print()
        print(f"Old environment {old_id} is still active. Archive it once nothing references it:")
        print(f"  curl -X POST https://api.anthropic.com/v1/environments/{old_id}/archive \\")
        print('    -H "x-api-key: $ANTHROPIC_API_KEY" \\')
        print('    -H "anthropic-version: 2023-06-01" \\')
        print('    -H "anthropic-beta: managed-agents-2026-04-01"')

    return 0


if __name__ == "__main__":
    sys.exit(main())
