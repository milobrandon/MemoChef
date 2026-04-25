#!/usr/bin/env python3
"""
Create the Memo Chef managed environment (cloud container).

Usage:
    python -m managed_agents.setup_environment

Pre-installs python-pptx, openpyxl, pandas, pdfplumber, rapidfuzz so
the agent can read/write Office files and PDFs without installing at runtime.

Networking is locked to api.fireflies.ai — the only outbound host the
agent legitimately needs (transcript fetches). Pip and other registries
are blocked at runtime, since required packages are pre-installed above.

Saves the environment ID to managed_agents/.env for reuse.
"""

from __future__ import annotations

from managed_agents.api_client import create_environment
from managed_agents.config import ENVIRONMENT_ID, save_ids
from managed_agents.environment_config import ENVIRONMENT_CONFIG, ENVIRONMENT_NAME


def main() -> None:
    if ENVIRONMENT_ID:
        print(f"Environment already provisioned: {ENVIRONMENT_ID}")
        print("Delete MANAGED_ENVIRONMENT_ID from managed_agents/.env to re-create,")
        print("or run `python -m managed_agents.recreate_environment` to migrate.")
        return

    env = create_environment(name=ENVIRONMENT_NAME, config=ENVIRONMENT_CONFIG)

    env_id = env["id"]
    print(f"Environment created: {env_id}")

    save_ids(environment_id=env_id)
    print("Saved environment ID to managed_agents/.env")


if __name__ == "__main__":
    main()
