#!/usr/bin/env python3
"""
Create the Memo Chef managed environment (cloud container).

Usage:
    python -m managed_agents.setup_environment

Pre-installs python-pptx, openpyxl, pandas, pdfplumber, rapidfuzz so
the agent can read/write Office files and PDFs without installing at runtime.

Saves the environment ID to managed_agents/.env for reuse.
"""

from __future__ import annotations

import anthropic

from managed_agents.config import ANTHROPIC_API_KEY, ENVIRONMENT_ID, save_ids


def create_environment() -> str:
    """Create the cloud environment and return its ID."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    environment = client.beta.environments.create(
        name="memo-chef-env",
        config={
            "type": "cloud",
            "packages": {
                "pip": [
                    "python-pptx>=1.0.0",
                    "openpyxl>=3.1.0",
                    "pandas>=2.0.0",
                    "pdfplumber>=0.10.0",
                    "rapidfuzz>=3.0",
                ],
            },
            "networking": {"type": "unrestricted"},
        },
    )

    print(f"Environment created: {environment.id}")
    return environment.id


def main() -> None:
    if ENVIRONMENT_ID:
        print(f"Environment already provisioned: {ENVIRONMENT_ID}")
        print("Delete managed_agents/.env entry to re-create.")
        return

    env_id = create_environment()
    save_ids(environment_id=env_id)
    print(f"Saved environment ID to managed_agents/.env")


if __name__ == "__main__":
    main()
