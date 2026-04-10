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

from managed_agents.api_client import create_environment
from managed_agents.config import ENVIRONMENT_ID, save_ids


def main() -> None:
    if ENVIRONMENT_ID:
        print(f"Environment already provisioned: {ENVIRONMENT_ID}")
        print("Delete MANAGED_ENVIRONMENT_ID from managed_agents/.env to re-create.")
        return

    env = create_environment(
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

    env_id = env["id"]
    print(f"Environment created: {env_id}")

    save_ids(environment_id=env_id)
    print("Saved environment ID to managed_agents/.env")


if __name__ == "__main__":
    main()
