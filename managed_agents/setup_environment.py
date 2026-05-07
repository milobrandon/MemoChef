#!/usr/bin/env python3
"""
Create the Memo Chef managed environment (cloud container).

Usage:
    python -m managed_agents.setup_environment

Reads the environment spec from
``managed_agents/environments/memo-chef.environment.yaml`` so the config is
version-controlled and reviewable. The YAML shape matches the Anthropic CLI
(`ant beta:environments create`), so a future migration to `ant` is a no-op.

Saves the environment ID to managed_agents/.env for reuse.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from managed_agents.api_client import create_environment
from managed_agents.config import ENVIRONMENT_ID, save_ids

ENVIRONMENT_SPEC_PATH = (
    Path(__file__).resolve().parent / "environments" / "memo-chef.environment.yaml"
)


def load_spec(path: Path = ENVIRONMENT_SPEC_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    if ENVIRONMENT_ID:
        print(f"Environment already provisioned: {ENVIRONMENT_ID}")
        print("Delete MANAGED_ENVIRONMENT_ID from managed_agents/.env to re-create.")
        return

    spec = load_spec()
    env = create_environment(name=spec["name"], config=spec["config"])

    env_id = env["id"]
    print(f"Environment created: {env_id}")

    save_ids(environment_id=env_id)
    print("Saved environment ID to managed_agents/.env")


if __name__ == "__main__":
    main()
