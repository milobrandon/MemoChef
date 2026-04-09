"""
Configuration for the Managed Agents backend.

After running setup_agent.py and setup_environment.py, their IDs are
persisted in managed_agents/.env so they survive across sessions.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load project root .env first (for ANTHROPIC_API_KEY), then local overrides
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
AGENT_ID: str = os.environ.get("MANAGED_AGENT_ID", "")
ENVIRONMENT_ID: str = os.environ.get("MANAGED_ENVIRONMENT_ID", "")

# Paths
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def save_ids(*, agent_id: str | None = None, environment_id: str | None = None) -> None:
    """Persist agent/environment IDs to managed_agents/.env for reuse."""
    env_path = Path(__file__).resolve().parent / ".env"
    lines: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                lines[k.strip()] = v.strip()
    if agent_id:
        lines["MANAGED_AGENT_ID"] = agent_id
    if environment_id:
        lines["MANAGED_ENVIRONMENT_ID"] = environment_id
    env_path.write_text("\n".join(f"{k}={v}" for k, v in lines.items()) + "\n")
