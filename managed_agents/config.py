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

# Also try Streamlit secrets (TOML with KEY = "value" lines).
# Check both the project root and the main repo (for worktree scenarios).
_main_repo = _project_root
_root_str = str(_project_root).replace("\\", "/")
if ".claude/worktrees" in _root_str:
    _main_repo = Path(_root_str.split(".claude/worktrees")[0].rstrip("/"))
_streamlit_secrets = None
for _candidate in [_project_root / ".streamlit" / "secrets.toml",
                   _main_repo / ".streamlit" / "secrets.toml"]:
    if _candidate.exists():
        _streamlit_secrets = _candidate
        break
if _streamlit_secrets is not None and not os.environ.get("ANTHROPIC_API_KEY"):
    for _line in _streamlit_secrets.read_text().splitlines():
        if _line.startswith("ANTHROPIC_API_KEY"):
            _val = _line.split("=", 1)[1].strip().strip('"')
            os.environ["ANTHROPIC_API_KEY"] = _val
            break

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
FIREFLIES_API_KEY: str = os.environ.get("FIREFLIES_API_KEY", "")
AGENT_ID: str = os.environ.get("MANAGED_AGENT_ID", "")
ENVIRONMENT_ID: str = os.environ.get("MANAGED_ENVIRONMENT_ID", "")

# Load Fireflies key from Streamlit secrets if not in env
if not FIREFLIES_API_KEY and _streamlit_secrets is not None:
    for _line in _streamlit_secrets.read_text().splitlines():
        if _line.startswith("FIREFLIES_API_KEY"):
            FIREFLIES_API_KEY = _line.split("=", 1)[1].strip().strip('"')
            break

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
