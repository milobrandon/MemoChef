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
# Bridge selected secrets from .streamlit/secrets.toml into os.environ so
# downstream code (and the module-level constants below) can rely on env
# vars on Streamlit Cloud, where secrets are normally only exposed via
# st.secrets.
_SECRETS_TO_ENV = (
    "ANTHROPIC_API_KEY",
    "FIREFLIES_API_KEY",
    "MANAGED_AGENT_ID",
    "MANAGED_ENVIRONMENT_ID",
)
if _streamlit_secrets is not None:
    for _line in _streamlit_secrets.read_text().splitlines():
        if "=" not in _line or _line.lstrip().startswith("#"):
            continue
        _line_key = _line.split("=", 1)[0].strip()
        if _line_key in _SECRETS_TO_ENV and not os.environ.get(_line_key):
            _val = _line.split("=", 1)[1].strip().strip('"')
            if _val:
                os.environ[_line_key] = _val

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
FIREFLIES_API_KEY: str = os.environ.get("FIREFLIES_API_KEY", "")
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
