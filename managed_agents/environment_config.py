"""
Single source of truth for the Memo Chef managed environment config.

Kept in its own module so setup_environment.py and recreate_environment.py
share the exact same config dict — drift between them would silently
produce two environments with different networking or package lists.

Docs: https://platform.claude.com/docs/en/managed-agents/environments
"""

from __future__ import annotations

ENVIRONMENT_NAME = "memo-chef-env"

# Hosts the agent is allowed to reach. Today the only legitimate
# outbound is the Fireflies GraphQL API used for transcript search.
# Everything else (proforma, memo, supplemental files) is uploaded as
# a session resource — no network needed.
ALLOWED_HOSTS: list[str] = ["api.fireflies.ai"]

ENVIRONMENT_CONFIG: dict = {
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
    "networking": {
        "type": "limited",
        "allowed_hosts": ALLOWED_HOSTS,
        # Pip is blocked at runtime — the packages list above is
        # installed at provision time and that's all the agent gets.
        "allow_package_managers": False,
        # No MCP servers configured on the agent.
        "allow_mcp_servers": False,
    },
}
