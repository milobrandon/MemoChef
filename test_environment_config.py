"""Unit tests for the limited-networking environment config.

Verifies the JSON shape that gets POSTed to /v1/environments matches
the docs and that the allowed-hosts list stays minimal.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from managed_agents import api_client
from managed_agents.environment_config import (
    ALLOWED_HOSTS,
    ENVIRONMENT_CONFIG,
    ENVIRONMENT_NAME,
)


def test_environment_name_is_set():
    assert ENVIRONMENT_NAME and isinstance(ENVIRONMENT_NAME, str)


def test_networking_is_limited_not_unrestricted():
    net = ENVIRONMENT_CONFIG["networking"]
    assert net["type"] == "limited", (
        "Production envs should never use unrestricted networking. "
        "If you need to add a host, append it to ALLOWED_HOSTS."
    )


def test_allowed_hosts_is_minimal():
    assert ENVIRONMENT_CONFIG["networking"]["allowed_hosts"] == ALLOWED_HOSTS
    assert "api.fireflies.ai" in ALLOWED_HOSTS
    # Guardrail: catch accidental wildcard or wide-open additions.
    for host in ALLOWED_HOSTS:
        assert "*" not in host, f"Wildcard host not allowed: {host}"
        assert "://" not in host, (
            f"Use bare hostnames in allowed_hosts, not URLs: {host}"
        )


def test_package_managers_blocked_at_runtime():
    net = ENVIRONMENT_CONFIG["networking"]
    assert net["allow_package_managers"] is False, (
        "Pip is blocked at runtime; required packages are pre-installed "
        "via packages.pip. Flip this only if a fix needs an unplanned dep."
    )


def test_mcp_servers_disabled():
    assert ENVIRONMENT_CONFIG["networking"]["allow_mcp_servers"] is False


def test_required_packages_are_preinstalled():
    pip = ENVIRONMENT_CONFIG["packages"]["pip"]
    required = {"python-pptx", "openpyxl", "pandas", "pdfplumber", "rapidfuzz"}
    found = {entry.split(">=")[0].split("==")[0] for entry in pip}
    missing = required - found
    assert not missing, f"Missing pre-installed packages: {missing}"


# ---- API wiring ----------------------------------------------------

class _FakeResp:
    def __init__(self, body: dict):
        self.status_code = 200
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


class _FakeClient:
    captured: dict | None = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None, **kwargs):
        type(self).captured = json
        return _FakeResp({"id": "env_test"})

    def get(self, url, headers=None, **kwargs):
        return _FakeResp({})


def test_create_environment_posts_limited_networking():
    _FakeClient.captured = None
    with patch.object(api_client.httpx, "Client", _FakeClient):
        api_client.create_environment(name=ENVIRONMENT_NAME, config=ENVIRONMENT_CONFIG)

    body = _FakeClient.captured
    assert body is not None
    assert body["name"] == ENVIRONMENT_NAME
    cfg = body["config"]
    assert cfg["networking"]["type"] == "limited"
    assert cfg["networking"]["allowed_hosts"] == ALLOWED_HOSTS
    assert cfg["networking"]["allow_package_managers"] is False
    assert cfg["networking"]["allow_mcp_servers"] is False
