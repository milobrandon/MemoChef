"""Playwright browser tests for Memo Chef Streamlit UI.

Run with:
    python -m pytest tests/test_playwright_ui.py -v --headed  (watch it)
    python -m pytest tests/test_playwright_ui.py -v            (headless)

Requires:
    pip install playwright pytest-playwright
    python -m playwright install chromium

The Streamlit app must be running on localhost:8501:
    streamlit run app.py --server.headless true --server.port 8501
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# Skip entire module if playwright is not installed
pw = pytest.importorskip("playwright.sync_api")


APP_URL = os.environ.get("MEMO_CHEF_URL", "http://localhost:8501")
USERNAME = os.environ.get("MEMO_CHEF_USER")
if not USERNAME:
    raise RuntimeError(
        "MEMO_CHEF_USER environment variable is required. "
        "Set it before running Playwright tests."
    )
PASSWORD = os.environ.get("MEMO_CHEF_PASS")
if not PASSWORD:
    raise RuntimeError(
        "MEMO_CHEF_PASS environment variable is required. "
        "Set it before running Playwright tests."
    )
WAIT_HYDRATE = 4  # seconds for Streamlit to hydrate

SANDBOX_DIR = Path(__file__).parent.parent / "a. Sandbox"
TEST_MEMO = SANDBOX_DIR / "EVER Lexington_20260205_TEST.pptx"
TEST_PROFORMA = SANDBOX_DIR / "Proforma_Lexington-Limestone_20241021.xlsm"


@pytest.fixture(scope="module")
def browser():
    """Launch browser once for all tests in this module."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(headless=not os.environ.get("HEADED"))
        yield b
        b.close()


@pytest.fixture
def authenticated_page(browser):
    """Return a page that is logged into Memo Chef."""
    page = browser.new_page()
    page.goto(APP_URL, timeout=20_000)
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=20_000)
    time.sleep(WAIT_HYDRATE)

    # Login
    page.locator('input[aria-label="Username"]').fill(USERNAME)
    page.locator('input[aria-label="Password"]').fill(PASSWORD)
    page.get_by_test_id("stBaseButton-primaryFormSubmit").click()
    time.sleep(WAIT_HYDRATE + 2)

    body = page.inner_text('[data-testid="stAppViewContainer"]')
    assert "Invalid username" not in body, "Login failed"

    yield page
    page.close()


class TestLoginFlow:
    def test_app_loads(self, browser):
        page = browser.new_page()
        page.goto(APP_URL, timeout=20_000)
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=20_000)
        time.sleep(WAIT_HYDRATE)
        body = page.inner_text('[data-testid="stAppViewContainer"]')
        assert "Memo Automator" in body
        assert "Username" in body
        assert "Password" in body
        page.close()

    def test_bad_password_rejected(self, browser):
        page = browser.new_page()
        page.goto(APP_URL, timeout=20_000)
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=20_000)
        time.sleep(WAIT_HYDRATE)
        page.locator('input[aria-label="Username"]').fill("brandon")
        page.locator('input[aria-label="Password"]').fill("wrongpassword")
        page.get_by_test_id("stBaseButton-primaryFormSubmit").click()
        time.sleep(3)
        body = page.inner_text('[data-testid="stAppViewContainer"]')
        assert "Invalid username or password" in body
        page.close()

    def test_login_success(self, authenticated_page):
        body = authenticated_page.inner_text('[data-testid="stAppViewContainer"]')
        assert "brandon" in body.lower() or "admin" in body.lower()
        assert "Sign out" in body


class TestMainUI:
    def test_no_exceptions(self, authenticated_page):
        errors = authenticated_page.query_selector_all('[data-testid="stException"]')
        assert len(errors) == 0, f"Found {len(errors)} exceptions on page"

    def test_file_uploaders_present(self, authenticated_page):
        uploaders = authenticated_page.query_selector_all(
            '[data-testid="stFileUploader"]'
        )
        # Memo + Proforma + Schedule + Market data + Supplemental = 5
        assert len(uploaders) >= 2, f"Expected >=2 uploaders, got {len(uploaders)}"

    def test_directions_for_claude_expander(self, authenticated_page):
        body = authenticated_page.inner_text('[data-testid="stAppViewContainer"]')
        assert "Directions for Claude" in body, "Missing per-source directives UI"

    def test_config_profile_selectbox(self, authenticated_page):
        selectboxes = authenticated_page.query_selector_all(
            '[data-testid="stSelectbox"]'
        )
        assert len(selectboxes) >= 1, "Expected at least 1 selectbox"

    def test_run_controls(self, authenticated_page):
        body = authenticated_page.inner_text('[data-testid="stAppViewContainer"]')
        # These are key controls that must be visible
        for label in ["Saved profile", "Memo deck", "Proforma"]:
            assert label in body, f"Missing UI element: {label}"

    def test_tabs_present(self, authenticated_page):
        body = authenticated_page.inner_text('[data-testid="stAppViewContainer"]')
        for tab in ["New Run", "Run History", "Operations"]:
            assert tab in body or tab.lower() in body.lower(), f"Missing tab: {tab}"


class TestFileUpload:
    @pytest.mark.skipif(
        not TEST_MEMO.exists() or not TEST_PROFORMA.exists(),
        reason="Sandbox test files not available",
    )
    def test_upload_memo_and_proforma(self, authenticated_page):
        """Upload test files and verify they're accepted."""
        page = authenticated_page

        # Find the memo uploader (first one)
        uploaders = page.query_selector_all('[data-testid="stFileUploader"]')
        assert len(uploaders) >= 2

        # Upload memo
        memo_input = uploaders[0].query_selector('input[type="file"]')
        memo_input.set_input_files(str(TEST_MEMO))
        time.sleep(2)

        # Upload proforma
        proforma_input = uploaders[1].query_selector('input[type="file"]')
        proforma_input.set_input_files(str(TEST_PROFORMA))
        time.sleep(2)

        # Check that files were accepted (no error messages)
        errors = page.query_selector_all('[data-testid="stException"]')
        assert len(errors) == 0, "Exception after file upload"

        # Take screenshot for review
        page.screenshot(
            path="test_output_randomized/ui_files_uploaded.png", full_page=True
        )
