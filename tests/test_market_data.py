"""Tests for market data extraction and chart update functionality."""

import os
import sys
import tempfile

import openpyxl
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memo_automator import (
    extract_market_data,
    load_config,
    _MARKET_DASHBOARD_TABS,
)


@pytest.fixture
def default_cfg():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    return load_config(config_path)


@pytest.fixture
def market_data_workbook(tmp_path):
    """Create a minimal market data workbook with dashboard tabs."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Tables"
    ws["A1"] = "IPEDS"
    ws["B1"] = 134130
    ws["A2"] = "University Name"
    ws["B2"] = "UF"

    for tab_name in _MARKET_DASHBOARD_TABS[1:]:
        ws2 = wb.create_sheet(tab_name)
        ws2["A1"] = f"Header for {tab_name}"
        ws2["B1"] = 123.45

    # Add a back-end tab (should be ignored)
    ws3 = wb.create_sheet("PROPERTIES")
    ws3["A1"] = "This should not be extracted"

    path = str(tmp_path / "test_market_data.xlsx")
    wb.save(path)
    wb.close()
    return path


@pytest.fixture
def empty_workbook(tmp_path):
    """Create a workbook with no dashboard tabs."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "RandomSheet"
    ws["A1"] = "Not a dashboard"
    path = str(tmp_path / "empty_market.xlsx")
    wb.save(path)
    wb.close()
    return path


class TestExtractMarketData:
    def test_happy_path(self, market_data_workbook, default_cfg):
        result = extract_market_data(market_data_workbook, default_cfg)
        assert "MARKET DATA (from RealPage)" in result
        assert "TAB: Tables" in result
        assert "TAB: Comp Set" in result
        assert "IPEDS" in result
        assert "134130" in result

    def test_ignores_backend_tabs(self, market_data_workbook, default_cfg):
        result = extract_market_data(market_data_workbook, default_cfg)
        assert "PROPERTIES" not in result.split("MARKET DATA")[1]  # Only check after header
        assert "This should not be extracted" not in result

    def test_all_dashboard_tabs_extracted(self, market_data_workbook, default_cfg):
        result = extract_market_data(market_data_workbook, default_cfg)
        for tab in _MARKET_DASHBOARD_TABS:
            assert f"TAB: {tab}" in result

    def test_no_dashboard_tabs_returns_empty(self, empty_workbook, default_cfg):
        result = extract_market_data(empty_workbook, default_cfg)
        assert result == ""

    def test_invalid_file_returns_empty(self, tmp_path, default_cfg):
        bad_path = str(tmp_path / "not_a_real_file.xlsx")
        with open(bad_path, "w") as f:
            f.write("not an excel file")
        result = extract_market_data(bad_path, default_cfg)
        assert result == ""

    def test_missing_file_returns_empty(self, default_cfg):
        result = extract_market_data("/nonexistent/path.xlsx", default_cfg)
        assert result == ""

    def test_partial_tabs(self, tmp_path, default_cfg):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Tables"
        ws["A1"] = "Data"
        ws2 = wb.create_sheet("Comp Set")
        ws2["A1"] = "Comps"
        path = str(tmp_path / "partial.xlsx")
        wb.save(path)
        wb.close()

        result = extract_market_data(path, default_cfg)
        assert "TAB: Tables" in result
        assert "TAB: Comp Set" in result
        assert "TAB: Comparison Graph" not in result

    def test_output_format_rows(self, market_data_workbook, default_cfg):
        """Verify tab-delimited row format matches proforma pattern."""
        result = extract_market_data(market_data_workbook, default_cfg)
        assert "Row 1:\t" in result  # Tab-delimited format


class TestRealMarketDataFile:
    REAL_FILE = "C:/Users/BrandonZmuda/Desktop/Claude/g. Memo Automator/a. Sandbox/New Template Test.xlsx"

    @pytest.mark.skipif(
        not os.path.exists(REAL_FILE),
        reason="Real market data file not available",
    )
    def test_real_file_extraction(self, default_cfg):
        result = extract_market_data(self.REAL_FILE, default_cfg)
        assert len(result) > 1000
        assert "MARKET DATA (from RealPage)" in result
        assert "TAB: Tables" in result
        assert "TAB: Comp Set" in result

    @pytest.mark.skipif(
        not os.path.exists(REAL_FILE),
        reason="Real market data file not available",
    )
    def test_real_file_prompt_size(self, default_cfg):
        """Verify market data doesn't blow up prompt size unreasonably."""
        result = extract_market_data(self.REAL_FILE, default_cfg)
        # Dashboard tabs should be well under 50K chars
        assert len(result) < 50000, f"Market data too large: {len(result)} chars"
