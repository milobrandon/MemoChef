"""Tests for proforma drift detection."""
from memo_chef.drift import parse_proforma_to_cells, compute_proforma_diff


SAMPLE_PROFORMA_TEXT = """
======================================================================
TAB: Assumptions
======================================================================
Row 1:\tProperty Name\tThe Reserve
Row 2:\tTotal Units\t250
Row 3:\tTotal Beds\t510
Row 5:\tAvg Rent/Bed\t1325

======================================================================
TAB: Development Summary
======================================================================
Row 1:\tCategory\tAmount\t% of Total
Row 2:\tLand\t12500000\t0.2403846
Row 3:\tHard Costs\t28000000\t0.5384615
"""


def test_parse_proforma_basic():
    result = parse_proforma_to_cells(SAMPLE_PROFORMA_TEXT)
    assert "Assumptions" in result
    assert "Development Summary" in result
    assert result["Assumptions"][1] == ["Property Name", "The Reserve"]
    assert result["Assumptions"][5] == ["Avg Rent/Bed", "1325"]
    assert result["Development Summary"][2] == ["Land", "12500000", "0.2403846"]


def test_parse_proforma_empty_string():
    result = parse_proforma_to_cells("")
    assert result == {}


def test_parse_proforma_skips_empty_rows():
    """Row 4 is missing from sample — should not appear in output."""
    result = parse_proforma_to_cells(SAMPLE_PROFORMA_TEXT)
    assert 4 not in result["Assumptions"]


SAMPLE_PROFORMA_V2 = """
======================================================================
TAB: Assumptions
======================================================================
Row 1:\tProperty Name\tThe Reserve
Row 2:\tTotal Units\t260
Row 3:\tTotal Beds\t530
Row 5:\tAvg Rent/Bed\t1350
Row 7:\tNew Metric\t42

======================================================================
TAB: Development Summary
======================================================================
Row 1:\tCategory\tAmount\t% of Total
Row 2:\tLand\t13000000\t0.25
Row 3:\tHard Costs\t28000000\t0.5384615
"""


def test_diff_detects_changed_values():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assumptions = diff["by_tab"]["Assumptions"]
    changed_rows = [c["row"] for c in assumptions["changed"]]
    assert 2 in changed_rows  # Units 250 -> 260
    assert 3 in changed_rows  # Beds 510 -> 530
    assert 5 in changed_rows  # Rent 1325 -> 1350


def test_diff_detects_added_rows():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assumptions = diff["by_tab"]["Assumptions"]
    added_rows = [a["row"] for a in assumptions["added"]]
    assert 7 in added_rows


def test_diff_detects_removed_rows():
    # Remove a row from v2 that exists in v1 — use reversed order
    diff = compute_proforma_diff(SAMPLE_PROFORMA_V2, SAMPLE_PROFORMA_TEXT)
    assumptions = diff["by_tab"]["Assumptions"]
    removed_rows = [r["row"] for r in assumptions["removed"]]
    assert 7 in removed_rows


def test_diff_unchanged_rows_not_in_changed():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assumptions = diff["by_tab"]["Assumptions"]
    changed_rows = [c["row"] for c in assumptions["changed"]]
    assert 1 not in changed_rows  # Property Name unchanged


def test_diff_total_changes():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    # Assumptions: 3 changed (rows 2,3,5) + 1 added (row 7)
    # Dev Summary: 1 changed (row 2)
    assert diff["total_changes"] == 5


def test_diff_summary_string():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assert "5 values changed" in diff["summary"]
    assert "2 tabs" in diff["summary"]


def test_diff_identical_proformas():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_TEXT)
    assert diff["total_changes"] == 0
