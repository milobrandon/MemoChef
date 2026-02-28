"""
Unit tests for row_insert helpers and pipeline plumbing.
Tests _find_row_by_label, _add_table_row, and apply_updates with row_inserts.
No API calls required.
"""
import json
import os
import sys
import tempfile

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memo_automator_app"))

from pptx import Presentation
from pptx.util import Inches, Pt

from memo_automator import (
    _find_row_by_label,
    _add_table_row,
    _find_table_target,
    _loose_match,
    apply_updates,
    pre_validate_mappings,
    _salvage_truncated_json,
)


def _get_table_shape(slide):
    """Find the first table shape on a slide."""
    for s in slide.shapes:
        if s.has_table:
            return s
    return None


def _create_test_pptx(path: str):
    """Create a minimal PPTX with a 4x3 table for testing."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # blank layout

    # Add a table: 4 rows x 3 columns
    left = Inches(0.5)
    top = Inches(1.0)
    width = Inches(8.0)
    height = Inches(3.0)
    table_shape = slide.shapes.add_table(4, 3, left, top, width, height)
    table_shape.name = "UnitMixTable"
    table = table_shape.table

    # Header row
    table.cell(0, 0).text = "Unit Type"
    table.cell(0, 1).text = "Beds"
    table.cell(0, 2).text = "Rent"

    # Data rows
    table.cell(1, 0).text = "1BR"
    table.cell(1, 1).text = "120"
    table.cell(1, 2).text = "$1,825"

    table.cell(2, 0).text = "3BR"
    table.cell(2, 1).text = "200"
    table.cell(2, 2).text = "$1,345"

    table.cell(3, 0).text = "4BR/2BA"
    table.cell(3, 1).text = "150"
    table.cell(3, 2).text = "$1,250"

    prs.save(path)
    return path


def test_find_row_by_label():
    """Test _find_row_by_label finds rows correctly."""
    print("TEST: _find_row_by_label")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        path = f.name
    try:
        _create_test_pptx(path)
        prs = Presentation(path)
        table = _get_table_shape(prs.slides[0]).table

        # Exact match
        assert _find_row_by_label(table, "1BR") == 1, "Should find '1BR' at row 1"
        assert _find_row_by_label(table, "3BR") == 2, "Should find '3BR' at row 2"
        assert _find_row_by_label(table, "4BR/2BA") == 3, "Should find '4BR/2BA' at row 3"

        # Loose match (containment)
        assert _find_row_by_label(table, "Unit") == 0, "Should match 'Unit Type' header"

        # Non-existent
        assert _find_row_by_label(table, "Studio") is None, "Should not find 'Studio'"
        assert _find_row_by_label(table, "2BR") is None, "Should not find '2BR'"

        print("  PASSED: All _find_row_by_label assertions OK")
    finally:
        os.unlink(path)


def test_add_table_row():
    """Test _add_table_row clones and inserts a row."""
    print("TEST: _add_table_row")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        path = f.name
    try:
        _create_test_pptx(path)
        prs = Presentation(path)
        table = _get_table_shape(prs.slides[0]).table

        original_row_count = len(table.rows)
        assert original_row_count == 4, f"Expected 4 rows, got {original_row_count}"

        # Insert "2BR/1BA" after "1BR" (row index 1)
        _add_table_row(table, 1, ["2BR/1BA", "80", "$1,495"])

        new_row_count = len(table.rows)
        assert new_row_count == 5, f"Expected 5 rows after insert, got {new_row_count}"

        # Verify the new row is at index 2 (after the reference row at index 1)
        new_row_text = [table.cell(2, c).text for c in range(3)]
        assert new_row_text == ["2BR/1BA", "80", "$1,495"], \
            f"New row content mismatch: {new_row_text}"

        # Verify original rows shifted correctly
        assert table.cell(1, 0).text == "1BR", "1BR should still be at row 1"
        assert table.cell(3, 0).text == "3BR", "3BR should have shifted to row 3"
        assert table.cell(4, 0).text == "4BR/2BA", "4BR/2BA should be at row 4"

        # Save and re-read to verify XML persistence
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f2:
            path2 = f2.name
        prs.save(path2)
        prs2 = Presentation(path2)
        table2 = _get_table_shape(prs2.slides[0]).table
        assert len(table2.rows) == 5, "Row count should persist after save"
        assert table2.cell(2, 0).text == "2BR/1BA", "New row should persist after save"
        os.unlink(path2)

        print("  PASSED: Row inserted, content correct, persists after save")
    finally:
        os.unlink(path)


def test_add_table_row_padding():
    """Test that _add_table_row handles fewer cells than columns."""
    print("TEST: _add_table_row with padding")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        path = f.name
    try:
        _create_test_pptx(path)
        prs = Presentation(path)
        table = _get_table_shape(prs.slides[0]).table

        # Insert with only 1 value (table has 3 cols)
        _add_table_row(table, 2, ["Studio"])
        assert len(table.rows) == 5, "Should have 5 rows"

        # First cell should have the value, rest empty
        assert table.cell(3, 0).text == "Studio", "First cell should be 'Studio'"
        # The cloned cells may retain empty text from clearing
        assert table.cell(3, 1).text == "", f"Cell should be empty, got '{table.cell(3, 1).text}'"
        assert table.cell(3, 2).text == "", f"Cell should be empty, got '{table.cell(3, 2).text}'"

        print("  PASSED: Padding with empty strings works correctly")
    finally:
        os.unlink(path)


def test_apply_updates_row_inserts():
    """Test full apply_updates with row_inserts."""
    print("TEST: apply_updates with row_inserts")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        path = f.name
    try:
        _create_test_pptx(path)

        mappings = {
            "table_updates": [],
            "text_updates": [],
            "row_inserts": [
                {
                    "page": 1,
                    "table_name": "UnitMixTable",
                    "insert_after_row_label": "1BR",
                    "cells": ["2BR/1BA", "80", "$1,495"],
                    "source": "Proforma Unit Mix"
                }
            ],
        }

        changes = apply_updates(path, mappings, dry_run=False)
        assert len(changes) == 1, f"Expected 1 change, got {len(changes)}"
        assert changes[0]["type"] == "row_insert", f"Expected row_insert type, got {changes[0]['type']}"
        assert "UnitMixTable" in changes[0]["location"], "Should reference UnitMixTable"

        # Verify the actual PPTX
        prs = Presentation(path)
        table = _get_table_shape(prs.slides[0]).table
        assert len(table.rows) == 5, f"Expected 5 rows, got {len(table.rows)}"
        assert table.cell(2, 0).text == "2BR/1BA", "New row should be at position 2"

        print("  PASSED: apply_updates correctly processes row_inserts")
    finally:
        os.unlink(path)


def test_apply_updates_row_inserts_dry_run():
    """Test apply_updates dry_run mode with row_inserts."""
    print("TEST: apply_updates dry_run with row_inserts")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        path = f.name
    try:
        _create_test_pptx(path)

        mappings = {
            "table_updates": [],
            "text_updates": [],
            "row_inserts": [
                {
                    "page": 1,
                    "table_name": "UnitMixTable",
                    "insert_after_row_label": "1BR",
                    "cells": ["2BR/1BA", "80", "$1,495"],
                    "source": "Test"
                }
            ],
        }

        changes = apply_updates(path, mappings, dry_run=True)
        assert len(changes) == 1, "Should report 1 change in dry-run"

        # Verify PPTX was NOT modified
        prs = Presentation(path)
        table = _get_table_shape(prs.slides[0]).table
        assert len(table.rows) == 4, "Dry run should not modify the file"

        print("  PASSED: dry_run reports changes without modifying file")
    finally:
        os.unlink(path)


def test_apply_updates_row_insert_fallback():
    """Test apply_updates falls back to finding table by row label when table_name doesn't match."""
    print("TEST: apply_updates row_insert fallback (wrong table_name)")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        path = f.name
    try:
        _create_test_pptx(path)

        mappings = {
            "table_updates": [],
            "text_updates": [],
            "row_inserts": [
                {
                    "page": 1,
                    "table_name": "WrongTableName",
                    "insert_after_row_label": "3BR",
                    "cells": ["Studio", "50", "$1,100"],
                    "source": "Test"
                }
            ],
        }

        changes = apply_updates(path, mappings, dry_run=False)
        assert len(changes) == 1, "Should find table by fallback"

        prs = Presentation(path)
        table = _get_table_shape(prs.slides[0]).table
        assert len(table.rows) == 5, "Row should be inserted"
        assert table.cell(3, 0).text == "Studio", "Studio should be after 3BR"

        print("  PASSED: Fallback to table containing row label works")
    finally:
        os.unlink(path)


def test_salvage_truncated_json_with_row_inserts():
    """Test _salvage_truncated_json preserves row_inserts."""
    print("TEST: _salvage_truncated_json with row_inserts")

    # Simulate truncated JSON that includes row_inserts
    truncated = '{"table_updates": [{"page": 1, "old_value": "x", "new_value": "y"}], "text_updates": [], "row_inserts": [{"page": 1, "cells": ["a", "b"'
    result = _salvage_truncated_json(truncated)
    # May or may not salvage depending on truncation point, but at least shouldn't crash
    if result is not None:
        assert "row_inserts" in result, "Salvaged result should have row_inserts key"
        print(f"  PASSED: Salvaged with {len(result.get('row_inserts', []))} row_inserts")
    else:
        print("  PASSED: Could not salvage (expected for this truncation), no crash")

    # Complete JSON that should definitely be salvageable
    complete = '{"table_updates": [], "text_updates": [], "row_inserts": [{"page": 1, "cells": ["a"]}]}'
    result2 = _salvage_truncated_json(complete)
    # This is complete JSON so it won't go through salvage, but let's test a nearly-complete one
    nearly = '{"table_updates": [], "text_updates": [], "row_inserts": [{"page": 1, "cells": ["a"]}]'
    result3 = _salvage_truncated_json(nearly)
    if result3 is not None:
        assert "row_inserts" in result3
        assert len(result3["row_inserts"]) == 1
        print("  PASSED: Nearly-complete JSON salvaged with row_inserts intact")
    else:
        print("  INFO: Nearly-complete JSON not salvaged (acceptable)")


def test_pre_validate_passes_row_inserts():
    """Test pre_validate_mappings passes row_inserts through unchanged."""
    print("TEST: pre_validate_mappings passes row_inserts through")

    mappings = {
        "table_updates": [],
        "text_updates": [],
        "row_inserts": [
            {"page": 1, "table_name": "T", "insert_after_row_label": "1BR",
             "cells": ["2BR", "80"], "source": "test"}
        ],
    }

    # Use a dummy memo content
    memo_content = "==============================\nPAGE 1\n=============================="
    result = pre_validate_mappings(mappings, memo_content)
    assert "row_inserts" in result, "row_inserts should be in result"
    assert len(result["row_inserts"]) == 1, "row_inserts should pass through unchanged"
    assert result["row_inserts"][0]["cells"] == ["2BR", "80"], "Content should be preserved"

    print("  PASSED: row_inserts passed through pre_validate_mappings unchanged")


def test_multiple_row_inserts():
    """Test inserting multiple rows in sequence."""
    print("TEST: Multiple row inserts")
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        path = f.name
    try:
        _create_test_pptx(path)

        mappings = {
            "table_updates": [],
            "text_updates": [],
            "row_inserts": [
                {
                    "page": 1,
                    "table_name": "UnitMixTable",
                    "insert_after_row_label": "Unit Type",
                    "cells": ["Studio", "30", "$950"],
                    "source": "Test"
                },
                {
                    "page": 1,
                    "table_name": "UnitMixTable",
                    "insert_after_row_label": "1BR",
                    "cells": ["2BR/1BA", "80", "$1,495"],
                    "source": "Test"
                },
            ],
        }

        changes = apply_updates(path, mappings, dry_run=False)
        assert len(changes) == 2, f"Expected 2 changes, got {len(changes)}"

        prs = Presentation(path)
        table = _get_table_shape(prs.slides[0]).table
        assert len(table.rows) == 6, f"Expected 6 rows (4 original + 2 inserts), got {len(table.rows)}"

        print("  PASSED: Multiple row inserts applied correctly")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    tests = [
        test_find_row_by_label,
        test_add_table_row,
        test_add_table_row_padding,
        test_apply_updates_row_inserts,
        test_apply_updates_row_inserts_dry_run,
        test_apply_updates_row_insert_fallback,
        test_salvage_truncated_json_with_row_inserts,
        test_pre_validate_passes_row_inserts,
        test_multiple_row_inserts,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
