"""Shared pytest fixtures for Memo Automator tests."""
import os
import tempfile

import pytest
from pptx import Presentation
from pptx.util import Inches


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_pptx(tmp_dir):
    """Create a minimal PPTX with a 4x3 table for testing.

    Returns the path to the saved file.
    """
    path = os.path.join(tmp_dir, "test_memo.pptx")
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # blank layout

    # Add a table: 4 rows x 3 columns
    table_shape = slide.shapes.add_table(
        4, 3, Inches(0.5), Inches(1.0), Inches(8.0), Inches(3.0),
    )
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


@pytest.fixture
def sample_config(tmp_dir):
    """Create a minimal config.yaml for testing. Returns the path."""
    path = os.path.join(tmp_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(
            "proforma:\n"
            "  tabs:\n"
            '    - "Executive Summary"\n'
            '    - "Cash Flow"\n'
            "  max_rows_per_tab: 100\n"
            "  max_cols_per_tab: 20\n"
            "memo:\n"
            '  pages: "all"\n'
            "claude:\n"
            '  model: "claude-sonnet-4-6"\n'
            "  max_tokens: 8000\n"
            "  temperature: 0\n"
        )
    return path
