"""Unit tests for update application, branding, and layout normalization."""
import os

import pytest
from pptx import Presentation
from pptx.util import Inches

from memo_automator import apply_branding, apply_updates, normalize_layout


def _find_table(slide):
    for shape in slide.shapes:
        if shape.has_table:
            return shape.table
    raise AssertionError("No table found on slide")


def test_apply_updates_table_and_text(sample_pptx):
    mappings = {
        "table_updates": [
            {
                "page": 1,
                "table_name": "UnitMixTable",
                "row_label": "1BR",
                "column_index": 1,
                "old_value": "120",
                "new_value": "130",
                "source": "Executive Summary B3",
            }
        ],
        "text_updates": [
            {
                "page": 1,
                "old_text": "IRR is 5.0%",
                "new_text": "IRR is 6.5%",
                "source": "Executive Summary B2",
            }
        ],
        "row_inserts": [],
    }
    changes = apply_updates(sample_pptx, mappings, dry_run=False)
    assert len(changes) == 2

    prs = Presentation(sample_pptx)
    table = _find_table(prs.slides[0])
    assert table.cell(1, 1).text == "130"
    narrative = next(s for s in prs.slides[0].shapes if s.name == "NarrativeBox")
    assert "IRR is 6.5%" in narrative.text_frame.text


def test_apply_branding(sample_pptx):
    theme_path = os.path.join(os.path.dirname(__file__), "Subtext Brand Theme.thmx")
    if not os.path.exists(theme_path):
        pytest.skip("Theme file not found for branding test")

    cfg = {
        "branding": {
            "heading_size_threshold": 18,
            "color_distance_threshold": 80,
        }
    }
    runs = apply_branding(sample_pptx, theme_path, cfg)
    assert runs > 0

    prs = Presentation(sample_pptx)
    table = _find_table(prs.slides[0])
    header_run = table.cell(0, 0).text_frame.paragraphs[0].runs[0]
    body_run = table.cell(1, 0).text_frame.paragraphs[0].runs[0]
    assert header_run.font.name == "Pragmatica Bold"
    assert body_run.font.name == "Pragmatica Book"


def test_normalize_layout(layout_test_pptx):
    cfg = {
        "layout": {
            "margin_left": 0.50,
            "margin_right": 0.50,
            "margin_top": 0.25,
            "margin_bottom": 0.50,
            "snap_tolerance": 0.05,
        }
    }
    summary = normalize_layout(layout_test_pptx, cfg)
    assert "shapes_clamped_to_margins" in summary
    assert summary["shapes_clamped_to_margins"] >= 1

    prs = Presentation(layout_test_pptx)
    slide = prs.slides[2]
    off_margin = next(s for s in slide.shapes if s.name == "OffMargin")
    assert off_margin.left >= Inches(0.50)
    assert off_margin.top >= Inches(0.25)
