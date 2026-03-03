"""Unit tests for proforma and memo extraction."""
import os

import pytest

from memo_automator import extract_memo_content, extract_proforma_data


def _base_cfg():
    return {
        "proforma": {
            "tabs": ["Executive Summary", "Cash Flow"],
            "max_rows_per_tab": 250,
            "max_cols_per_tab": 30,
        },
        "memo": {"pages": "all"},
    }


def test_extract_proforma_data_success(sample_proforma_xlsx):
    text = extract_proforma_data(sample_proforma_xlsx, _base_cfg())
    assert "TAB: Executive Summary" in text
    assert "TAB: Cash Flow" in text
    assert "IRR" in text
    assert "6.5%" in text


def test_extract_proforma_data_missing_tabs(sample_proforma_xlsx):
    cfg = _base_cfg()
    cfg["proforma"]["tabs"] = ["Not A Tab"]
    with pytest.raises(ValueError, match="No configured tabs found"):
        extract_proforma_data(sample_proforma_xlsx, cfg)


def test_extract_proforma_data_empty_rows(empty_proforma_xlsx):
    with pytest.raises(ValueError, match="no non-empty values"):
        extract_proforma_data(empty_proforma_xlsx, _base_cfg())


def test_extract_memo_content_success(sample_pptx):
    text = extract_memo_content(sample_pptx, _base_cfg())
    assert "PAGE 1" in text
    assert "UnitMixTable" in text
    assert "NarrativeBox" in text


def test_extract_memo_content_invalid_pptx(tmp_dir):
    bad_pptx = os.path.join(tmp_dir, "bad.pptx")
    with open(bad_pptx, "w", encoding="utf-8") as f:
        f.write("not a real pptx")

    with pytest.raises(ValueError, match="Unable to open memo PPTX"):
        extract_memo_content(bad_pptx, _base_cfg())
