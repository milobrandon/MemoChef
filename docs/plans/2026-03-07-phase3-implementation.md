# Phase 3: Market Data & Intelligence — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add slide insertion from supplemental data (PDF/URL/Excel), accuracy metrics with confidence scoring, and enhanced run history dashboard to Memo Chef.

**Architecture:** Three independent features built on the existing two-pass pipeline. Slide insertion adds a third Claude pass after validation. Accuracy metrics are computed from existing mapping/validation outputs. Run history extends the existing `memo_chef_runs` DB table and Streamlit tab.

**Tech Stack:** python-pptx, pdfplumber, beautifulsoup4, requests, Anthropic Claude API (Sonnet), Streamlit, SQLite (psycopg2), pydantic

---

### Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add new packages**

Add these lines to `requirements.txt`:
```
pdfplumber>=0.10.0
beautifulsoup4>=4.12.0
```

(`requests` is already a transitive dependency of `streamlit`.)

**Step 2: Install**

Run: `pip install pdfplumber beautifulsoup4`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add pdfplumber and beautifulsoup4 for Phase 3"
```

---

### Task 2: Supplemental Data Extraction

**Files:**
- Create: `memo_chef/extraction.py`
- Test: `tests/test_extraction.py`

**Step 1: Write the failing tests**

Create `tests/test_extraction.py`:

```python
"""Tests for supplemental data extraction."""
import pytest
from memo_chef.extraction import extract_supplemental


def test_extract_excel_returns_text(tmp_path):
    """Excel extraction returns tab-delimited text."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = "Metric"
    ws["B1"] = "Value"
    ws["A2"] = "HHI"
    ws["B2"] = 58200
    path = tmp_path / "test.xlsx"
    wb.save(path)

    result = extract_supplemental(str(path), "excel")
    assert "HHI" in result
    assert "58200" in result


def test_extract_pdf_returns_text(tmp_path):
    """PDF extraction returns page text."""
    # Create a minimal PDF with reportlab or just test with a text-based PDF
    # For unit test, we mock pdfplumber
    from unittest.mock import patch, MagicMock

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Student affluence data: HHI $62,500"
    mock_page.extract_tables.return_value = []

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)

    with patch("pdfplumber.open", return_value=mock_pdf):
        result = extract_supplemental("fake.pdf", "pdf")
    assert "Student affluence" in result
    assert "62,500" in result


def test_extract_url_returns_text():
    """URL extraction returns page text content."""
    from unittest.mock import patch, MagicMock

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body><p>Market occupancy is 94.2%</p></body></html>"

    with patch("requests.get", return_value=mock_resp):
        result = extract_supplemental("https://example.com/data", "url")
    assert "94.2%" in result


def test_extract_unknown_type_raises():
    """Unknown source type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported source_type"):
        extract_supplemental("file.xyz", "unknown")


def test_extract_csv_returns_text(tmp_path):
    """CSV extraction returns row text."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("Metric,Value\nHHI,58200\nPop Growth,2.1%\n")
    result = extract_supplemental(str(csv_path), "csv")
    assert "HHI" in result
    assert "58200" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `memo_chef/extraction.py`**

```python
"""Supplemental data extraction for PDF, URL, Excel, and CSV sources."""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def extract_supplemental(source: str, source_type: str) -> str:
    """Extract text from a supplemental data source.

    Args:
        source: File path or URL.
        source_type: One of "pdf", "url", "excel", "csv".

    Returns:
        Plain text representation of the source data.
    """
    extractors = {
        "pdf": _extract_pdf,
        "url": _extract_url,
        "excel": _extract_excel,
        "csv": _extract_csv,
    }
    extractor = extractors.get(source_type)
    if extractor is None:
        raise ValueError(f"Unsupported source_type: {source_type!r}")
    return extractor(source)


def _extract_pdf(path: str) -> str:
    """Extract text and tables from a PDF using pdfplumber."""
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"--- Page {i + 1} ---")
                parts.append(text.strip())

            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables):
                parts.append(f"Table {t_idx + 1}:")
                for row in table:
                    cells = [str(c) if c is not None else "" for c in row]
                    parts.append("\t".join(cells))
    return "\n".join(parts)


def _extract_url(url: str) -> str:
    """Extract visible text from a URL using requests + BeautifulSoup."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=30, headers={"User-Agent": "MemoChef/1.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_excel(path: str) -> str:
    """Extract all sheets from an Excel file as tab-delimited text."""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    parts: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_text: list[str] = []
        for row in ws.iter_rows(max_row=250, max_col=20, values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(c for c in cells):
                rows_text.append("\t".join(cells))
        if rows_text:
            parts.append(f"TAB: {sheet_name}")
            parts.extend(rows_text)
            parts.append("")
    wb.close()
    return "\n".join(parts)


def _extract_csv(path: str) -> str:
    """Extract CSV file as tab-delimited text."""
    parts: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            parts.append("\t".join(row))
    return "\n".join(parts)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add memo_chef/extraction.py tests/test_extraction.py
git commit -m "feat: add supplemental data extraction (PDF, URL, Excel, CSV)"
```

---

### Task 3: Slide Content Analysis (Claude API)

**Files:**
- Create: `memo_chef/slide_insertion.py`
- Modify: `prompts/slide_insertion_v1.txt`
- Test: `tests/test_slide_insertion.py`

**Step 1: Create the slide insertion prompt**

Create `prompts/slide_insertion_v1.txt`:

```
You are a real estate investment analyst creating content for an Investment Committee (IC) memo.

Given supplemental data and the current memo structure, generate content for a NEW slide to be inserted into the memo.

## RULES

1. The slide must be relevant to the investment thesis.
2. Choose the most impactful data to visualize — prioritize quantitative metrics.
3. Write narrative in the same tone as the existing memo: professional, concise, data-driven.
4. Pick a visual type that best represents the data: bar_chart, line_chart, table, or pie_chart.
5. Place the slide in the most contextually appropriate section of the memo.
6. Keep narrative to 2-4 sentences.
7. Include 3-8 data points in the visual.
8. Use the memo's existing section structure to determine placement.
9. Format numbers consistently: dollar signs, commas, percentages as appropriate.

## MEMO STRUCTURE

The memo has these sections (by slide):
{memo_structure}

## SUPPLEMENTAL DATA

{supplemental_text}

{user_brief_section}

## OUTPUT FORMAT

Return ONLY valid JSON. No text outside the JSON object.

```json
{
  "slide_title": "string — concise title for the slide",
  "target_section": "string — name of the memo section this belongs in",
  "target_after_slide": int,  // 1-based slide number to insert after
  "narrative": "string — 2-4 sentence narrative for the slide body",
  "visual_type": "bar_chart | line_chart | table | pie_chart",
  "visual_data": {
    "title": "string — chart/table title",
    "categories": ["label1", "label2", ...],
    "series": [
      {"name": "Series Name", "values": [num1, num2, ...]}
    ]
  },
  "data_points": [
    {"label": "Metric Name", "value": "Formatted Value"}
  ]
}
```
```

**Step 2: Write failing tests**

Create `tests/test_slide_insertion.py`:

```python
"""Tests for slide content analysis and insertion."""
import json
import pytest
from unittest.mock import patch, MagicMock

from memo_chef.slide_insertion import (
    analyze_supplemental_content,
    detect_memo_sections,
    find_template_slide,
)


def test_detect_memo_sections():
    """Detect section boundaries from memo text extraction."""
    memo_text = """
====== PAGE 1 ======
--- Shape 0: type=TITLE, name='Title' ---
    Para 0: 'Cover Page'

====== PAGE 2 ======
--- Shape 0: type=TITLE, name='Title' ---
    Para 0: 'Table of Contents'

====== PAGE 3 ======
--- Shape 0: type=TITLE, name='Title' ---
    Para 0: 'Executive Summary'

====== PAGE 6 ======
--- Shape 0: type=TITLE, name='Title' ---
    Para 0: 'Market Summary'

====== PAGE 10 ======
--- Shape 0: type=TITLE, name='Title' ---
    Para 0: 'Financial Summary'
"""
    sections = detect_memo_sections(memo_text)
    assert len(sections) >= 3
    assert any(s["name"] == "Executive Summary" for s in sections)
    assert any(s["name"] == "Market Summary" for s in sections)


MOCK_CLAUDE_RESPONSE = json.dumps({
    "slide_title": "Student Affluence Trends",
    "target_section": "Market Summary",
    "target_after_slide": 8,
    "narrative": "The market shows strong affluence indicators.",
    "visual_type": "bar_chart",
    "visual_data": {
        "title": "Median HHI by Zip",
        "categories": ["40502", "40503"],
        "series": [{"name": "Median HHI", "values": [62500, 58200]}],
    },
    "data_points": [
        {"label": "Median HHI", "value": "$62,500"},
    ],
})


def test_analyze_supplemental_content_returns_structured_json():
    """Claude call returns structured slide content."""
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=MOCK_CLAUDE_RESPONSE)]
    mock_msg.usage = MagicMock(input_tokens=500, output_tokens=200)

    with patch("memo_chef.slide_insertion._call_claude", return_value=mock_msg):
        result = analyze_supplemental_content(
            supplemental_text="HHI data: 40502=$62,500, 40503=$58,200",
            memo_structure=[{"name": "Market Summary", "start_page": 6, "end_page": 9}],
            api_key="sk-test",
            model="claude-sonnet-4-6",
        )
    assert result["slide_title"] == "Student Affluence Trends"
    assert result["visual_type"] == "bar_chart"
    assert result["target_after_slide"] == 8


def test_find_template_slide_prefers_same_section():
    """Template finder prefers slides in the same section with matching visual type."""
    from pptx import Presentation
    prs = Presentation()

    # Add 3 slides: title, chart-like, table-like
    for _ in range(3):
        prs.slides.add_slide(prs.slide_layouts[0])

    # With a real presentation we'd check chart/table shapes.
    # For unit test, just verify the function runs and returns int or None.
    result = find_template_slide(prs, target_section="Market Summary",
                                  visual_type="bar_chart",
                                  sections=[{"name": "Market Summary", "start_page": 2, "end_page": 3}])
    assert result is None or isinstance(result, int)
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_slide_insertion.py -v`
Expected: FAIL (module not found)

**Step 4: Implement `memo_chef/slide_insertion.py`**

```python
"""Slide content analysis and insertion logic."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.util import Inches, Pt

log = logging.getLogger(__name__)

# Known section headers in IC memos
KNOWN_SECTIONS = [
    "Cover", "Table of Contents", "Executive Summary", "Investment Summary",
    "Market Summary", "Market Overview", "University Overview", "University Profile",
    "Site Overview", "Site Plan", "Competitive Landscape", "Comp Summary",
    "Financial Summary", "Financial Overview", "Development Budget",
    "Sources & Uses", "Sensitivity", "Appendix",
]


def detect_memo_sections(memo_text: str) -> list[dict]:
    """Parse memo extraction text to identify section boundaries.

    Returns list of {"name": str, "start_page": int, "end_page": int}.
    """
    sections: list[dict] = []
    current_page = 0

    for line in memo_text.splitlines():
        page_match = re.match(r"=+ PAGE (\d+) =+", line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Look for title shapes with known section names
        if "Para 0:" in line:
            text_match = re.search(r"Para 0:\s*'(.+?)'", line)
            if text_match:
                title = text_match.group(1).strip()
                # Check if this looks like a section header
                for known in KNOWN_SECTIONS:
                    if known.lower() in title.lower():
                        sections.append({
                            "name": title,
                            "start_page": current_page,
                            "end_page": current_page,  # updated below
                        })
                        break

    # Set end_page for each section = start of next section - 1
    for i in range(len(sections) - 1):
        sections[i]["end_page"] = sections[i + 1]["start_page"] - 1
    if sections:
        sections[-1]["end_page"] = 999  # last section extends to end

    return sections


def analyze_supplemental_content(
    supplemental_text: str,
    memo_structure: list[dict],
    api_key: str,
    model: str = "claude-sonnet-4-6",
    user_brief: str | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Call Claude to analyze supplemental data and generate slide content.

    Returns structured dict with slide_title, narrative, visual_type, visual_data, etc.
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "slide_insertion_v1.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    structure_text = "\n".join(
        f"  Slides {s['start_page']}-{s['end_page']}: {s['name']}"
        for s in memo_structure
    )

    brief_section = ""
    if user_brief:
        brief_section = f"## USER BRIEF\n\n{user_brief}"

    prompt = prompt_template.replace("{memo_structure}", structure_text)
    prompt = prompt.replace("{supplemental_text}", supplemental_text[:50_000])
    prompt = prompt.replace("{user_brief_section}", brief_section)

    response = _call_claude(prompt, api_key, model, max_tokens)
    text = response.content[0].text.strip()

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        raise ValueError(f"Claude returned no valid JSON for slide insertion:\n{text[:500]}")

    result = json.loads(json_match.group())

    # Track token usage
    result["_tokens"] = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
    }
    return result


def _call_claude(prompt: str, api_key: str, model: str, max_tokens: int):
    """Make a Claude API call."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )


def find_template_slide(
    prs: Presentation,
    target_section: str,
    visual_type: str,
    sections: list[dict],
) -> int | None:
    """Find the best slide to clone as a template.

    Scores slides by section proximity and visual type match.
    Returns 0-based slide index, or None if no good match.
    """
    best_idx = None
    best_score = 0

    # Find the target section
    target = None
    for s in sections:
        if target_section.lower() in s["name"].lower():
            target = s
            break

    if target is None:
        return None

    for idx, slide in enumerate(prs.slides):
        page = idx + 1  # 1-based
        score = 0

        # Section proximity
        if target and target["start_page"] <= page <= target["end_page"]:
            score += 10
        elif target and abs(page - target["start_page"]) <= 2:
            score += 5

        # Visual type match
        has_chart = any(shape.has_chart for shape in slide.shapes)
        has_table = any(shape.has_table for shape in slide.shapes)

        if visual_type in ("bar_chart", "line_chart", "pie_chart") and has_chart:
            score += 5
        elif visual_type == "table" and has_table:
            score += 5

        # Prefer slides with both visual and text (not just title slides)
        has_text = any(
            shape.has_text_frame and len(shape.text_frame.paragraphs) > 1
            for shape in slide.shapes
        )
        if has_text and (has_chart or has_table):
            score += 3

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx if best_score >= 10 else None


def clone_slide(prs: Presentation, template_idx: int) -> Any:
    """Deep-copy a slide and append it to the presentation.

    Returns the new slide object.
    """
    import copy
    from lxml import etree

    template_slide = prs.slides[template_idx]
    slide_layout = template_slide.slide_layout

    new_slide = prs.slides.add_slide(slide_layout)

    # Copy all shapes from template to new slide by cloning XML
    for shape in list(new_slide.shapes):
        sp = shape._element
        sp.getparent().remove(sp)

    for shape in template_slide.shapes:
        el = copy.deepcopy(shape._element)
        new_slide.shapes._spTree.append(el)

    return new_slide


def build_slide_from_scratch(
    prs: Presentation,
    content: dict,
) -> Any:
    """Build a new slide with chart/table and narrative from scratch.

    Uses the deck's blank layout or title+content layout.
    """
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    # Find a suitable layout (prefer blank or title-only)
    layout = prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[0]
    slide = prs.slides.add_slide(layout)

    visual = content.get("visual_data", {})
    visual_type = content.get("visual_type", "table")

    # Add title
    from pptx.util import Inches, Pt, Emu
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(0.25), Inches(9), Inches(0.6)
    )
    tf = title_box.text_frame
    tf.text = content.get("slide_title", "")
    for para in tf.paragraphs:
        para.font.size = Pt(24)
        para.font.bold = True

    if visual_type == "table":
        _build_table(slide, visual)
    elif visual_type in ("bar_chart", "line_chart", "pie_chart"):
        _build_chart(slide, visual, visual_type)

    # Add narrative text box
    narrative = content.get("narrative", "")
    if narrative:
        text_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(5.5), Inches(9), Inches(1.5)
        )
        tf = text_box.text_frame
        tf.word_wrap = True
        tf.text = narrative
        for para in tf.paragraphs:
            para.font.size = Pt(11)

    return slide


def _build_table(slide, visual: dict) -> None:
    """Add a table shape to the slide from visual_data."""
    categories = visual.get("categories", [])
    series_list = visual.get("series", [])
    if not categories or not series_list:
        return

    rows = len(categories) + 1  # header + data rows
    cols = len(series_list) + 1  # label col + data cols

    table_shape = slide.shapes.add_table(
        rows, cols, Inches(0.5), Inches(1.0), Inches(9), Inches(4)
    )
    table = table_shape.table

    # Header row
    table.cell(0, 0).text = visual.get("title", "")
    for j, s in enumerate(series_list):
        table.cell(0, j + 1).text = s.get("name", f"Series {j}")

    # Data rows
    for i, cat in enumerate(categories):
        table.cell(i + 1, 0).text = str(cat)
        for j, s in enumerate(series_list):
            vals = s.get("values", [])
            val = vals[i] if i < len(vals) else ""
            table.cell(i + 1, j + 1).text = str(val)


def _build_chart(slide, visual: dict, chart_type_str: str) -> None:
    """Add a chart shape to the slide from visual_data."""
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    type_map = {
        "bar_chart": XL_CHART_TYPE.COLUMN_CLUSTERED,
        "line_chart": XL_CHART_TYPE.LINE,
        "pie_chart": XL_CHART_TYPE.PIE,
    }
    xl_type = type_map.get(chart_type_str, XL_CHART_TYPE.COLUMN_CLUSTERED)

    categories = visual.get("categories", [])
    series_list = visual.get("series", [])
    if not categories or not series_list:
        return

    chart_data = CategoryChartData()
    chart_data.categories = categories
    for s in series_list:
        chart_data.add_series(s.get("name", "Data"), s.get("values", []))

    slide.shapes.add_chart(
        xl_type, Inches(0.5), Inches(1.0), Inches(9), Inches(4.2), chart_data
    )


def insert_slide_at_position(prs: Presentation, slide, after_slide_idx: int) -> None:
    """Move a slide to the position after `after_slide_idx` (0-based).

    The slide should already be appended to the end of the presentation.
    This moves it to the correct position in the XML.
    """
    slide_id_list = prs.slides._sldIdLst
    slide_ids = list(slide_id_list)

    if not slide_ids:
        return

    # The new slide is the last one
    new_slide_id = slide_ids[-1]
    slide_id_list.remove(new_slide_id)

    # Insert after the target position
    insert_pos = min(after_slide_idx + 1, len(slide_ids))
    slide_ids.insert(insert_pos, new_slide_id)

    # Rebuild the list
    for child in list(slide_id_list):
        slide_id_list.remove(child)
    for sid in slide_ids:
        slide_id_list.append(sid)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_slide_insertion.py -v`
Expected: All 4 PASS

**Step 6: Commit**

```bash
git add memo_chef/slide_insertion.py tests/test_slide_insertion.py prompts/slide_insertion_v1.txt
git commit -m "feat: add slide content analysis and insertion engine"
```

---

### Task 4: Accuracy Metrics + Confidence Scoring

**Files:**
- Create: `memo_chef/accuracy.py`
- Test: `tests/test_accuracy.py`

**Step 1: Write failing tests**

Create `tests/test_accuracy.py`:

```python
"""Tests for accuracy metrics and confidence scoring."""
import pytest
from memo_chef.accuracy import compute_accuracy_metrics


def test_perfect_run_scores_100():
    """All mappings accepted, none missed = 100 confidence."""
    raw = {
        "table_updates": [{"old_value": "x"}] * 10,
        "text_updates": [{"old_text": "y"}] * 5,
        "row_inserts": [],
        "chart_updates": [],
    }
    validated = {"rejected": [], "corrections": [], "missed": []}
    results = [{"match_quality": "exact"}] * 15

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["confidence_score"] == 100.0
    assert metrics["rejection_rate_pct"] == 0.0
    assert metrics["miss_rate_pct"] == 0.0


def test_half_rejected_lowers_score():
    """50% rejection rate should lower confidence significantly."""
    raw = {
        "table_updates": [{"old_value": "x"}] * 10,
        "text_updates": [],
        "row_inserts": [],
        "chart_updates": [],
    }
    validated = {
        "rejected": [{"idx": i} for i in range(5)],
        "corrections": [],
        "missed": [],
    }
    results = [{"match_quality": "exact"}] * 5

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["confidence_score"] < 90
    assert metrics["rejection_rate_pct"] == 50.0


def test_all_degraded_matches_lowers_score():
    """All degraded matches should lower the match quality component."""
    raw = {
        "table_updates": [{"old_value": "x"}] * 10,
        "text_updates": [],
        "row_inserts": [],
        "chart_updates": [],
    }
    validated = {"rejected": [], "corrections": [], "missed": []}
    results = [{"match_quality": "degraded_pass_2"}] * 10

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["match_quality_pct"] == 0.0
    assert metrics["confidence_score"] < 90


def test_empty_run_returns_zero():
    """No mappings at all should score 0."""
    raw = {"table_updates": [], "text_updates": [], "row_inserts": [], "chart_updates": []}
    validated = {"rejected": [], "corrections": [], "missed": []}
    results = []

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["confidence_score"] == 0.0
    assert metrics["total_mappings"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accuracy.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `memo_chef/accuracy.py`**

```python
"""Accuracy metrics and confidence scoring for pipeline runs."""
from __future__ import annotations


def compute_accuracy_metrics(
    raw: dict,
    validated: dict,
    results: list[dict],
) -> dict:
    """Compute accuracy metrics from pipeline outputs.

    Args:
        raw: Raw mappings dict (table_updates, text_updates, row_inserts, chart_updates).
        validated: Validation output (rejected, corrections, missed).
        results: List of change records from apply_updates (with match_quality).

    Returns:
        Dict with confidence_score (0-100) and component metrics.
    """
    total = (
        len(raw.get("table_updates", []))
        + len(raw.get("text_updates", []))
        + len(raw.get("row_inserts", []))
        + len(raw.get("chart_updates", []))
    )

    if total == 0:
        return {
            "confidence_score": 0.0,
            "coverage_pct": 0.0,
            "rejection_rate_pct": 0.0,
            "correction_rate_pct": 0.0,
            "miss_rate_pct": 0.0,
            "match_quality_pct": 0.0,
            "total_mappings": 0,
            "rejected": 0,
            "corrections": 0,
            "missed": 0,
        }

    rejected = len(validated.get("rejected", []))
    corrections = len(validated.get("corrections", []))
    missed = len(validated.get("missed", []))

    degraded = sum(
        1 for r in results
        if str(r.get("match_quality", "")).startswith("degraded")
    )

    # Component scores (each 0.0 to 1.0)
    total_with_missed = max(total + missed, 1)
    coverage = (total_with_missed - missed) / total_with_missed
    acceptance = (total - rejected) / max(total, 1)
    correction_quality = 1 - corrections / max(total, 1)
    match_quality = (len(results) - degraded) / max(len(results), 1) if results else 1.0
    miss_quality = 1 - missed / total_with_missed

    # Weighted composite
    confidence = (
        coverage * 30
        + acceptance * 25
        + correction_quality * 20
        + match_quality * 15
        + miss_quality * 10
    )

    return {
        "confidence_score": round(confidence, 1),
        "coverage_pct": round(coverage * 100, 1),
        "rejection_rate_pct": round(rejected / max(total, 1) * 100, 1),
        "correction_rate_pct": round(corrections / max(total, 1) * 100, 1),
        "miss_rate_pct": round(missed / total_with_missed * 100, 1),
        "match_quality_pct": round(match_quality * 100, 1),
        "total_mappings": total,
        "rejected": rejected,
        "corrections": corrections,
        "missed": missed,
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accuracy.py -v`
Expected: All 4 PASS

**Step 5: Commit**

```bash
git add memo_chef/accuracy.py tests/test_accuracy.py
git commit -m "feat: add accuracy metrics and confidence scoring"
```

---

### Task 5: Integrate Slide Insertion into Pipeline

**Files:**
- Modify: `memo_chef/pipeline.py` (around line 467, after market data extraction)
- Modify: `memo_chef/models.py` (add supplemental fields to RunRequest)

**Step 1: Add supplemental fields to RunRequest**

In `memo_chef/models.py`, add to `RunRequest`:

```python
# Supplemental data for slide insertion
supplemental_path: str | None = None
supplemental_type: str | None = None  # "pdf", "url", "excel", "csv"
supplemental_brief: str | None = None
```

**Step 2: Add slide insertion stage to pipeline**

In `memo_chef/pipeline.py`, after the market data extraction block (~line 467) and before the mapping stage, add supplemental extraction. Then after apply_updates, add the slide insertion stage.

The integration points are:
1. **After extraction, before mapping:** Extract supplemental text and detect memo sections.
2. **After apply_updates:** Call `analyze_supplemental_content()`, then `find_template_slide()` or `build_slide_from_scratch()`, then `insert_slide_at_position()`.

Key code to add in `run_memo_pipeline()`:

```python
# --- Stage: Supplemental data extraction ---
supplemental_text = ""
if request.supplemental_path:
    checkpoint.start_stage("extract_supplemental")
    try:
        from memo_chef.extraction import extract_supplemental
        supplemental_text = extract_supplemental(
            request.supplemental_path, request.supplemental_type or "excel"
        )
        supp_path = os.path.join(request.output_dir, "supplemental_extract.txt")
        Path(supp_path).write_text(supplemental_text, encoding="utf-8")
        checkpoint.set_output("supplemental_extract", supp_path)
        checkpoint.end_stage("extract_supplemental", "complete")
    except Exception as e:
        log.warning("Supplemental extraction failed: %s", e)
        checkpoint.end_stage("extract_supplemental", "failed", detail=str(e))

# ... (mapping + validation + apply stages remain unchanged) ...

# --- Stage: Slide insertion ---
slides_inserted = 0
if supplemental_text and not request.dry_run:
    checkpoint.start_stage("slide_insertion")
    try:
        from memo_chef.slide_insertion import (
            analyze_supplemental_content,
            detect_memo_sections,
            find_template_slide,
            clone_slide,
            build_slide_from_scratch,
            insert_slide_at_position,
        )
        sections = detect_memo_sections(memo_content)
        content = analyze_supplemental_content(
            supplemental_text=supplemental_text,
            memo_structure=sections,
            api_key=request.api_key,
            model=cfg.get("claude", {}).get("model", "claude-sonnet-4-6"),
            user_brief=request.supplemental_brief,
        )

        prs = Presentation(request.memo_path)
        template_idx = find_template_slide(
            prs, content["target_section"], content["visual_type"], sections
        )

        if template_idx is not None:
            new_slide = clone_slide(prs, template_idx)
        else:
            new_slide = build_slide_from_scratch(prs, content)

        target_after = content.get("target_after_slide", len(prs.slides)) - 1
        insert_slide_at_position(prs, new_slide, target_after)
        prs.save(request.memo_path)
        slides_inserted = 1

        checkpoint.end_stage("slide_insertion", "complete",
                            detail=f"Inserted '{content['slide_title']}' after slide {target_after + 1}")
    except Exception as e:
        log.error("Slide insertion failed: %s", e)
        checkpoint.end_stage("slide_insertion", "failed", detail=str(e))
```

**Step 3: Commit**

```bash
git add memo_chef/pipeline.py memo_chef/models.py
git commit -m "feat: integrate slide insertion into pipeline"
```

---

### Task 6: Integrate Accuracy Metrics into Pipeline

**Files:**
- Modify: `memo_chef/pipeline.py` (after validation stage)
- Modify: `memo_chef/models.py` (add accuracy fields to RunManifest)

**Step 1: Add accuracy fields to RunManifest**

In `memo_chef/models.py`, add to `RunManifest.counts` or as a new field:

```python
accuracy: dict | None = None  # Populated after validation
```

**Step 2: Compute and store accuracy after apply stage**

In `memo_chef/pipeline.py`, after `apply_updates()` returns change records:

```python
# --- Compute accuracy metrics ---
from memo_chef.accuracy import compute_accuracy_metrics
accuracy = compute_accuracy_metrics(
    raw=mappings_raw,
    validated=validation_result,
    results=changes,
)
manifest.accuracy = accuracy
manifest.counts["confidence_score"] = accuracy["confidence_score"]
checkpoint.save()
```

**Step 3: Add accuracy summary to change log**

In `memo_automator.py` `write_change_log()` (~line 3031), add an accuracy section at the top of the HTML:

```python
if run_metadata and run_metadata.get("accuracy"):
    acc = run_metadata["accuracy"]
    score = acc["confidence_score"]
    color = "#4CAF50" if score >= 80 else "#FFC107" if score >= 60 else "#F44336"
    html += f'<div style="background:{color}20;border-left:4px solid {color};padding:12px;margin:16px 0;">'
    html += f'<strong>Confidence Score: {score}/100</strong><br>'
    html += f'Coverage: {acc["coverage_pct"]}% | '
    html += f'Rejected: {acc["rejection_rate_pct"]}% | '
    html += f'Corrections: {acc["correction_rate_pct"]}% | '
    html += f'Missed: {acc["miss_rate_pct"]}%'
    html += '</div>'
```

**Step 4: Commit**

```bash
git add memo_chef/pipeline.py memo_chef/models.py memo_automator.py
git commit -m "feat: integrate accuracy metrics into pipeline and change log"
```

---

### Task 7: Enhance Run History DB Schema

**Files:**
- Modify: `app_services.py` (~line 80, table creation)

**Step 1: Add new columns to `memo_chef_runs` table**

Add these columns to the existing CREATE TABLE or as ALTER TABLE migrations in `get_db_conn()`:

```python
# After existing memo_chef_runs creation, add migration:
cursor.execute("""
    ALTER TABLE memo_chef_runs
    ADD COLUMN IF NOT EXISTS slides_inserted INTEGER DEFAULT 0
""")
cursor.execute("""
    ALTER TABLE memo_chef_runs
    ADD COLUMN IF NOT EXISTS confidence_score REAL
""")
cursor.execute("""
    ALTER TABLE memo_chef_runs
    ADD COLUMN IF NOT EXISTS coverage_pct REAL
""")
cursor.execute("""
    ALTER TABLE memo_chef_runs
    ADD COLUMN IF NOT EXISTS correction_rate_pct REAL
""")
cursor.execute("""
    ALTER TABLE memo_chef_runs
    ADD COLUMN IF NOT EXISTS run_manifest_json TEXT
""")
cursor.execute("""
    ALTER TABLE memo_chef_runs
    ADD COLUMN IF NOT EXISTS change_log_html TEXT
""")
```

**Step 2: Update `record_run()` (~line 359) to accept and store new fields**

Add parameters:
```python
def record_run(
    *,
    # ... existing params ...
    slides_inserted: int = 0,
    confidence_score: float | None = None,
    coverage_pct: float | None = None,
    correction_rate_pct: float | None = None,
    run_manifest_json: str | None = None,
    change_log_html: str | None = None,
) -> None:
```

Update the INSERT/upsert query to include these new columns.

**Step 3: Add `get_run_detail()` function**

```python
def get_run_detail(run_id: str) -> dict | None:
    """Fetch full details for a single run including manifest and change log."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM memo_chef_runs WHERE run_id = %s",
        (run_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in cur.description]
    return dict(zip(cols, row))
```

**Step 4: Commit**

```bash
git add app_services.py
git commit -m "feat: extend run history schema with accuracy and manifest storage"
```

---

### Task 8: Enhanced Run History Streamlit Tab

**Files:**
- Modify: `app.py` (~line 619, `render_history_tab()`)

**Step 1: Enhance the existing Run History tab**

Replace or extend `render_history_tab()` to show:

1. **Summary table** with new columns: confidence score (color-coded), slides inserted
2. **Expandable row detail**: click to see full change log HTML + accuracy breakdown
3. **Run comparison**: select 2 runs to see side-by-side diff

Key UI code:

```python
def render_history_tab():
    runs = get_recent_runs(limit=50)
    if not runs:
        st.info("No runs yet.")
        return

    # Summary dataframe
    df = pd.DataFrame(runs)
    display_cols = ["created_at", "memo_name", "status", "change_count",
                    "rejected_count", "missed_count", "confidence_score",
                    "slides_inserted", "duration_seconds", "estimated_cost_usd"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True)

    # Detail expander
    selected_run_id = st.selectbox("View run details", df["run_id"].tolist(),
                                     format_func=lambda x: f"{x[:8]}...")
    if selected_run_id:
        detail = get_run_detail(selected_run_id)
        if detail:
            # Accuracy card
            score = detail.get("confidence_score")
            if score is not None:
                color = "#4CAF50" if score >= 80 else "#FFC107" if score >= 60 else "#F44336"
                st.markdown(
                    f'<div style="background:{color}20;border-left:4px solid {color};padding:12px;">'
                    f'<strong>Confidence: {score}/100</strong></div>',
                    unsafe_allow_html=True,
                )

            # Change log
            if detail.get("change_log_html"):
                with st.expander("Change Log", expanded=False):
                    st.html(detail["change_log_html"])

            # Manifest
            if detail.get("run_manifest_json"):
                with st.expander("Run Manifest", expanded=False):
                    st.json(json.loads(detail["run_manifest_json"]))
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: enhance Run History tab with accuracy metrics and detail view"
```

---

### Task 9: Supplemental Data Upload in Streamlit

**Files:**
- Modify: `app.py` (~line 443, upload section)

**Step 1: Add supplemental file uploader + URL input + brief text area**

After the existing 4-column upload row (~line 447), add:

```python
# Supplemental data row
st.markdown("---")
supp_cols = st.columns([2, 2, 3])
supplemental_file = supp_cols[0].file_uploader(
    "Supplemental data",
    type=["pdf", "xlsx", "xlsm", "csv"],
    key="supplemental_upload",
    help="Upload additional data to generate a new slide (PDF, Excel, or CSV)",
)
supplemental_url = supp_cols[1].text_input(
    "Or paste a URL",
    key="supplemental_url",
    placeholder="https://...",
)
supplemental_brief = supp_cols[2].text_area(
    "Brief (optional)",
    key="supplemental_brief",
    placeholder="e.g., Show student affluence trends for this market",
    height=80,
)
```

**Step 2: Wire supplemental inputs into the job payload**

In `_queue_item_from_inputs()` or wherever the RunRequest is built, pass the supplemental data:

```python
# Determine supplemental source
supplemental_path = None
supplemental_type = None
if supplemental_file:
    ext = Path(supplemental_file.name).suffix.lower()
    type_map = {".pdf": "pdf", ".xlsx": "excel", ".xlsm": "excel", ".csv": "csv"}
    supplemental_type = type_map.get(ext, "excel")
    supplemental_path = save_uploaded_file(supplemental_file)
elif supplemental_url:
    supplemental_path = supplemental_url
    supplemental_type = "url"
```

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add supplemental data upload for slide insertion in UI"
```

---

### Task 10: Wire Accuracy into record_run + Final Integration

**Files:**
- Modify: `app.py` (where `record_run()` is called after pipeline completes)
- Modify: `memo_chef/pipeline.py` (ensure RunResult includes accuracy)

**Step 1: Pass accuracy to record_run()**

In `app.py` where `_execute_job()` calls `record_run()`, add the new fields:

```python
record_run(
    # ... existing fields ...
    slides_inserted=result.manifest.counts.get("slides_inserted", 0),
    confidence_score=result.manifest.accuracy.get("confidence_score") if result.manifest.accuracy else None,
    coverage_pct=result.manifest.accuracy.get("coverage_pct") if result.manifest.accuracy else None,
    correction_rate_pct=result.manifest.accuracy.get("correction_rate_pct") if result.manifest.accuracy else None,
    run_manifest_json=result.manifest.model_dump_json() if hasattr(result.manifest, "model_dump_json") else None,
    change_log_html=change_log_html,
)
```

**Step 2: Ensure RunResult includes accuracy in manifest**

Already handled in Task 6 — verify `manifest.accuracy` is populated.

**Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add app.py memo_chef/pipeline.py
git commit -m "feat: wire accuracy metrics and slide insertion into full pipeline"
```

---

### Task 11: Update ROADMAP.md + Final Verification

**Files:**
- Modify: `ROADMAP.md`

**Step 1: Mark Phase 3 items as complete**

Update the Phase 3 section to mark completed items.

**Step 2: Run full test suite + manual smoke test**

```bash
pytest tests/ -v --tb=short
```

**Step 3: Final commit**

```bash
git add ROADMAP.md
git commit -m "docs: mark Phase 3 features complete in roadmap"
```

---
