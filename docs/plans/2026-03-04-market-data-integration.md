# Market Data Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional market data Excel upload to Memo Chef that extracts dashboard tabs and chart metadata, appends to proforma data for the mapping pass, and supports updating embedded PowerPoint charts.

**Architecture:** New `extract_market_data()` function mirrors `extract_proforma_data()` pattern. Enhanced `extract_memo_content()` adds chart metadata extraction. New `chart_updates` type in mapping schema handled by `_apply_chart_updates()`. Market data text appended to `proforma_data` before mapping. Formatting-preservation bug in `_replace_in_para()` investigated and fixed.

**Tech Stack:** openpyxl (Excel reading), python-pptx (chart extraction + update), anthropic (Claude API)

---

### Task 1: Add `extract_market_data()` to `memo_automator.py`

**Files:**
- Modify: `memo_automator.py:369` (after `extract_proforma_data()`, before schedule extraction section)

**Step 1: Write the extraction function**

Insert after line 369 (after `extract_proforma_data` ends, before the schedule section comment):

```python
# ============================================================================
# 4a. MARKET DATA EXTRACTION  (openpyxl, data_only=True)
# ============================================================================
_MARKET_DASHBOARD_TABS = [
    "Tables",
    "Comparison Graph",
    "Uncaptured Demand Comparison",
    "Rent Growth Comparison By Year",
    "Occupancy Comparison By Year",
    "Comp Set",
]


def extract_market_data(market_data_path: str, cfg: dict) -> str:
    """
    Read the RealPage market data workbook and return a compact text
    representation of the 6 dashboard tabs (ignoring back-end data tabs).

    Uses data_only=True so formulas resolve to their cached values.
    Returns empty string if no dashboard tabs are found (non-fatal).
    """
    max_rows = cfg["proforma"]["max_rows_per_tab"]
    max_cols = cfg["proforma"]["max_cols_per_tab"]

    log.info("Opening market data (data_only): %s", market_data_path)
    try:
        wb = openpyxl.load_workbook(market_data_path, data_only=True)
    except (InvalidFileException, zipfile.BadZipFile) as e:
        log.warning(
            "Unable to open market data '%s': %s. Continuing without market data.",
            market_data_path, e,
        )
        return ""
    log.info("Market data sheets: %s", wb.sheetnames)

    lines = [
        f"\n{'='*70}",
        "MARKET DATA (from RealPage)",
        f"{'='*70}",
    ]
    found_tabs = 0
    data_rows = 0
    for tab_name in _MARKET_DASHBOARD_TABS:
        if tab_name not in wb.sheetnames:
            log.warning("Market data tab '%s' not found - skipping", tab_name)
            continue
        found_tabs += 1
        ws = wb[tab_name]
        lines.append(f"\n{'='*70}")
        lines.append(f"TAB: {tab_name}")
        lines.append(f"{'='*70}")

        end_row = ws.max_row if max_rows == 0 else min(ws.max_row, max_rows)
        end_col = ws.max_column if max_cols == 0 else min(ws.max_column, max_cols)

        for row in ws.iter_rows(
            min_row=1, max_row=end_row, max_col=end_col, values_only=False
        ):
            row_data = []
            for cell in row:
                if cell.value is not None:
                    row_data.append(str(cell.value))
            if row_data:
                lines.append(f"Row {row[0].row}:\t" + "\t".join(row_data))
                data_rows += 1

    wb.close()

    if found_tabs == 0:
        log.warning(
            "No dashboard tabs found in market data file. "
            "Expected tabs: %s. Available sheets: %s. Skipping market data.",
            _MARKET_DASHBOARD_TABS, wb.sheetnames,
        )
        return ""

    if data_rows == 0:
        log.warning(
            "Market data extraction found no non-empty values. "
            "If this workbook contains formulas, open it in Excel, let it "
            "recalculate, save, and retry."
        )
        return ""

    market_text = "\n".join(lines)
    log.info(
        "Market data extraction complete (%d tabs, %d lines, %d chars)",
        found_tabs, len(lines), len(market_text),
    )
    return market_text
```

**Step 2: Add to module exports**

Find the imports at the top of `app.py` (line ~19-30) where functions are imported from `memo_automator`. Add `extract_market_data` to the import list.

**Step 3: Run a quick smoke test**

```bash
cd "C:/Users/BrandonZmuda/Desktop/Claude/g. Memo Automator/v2"
python3 -c "
from memo_automator import extract_market_data, load_config
cfg = load_config('config.yaml')
text = extract_market_data('C:/Users/BrandonZmuda/Desktop/Claude/g. Memo Automator/a. Sandbox/New Template Test.xlsx', cfg)
print(f'Extracted {len(text)} chars')
print(text[:500])
"
```

Expected: Prints ~500 chars of extracted market data starting with the header.

**Step 4: Commit**

```bash
git add memo_automator.py
git commit -m "feat: add extract_market_data() for RealPage dashboard extraction"
```

---

### Task 2: Add Chart Extraction to `extract_memo_content()`

**Files:**
- Modify: `memo_automator.py:512-537` (inside the shape iteration loop in `extract_memo_content()`)

**Step 1: Add chart metadata extraction**

After the table extraction block (line ~537, after the `if shape.has_table:` block), add chart extraction:

```python
            # Charts (embedded Excel chart objects)
            if shape.has_chart:
                chart = shape.chart
                chart_type_name = str(chart.chart_type) if chart.chart_type else "UNKNOWN"
                lines.append(f"    Chart type: {chart_type_name}")

                # Chart title
                if chart.has_title and chart.chart_title and chart.chart_title.has_text_frame:
                    title_text = chart.chart_title.text_frame.text.strip()
                    lines.append(f"    Chart title: '{title_text}'")

                # Extract series and data
                try:
                    for s_idx, series in enumerate(chart.series):
                        s_name = ""
                        try:
                            s_name = series.tx.strRef.strCache.pt[0].v if series.tx and series.tx.strRef else f"Series {s_idx}"
                        except (AttributeError, IndexError):
                            s_name = f"Series {s_idx}"

                        # Extract series values
                        vals = []
                        try:
                            if series.values:
                                vals = [v for v in series.values if v is not None]
                        except Exception:
                            pass

                        lines.append(f"    Series {s_idx} ('{s_name}'): {vals[:20]}")
                except Exception as e:
                    lines.append(f"    (chart data extraction failed: {e})")

                # Extract category labels (x-axis)
                try:
                    plot = chart.plots[0]
                    if plot.categories:
                        cats = list(plot.categories)[:30]
                        lines.append(f"    Categories: {cats}")
                except Exception:
                    pass
```

**Step 2: Run smoke test**

Create a small test .pptx with an embedded chart or use an existing memo to verify chart shapes are detected.

```bash
python3 -c "
from memo_automator import extract_memo_content, load_config
cfg = load_config('config.yaml')
# Test with a real memo that has charts
# text = extract_memo_content('path/to/memo-with-charts.pptx', cfg)
# Verify 'Chart type:' appears in output
print('Chart extraction code added - needs a real memo to verify')
"
```

**Step 3: Commit**

```bash
git add memo_automator.py
git commit -m "feat: extract chart metadata (type, title, series, categories) from memo"
```

---

### Task 3: Add `chart_updates` to Mapping Schema and Prompt

**Files:**
- Modify: `prompts/mapping_v1.txt:148-196` (schema section at end of prompt)
- Modify: `memo_automator.py:585-783` (inline MAPPING_PROMPT fallback)

**Step 1: Add chart_updates to prompt schema**

In `prompts/mapping_v1.txt`, before the closing `}}` of the schema (after `row_inserts`), add:

```
  "chart_updates": [
    {{
      "page": <int>,
      "chart_name": "<shape name from memo>",
      "chart_title": "<chart title text, if visible>",
      "series_name": "<name of series to update>",
      "old_values": [<current values as numbers>],
      "new_values": [<replacement values from market data>],
      "categories": [<category labels if changed, else omit>],
      "source": "<market data tab name + cell reference>"
    }}
  ]
```

**Step 2: Add market data + chart reasoning instructions**

Append before the "Important:" section at the end of `prompts/mapping_v1.txt`:

```
17. **Market data & chart updates** — When MARKET DATA sections are present
    and the memo contains embedded charts (type=CHART shapes):
    a. Match market data tabs to memo charts by semantic similarity of chart
       titles, series names, and axis labels. Names may differ slightly
       (e.g. "Florida" in data ↔ "UF" in chart title).
    b. For each matched chart, emit a chart_update entry per series with the
       updated values from market data.
    c. Preserve all existing series — do not add or remove series unless the
       market data clearly warrants it.
    d. If a chart's year range or comparison markets differ from the data,
       adapt the data to fit the chart's existing structure.
    e. Only update values that actually differ from the current chart data.
    f. If a match is uncertain, include it but add "(uncertain match)" to
       the source field so the validator can flag it.
```

**Step 3: Mirror changes in the inline MAPPING_PROMPT fallback**

Update the inline `MAPPING_PROMPT` string in `memo_automator.py` (lines 585-783) to include the same `chart_updates` schema and instruction #17.

**Step 4: Commit**

```bash
git add prompts/mapping_v1.txt memo_automator.py
git commit -m "feat: add chart_updates schema and market data reasoning to mapping prompt"
```

---

### Task 4: Add `_apply_chart_updates()` to `memo_automator.py`

**Files:**
- Modify: `memo_automator.py:1970` (after `apply_updates()` ends, before branding section)

**Step 1: Write the chart update function**

```python
def _apply_chart_updates(memo_path: str, chart_updates: list, dry_run: bool = False) -> list:
    """
    Update embedded PowerPoint chart data based on chart_updates from the
    mapping pass. Preserves all visual formatting (colors, styles, etc.).

    Returns a list of change records.
    """
    from pptx.chart.data import CategoryChartData, XyChartData
    from copy import deepcopy

    if not chart_updates:
        return []

    prs = _load_presentation(memo_path)
    changes = []

    for upd in chart_updates:
        page = upd["page"]
        chart_name = upd.get("chart_name", "")
        chart_title = upd.get("chart_title", "")
        series_name = upd.get("series_name", "")
        new_values = upd.get("new_values", [])
        old_values = upd.get("old_values", [])
        new_categories = upd.get("categories", None)
        source = upd.get("source", "")

        try:
            slide = prs.slides[page - 1]
        except IndexError:
            log.warning("Chart update SKIPPED: page %d does not exist", page)
            continue

        # Find the target chart
        target_chart = None
        target_shape = None
        for shape in slide.shapes:
            if not shape.has_chart:
                continue
            # Match by shape name or chart title
            name_match = chart_name and _loose_match(chart_name, shape.name)
            title_match = False
            if shape.chart.has_title and shape.chart.chart_title:
                try:
                    ct_text = shape.chart.chart_title.text_frame.text.strip()
                    title_match = chart_title and _loose_match(chart_title, ct_text)
                except Exception:
                    pass
            if name_match or title_match:
                target_chart = shape.chart
                target_shape = shape
                break

        if target_chart is None:
            # Fallback: if only one chart on the page, use it
            chart_shapes = [s for s in slide.shapes if s.has_chart]
            if len(chart_shapes) == 1:
                target_chart = chart_shapes[0].chart
                target_shape = chart_shapes[0]
                log.info("Chart update: single chart fallback on page %d", page)
            else:
                log.warning(
                    "Chart update NOT FOUND: page %d, name='%s', title='%s'",
                    page, chart_name, chart_title,
                )
                continue

        # Find and update the target series
        found_series = False
        for series in target_chart.series:
            s_name = ""
            try:
                s_name = series.tx.strRef.strCache.pt[0].v if series.tx and series.tx.strRef else ""
            except (AttributeError, IndexError):
                pass
            if not _loose_match(series_name, s_name):
                continue

            found_series = True
            if dry_run:
                changes.append({
                    "page": page, "type": "chart",
                    "location": f"{target_shape.name} / series '{s_name}'",
                    "old": str(old_values[:5]) + ("..." if len(old_values) > 5 else ""),
                    "new": str(new_values[:5]) + ("..." if len(new_values) > 5 else ""),
                    "source": source,
                })
                break

            # Update series values via the underlying XML
            try:
                # Access the numeric reference cache
                num_ref = series.val
                if num_ref is not None and hasattr(num_ref, 'numRef') and num_ref.numRef is not None:
                    cache = num_ref.numRef.numCache
                    pts = list(cache.pt)
                    for i, pt in enumerate(pts):
                        if i < len(new_values):
                            pt.v = str(new_values[i])
                    changes.append({
                        "page": page, "type": "chart",
                        "location": f"{target_shape.name} / series '{s_name}'",
                        "old": str(old_values[:5]) + ("..." if len(old_values) > 5 else ""),
                        "new": str(new_values[:5]) + ("..." if len(new_values) > 5 else ""),
                        "source": source,
                    })
                else:
                    log.warning("Chart series '%s' has no numeric reference cache", s_name)
            except Exception as e:
                log.warning("Chart update FAILED for series '%s': %s", s_name, e)
            break

        if not found_series:
            log.warning(
                "Chart series '%s' NOT FOUND in chart on page %d",
                series_name, page,
            )

    if not dry_run and changes:
        prs.save(memo_path)
        log.info("Chart updates saved: %d changes.", len(changes))

    return changes
```

**Step 2: Commit**

```bash
git add memo_automator.py
git commit -m "feat: add _apply_chart_updates() for embedded PowerPoint chart data"
```

---

### Task 5: Integrate into `apply_updates()` and Wire Chart Updates

**Files:**
- Modify: `memo_automator.py:1964-1970` (end of `apply_updates()`)

**Step 1: Call `_apply_chart_updates()` from `apply_updates()`**

After the row_inserts section and before the save, add chart update handling. Replace the save block at lines 1964-1970:

```python
    # --- Chart updates ---
    chart_changes = _apply_chart_updates(
        memo_path, mappings.get("chart_updates", []), dry_run=dry_run
    )
    changes.extend(chart_changes)

    if not dry_run:
        # Note: _apply_chart_updates already saves if it made changes,
        # but we need to save for table/text/row changes too
        if changes and not chart_changes:
            prs.save(memo_path)
        log.info("Memo saved with %d updates.", len(changes))
    else:
        log.info("Dry-run: %d updates identified (not saved).", len(changes))

    return changes
```

Wait — `_apply_chart_updates` loads and saves the presentation separately. We need to restructure so the table/text/row updates save first, then chart updates run on the saved file. Actually the current code already saves `prs` for table/text/row. Then `_apply_chart_updates` opens a fresh copy and saves again. This two-pass approach is correct — table/text/row changes saved first, then chart changes applied on top.

So the actual change is simpler — just add the chart_updates call AFTER the existing save:

At the end of `apply_updates()`, after `prs.save(memo_path)` (line 1965), before the return:

```python
    # --- Chart updates (separate pass - charts need fresh prs load) ---
    chart_updates_list = mappings.get("chart_updates", [])
    if chart_updates_list:
        chart_changes = _apply_chart_updates(memo_path, chart_updates_list, dry_run=dry_run)
        changes.extend(chart_changes)
```

**Step 2: Commit**

```bash
git add memo_automator.py
git commit -m "feat: wire chart_updates into apply_updates pipeline"
```

---

### Task 6: Add Market Data Uploader to Streamlit UI

**Files:**
- Modify: `app.py:348-351` (file uploaders section)

**Step 1: Add 4th column and market data uploader**

Replace lines 348-351:

```python
col1, col2, col3, col4 = st.columns(4)
memo_file = col1.file_uploader("The Memo (.pptx)", type=["pptx"])
proforma_file = col2.file_uploader("The Proforma (.xlsx / .xlsm)", type=["xlsx", "xlsm"])
schedule_file = col3.file_uploader("The Schedule (.mpp)", type=["mpp"])
market_data_file = col4.file_uploader("Market Data (.xlsx)", type=["xlsx", "xlsm"])
```

**Step 2: Add import of `extract_market_data` at top of `app.py`**

In the import block from `memo_automator` (around line 19-30), add `extract_market_data` to the import list.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add market data file uploader to Streamlit UI"
```

---

### Task 7: Wire Market Data into Pipeline Flow

**Files:**
- Modify: `app.py:449-463` (between schedule extraction and memo extraction)

**Step 1: Add market data extraction step**

After the schedule extraction block (around line 453) and before the memo extraction comment, add:

```python
            # Step b3: Extract market data (optional)
            if market_data_file:
                progress_bar.progress(13, text="\U0001f4ca Reading the market data...")
                st.write("\U0001f4ca Reading the market data...")
                market_data_path = os.path.join(tmpdir, market_data_file.name)
                with open(market_data_path, "wb") as f:
                    f.write(market_data_file.getvalue())
                try:
                    market_text = extract_market_data(market_data_path, cfg)
                    if market_text:
                        proforma_data += "\n\n" + market_text
                        st.write(f"\U0001f4ca Market data extracted ({len(market_text):,} chars)")
                    else:
                        st.warning("No dashboard tabs found in market data file — continuing without it.")
                except Exception as e:
                    st.warning(f"Could not read market data: {e}. Continuing without it.")
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: wire market data extraction into pipeline flow"
```

---

### Task 8: Investigate and Fix Title Formatting Bug

**Files:**
- Modify: `memo_automator.py:1557-1586` (`_replace_in_para()`)
- Modify: `memo_automator.py:2062-2084` (`apply_branding()`)

**Step 1: Fix `_replace_in_para()` cross-run replacement**

The current Pass 2 (cross-run replacement, lines 1576-1583) dumps all text into `runs[0]` and clears subsequent runs. This destroys formatting when the old_text spans runs with DIFFERENT formatting (e.g., "**bold** normal" → all becomes run[0]'s format).

Replace the cross-run replacement (lines 1576-1584) with a format-preserving version:

```python
    # Pass 2: cross-run replacement (format-preserving)
    full_text = "".join(r.text for r in para.runs)
    if old_text not in full_text:
        return False

    new_full = full_text.replace(old_text, new_text, 1)

    # Find the runs that contain the old_text span
    if para.runs:
        # Calculate character positions for each run
        run_starts = []
        pos = 0
        for run in para.runs:
            run_starts.append(pos)
            pos += len(run.text)

        # Find where old_text starts and ends
        match_start = full_text.index(old_text)
        match_end = match_start + len(old_text)

        # Find runs that overlap with the match
        first_run_idx = None
        last_run_idx = None
        for i, run in enumerate(para.runs):
            run_end = run_starts[i] + len(run.text)
            if first_run_idx is None and run_end > match_start:
                first_run_idx = i
            if run_starts[i] < match_end:
                last_run_idx = i

        if first_run_idx is not None and last_run_idx is not None:
            # Preserve text before match in first run
            pre = full_text[run_starts[first_run_idx]:match_start]
            # Preserve text after match in last run
            last_run_end = run_starts[last_run_idx] + len(para.runs[last_run_idx].text)
            post = full_text[match_end:last_run_end]
            # Write replacement into first overlapping run, clear the rest
            para.runs[first_run_idx].text = pre + new_text + post
            for i in range(first_run_idx + 1, last_run_idx + 1):
                para.runs[i].text = ""
            return True

    return False
```

**Step 2: Fix `apply_branding()` to preserve existing bold/italic**

In `_reformat_run()` (line 2091-2115), the function unconditionally sets the font name for ALL runs. The issue is that it does NOT preserve the `bold` attribute — when the theme XML is replaced, runs that were bold via theme inheritance lose that inheritance, but `_reformat_run` doesn't restore it.

Add bold preservation to `_reformat_run()`:

```python
def _reformat_run(run, is_heading_context: bool, size_threshold: int,
                  heading_font: str, body_font: str, color_threshold: float):
    """Reformat a single text run's font and color, preserving bold/italic."""
    # Snapshot existing formatting BEFORE changes
    was_bold = run.font.bold
    was_italic = run.font.italic

    # Determine if this run is a heading
    font_size = run.font.size
    if font_size is not None:
        size_pt = font_size.pt
    else:
        size_pt = 0

    is_heading = is_heading_context or size_pt >= size_threshold

    # Set font family
    run.font.name = heading_font if is_heading else body_font

    # Restore bold/italic (they may have been inherited from theme,
    # which we just replaced)
    if was_bold is not None:
        run.font.bold = was_bold
    if was_italic is not None:
        run.font.italic = was_italic

    # Remap color if it's a hard-coded RGB
    try:
        color = run.font.color
        if color.type is not None and color.rgb is not None:
            r, g, b = color.rgb[0], color.rgb[1], color.rgb[2]
            nearest = _nearest_brand_color(r, g, b, color_threshold)
            if nearest is not None:
                run.font.color.rgb = nearest
    except (AttributeError, TypeError):
        pass  # No color set or theme color - skip
```

**Step 3: Commit**

```bash
git add memo_automator.py
git commit -m "fix: preserve formatting in cross-run replacement and branding pass"
```

---

### Task 9: Add Market Data Progress Slogans

**Files:**
- Modify: `app.py` (inside `_PROGRESS_SLOGANS` dict)

**Step 1: Add market data slogans**

In the `_PROGRESS_SLOGANS` dict (added earlier in this session), add a new key:

```python
            "market": [
                "\U0001f4ca Reading the market data...",
                "\U0001f4ca Checking the comps...",
                "\U0001f4ca Scanning the rent rolls...",
                "\U0001f4ca Pulling the market pulse...",
            ],
```

**Step 2: Use slogan in the market data extraction step**

In the market data extraction code added in Task 7, replace the hardcoded text:

```python
                progress_bar.progress(13, text=_slogan("market"))
```

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add market data progress slogans"
```

---

### Task 10: End-to-End Testing

**Files:**
- Create: `tests/test_market_data.py`

**Step 1: Write unit tests**

```python
"""Tests for market data extraction and chart update functionality."""

import os
import tempfile
import pytest
import openpyxl
from unittest.mock import MagicMock, patch

# Adjust import path as needed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memo_automator import (
    extract_market_data,
    extract_memo_content,
    load_config,
    _MARKET_DASHBOARD_TABS,
)


@pytest.fixture
def default_cfg():
    return load_config(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    )


@pytest.fixture
def market_data_workbook(tmp_path):
    """Create a minimal market data workbook with dashboard tabs."""
    wb = openpyxl.Workbook()
    # Rename default sheet to first dashboard tab
    ws = wb.active
    ws.title = "Tables"
    ws["A1"] = "IPEDS"
    ws["B1"] = 134130
    ws["A2"] = "University Name"
    ws["B2"] = "UF"

    # Add remaining dashboard tabs
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
        assert "PROPERTIES" not in result
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
        """Only some dashboard tabs exist — extracts what's available."""
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


class TestRealMarketDataFile:
    """Integration test with the actual template file."""

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
        # Check some known values from the template
        assert "UF" in result or "University of Florida" in result
```

**Step 2: Run tests**

```bash
cd "C:/Users/BrandonZmuda/Desktop/Claude/g. Memo Automator/v2"
python3 -m pytest tests/test_market_data.py -v
```

Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/test_market_data.py
git commit -m "test: add unit and integration tests for market data extraction"
```

---

### Task 11: Stress Testing and Edge Cases

**Step 1: Test with real template file through full pipeline**

```bash
python3 -c "
from memo_automator import extract_market_data, load_config
cfg = load_config('config.yaml')
text = extract_market_data(
    'C:/Users/BrandonZmuda/Desktop/Claude/g. Memo Automator/a. Sandbox/New Template Test.xlsx',
    cfg
)
print(f'Total chars: {len(text):,}')
print(f'Total lines: {text.count(chr(10)):,}')
# Check it doesn't blow up the prompt size
proforma_size = 50000  # typical proforma
combined = proforma_size + len(text)
print(f'Combined with typical proforma: {combined:,} chars')
print(f'Would trigger batching (>80K): {combined > 80000}')
"
```

**Step 2: Verify formatting preservation**

Manually test by running the full app with a memo + proforma + market data:
- Check that title formatting is preserved (bold stays bold, colors stay correct)
- Check that chart data is updated if charts exist
- Check that table values update correctly
- Verify the change log includes chart updates

**Step 3: Test edge cases**

- Upload market data file but no charts in memo → should work fine, just table/text updates
- Upload corrupted .xlsx → should warn and continue
- Upload file with only some dashboard tabs → should extract available ones
- Very large market data (>50K chars extracted) → verify batching kicks in correctly

**Step 4: Final commit**

```bash
git add -A
git commit -m "test: stress testing and edge case verification complete"
```

---

### Task Summary

| Task | Component | Estimated Effort |
|------|-----------|-----------------|
| 1 | `extract_market_data()` function | Core extraction |
| 2 | Chart metadata extraction in `extract_memo_content()` | Chart visibility |
| 3 | Prompt schema + reasoning instructions | Claude guidance |
| 4 | `_apply_chart_updates()` function | Chart write-back |
| 5 | Wire chart updates into `apply_updates()` | Pipeline hookup |
| 6 | Streamlit UI 4th uploader | User interface |
| 7 | Pipeline integration (market data → proforma_data) | Data flow |
| 8 | Fix formatting bugs (_replace_in_para + branding) | Bug fix |
| 9 | Market data progress slogans | UX polish |
| 10 | Unit + integration tests | Quality |
| 11 | Stress testing + edge cases | Robustness |
