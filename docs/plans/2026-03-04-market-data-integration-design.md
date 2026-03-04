# Market Data Integration — Design Document

**Date:** 2026-03-04
**Status:** Approved

## Problem

IC memos contain market data dashboards (charts, tables) sourced from RealPage
Excel templates. Currently, these must be updated manually. The Memo Chef should
accept an optional market data `.xlsx` upload and use the dashboard sheets as
additional source data for the mapping pass — including updates to embedded
PowerPoint charts.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration approach | Append to proforma_data | Simplest, lowest risk, reuses entire pipeline |
| Tab selection | Hardcoded 6 dashboard tabs | User preference; avoids config complexity |
| Required vs optional | Optional (like schedule) | Not every memo has market data pages |
| Chart handling | Claude reasons about fuzzy matching | Charts won't always be 1:1 with data tabs |

## Template Structure

The RealPage market data Excel has 15 sheets. We extract only the 6 dashboard
sheets:

1. **Tables** — Supply/demand summary by year, rent growth/occupancy at
   Market/1-Mile/Half-Mile radii (36 rows × 31 cols)
2. **Comparison Graph** — Snapshot rent growth vs occupancy comparison
   (21 rows × 21 cols)
3. **Uncaptured Demand Comparison** — Target market vs comparable markets with
   Power 4 and Subtext 30 benchmarks (106 rows × 39 cols)
4. **Rent Growth Comparison By Year** — YoY rent growth across Market/1-Mile/
   Half-Mile + comp set selections (61 rows × 42 cols)
5. **Occupancy Comparison By Year** — YoY occupancy across Market/1-Mile/
   Half-Mile + comp set selections (61 rows × 42 cols)
6. **Comp Set** — Individual comp property details: name, year built, units,
   beds, rent/bed, YoY rent growth, occupancy, prelease (61 rows × 24 cols)

Ignored: Supply and Demand (5,863 rows), YoY RG (16,261 rows),
Occupancy (16,261 rows), IPEDS (82,554 rows), AXIO Crosswalk,
RPProperties (16,668 rows), University List, Power 4 & Subtext 30,
PROPERTIES (32,275 rows).

## Architecture

### 1. Extraction — `extract_market_data()`

New function in `memo_automator.py`. Same pattern as `extract_proforma_data()`:

```
MARKET DATA (from RealPage)
======================================================================
TAB: Tables
======================================================================
Row 1: IPEDS    134130
Row 2: University Name    UF
...

======================================================================
TAB: Comp Set
======================================================================
Row 4: Project Name    Year Built    # of Units    # of Beds    ...
Row 5: Archer Court Gainesville    1977    72    84    895    ...
```

- Opens workbook with `openpyxl`, `data_only=True`
- Hardcoded tab list; warns if a tab is missing; skips missing tabs
- Returns empty string if no dashboard tabs found (non-fatal)
- Respects existing `max_rows_per_tab` / `max_cols_per_tab` limits from config

### 2. Chart Extraction — Memo Content Enhancement

Enhance `extract_memo_content()` to also extract embedded chart metadata from
PowerPoint slides:

- Chart title
- Chart type (bar, line, pie, etc.)
- Series names and data points
- Category (axis) labels

This gives Claude visibility into what charts exist on each memo page, so it can
match market data to the right chart even when names don't align exactly.

Output format appended per chart shape:
```
--- Shape 5: type=CHART, name='Chart 1' ---
    Position: left=..., top=..., width=..., height=...
    Chart type: LINE
    Title: "Rent Growth Comparison"
    Series: ["UF", "Power 4", "Subtext 30"]
    Categories: [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    Data:
      UF: [0.0512, 0.0583, 0.0076, 0.0335, 0.0901, 0.0693, 0.0161, 0.0404]
      Power 4: [0.0156, 0.0205, -0.0037, 0.0396, 0.0877, 0.0470, 0.0250, 0.0300]
```

### 3. Chart Updates — New Update Type

Add a `chart_updates` array to the mapping JSON schema:

```json
{
  "chart_updates": [
    {
      "page": 7,
      "chart_name": "Chart 1",
      "chart_title": "Rent Growth Comparison",
      "series_name": "UF",
      "old_values": [0.0512, 0.0583, ...],
      "new_values": [0.0429, 0.0375, ...],
      "source": "Market Data / Rent Growth Comparison By Year / Q column"
    }
  ]
}
```

Claude must **reason about which chart matches which data**:
- Match by semantic similarity of chart title, series names, axis labels
- Not require exact name matches (e.g., "Florida" in data ↔ "UF" in chart)
- Approve new/updated data series that are reasonable for the chart's purpose
- Flag mismatches that seem unreasonable for human review

### 4. Apply Chart Updates — `_apply_chart_updates()`

New helper in `memo_automator.py` that uses `python-pptx`'s chart API:

```python
chart = shape.chart
for series in chart.series:
    if series.name == target_series:
        # Update series values via the underlying Excel data
        chart_data = CategoryChartData()
        # Rebuild chart data with updated values
```

Uses `python-pptx` `ChartData` objects to update embedded chart data while
preserving all visual formatting (colors, line styles, etc.).

### 5. Formatting Preservation — CRITICAL

**Existing bug:** Title formatting getting corrupted (random bolding, color
changes) during runs. Root cause to investigate in `apply_branding()` and
`_replace_in_para()`.

**Rules for this integration:**
- `extract_market_data()` is read-only — no formatting risk
- Chart updates must preserve ALL chart visual formatting (colors, line styles,
  chart type, legend position, axis formatting)
- Text/table updates from market data follow the same formatting-preserving
  `_replace_in_para()` / `_replace_in_cell()` path as proforma updates
- The `apply_branding()` function should NOT touch chart objects

### 6. Streamlit UI

Add a 4th column to the file upload row:

```
col1: Memo (.pptx) | col2: Proforma (.xlsx) | col3: Schedule (.mpp) | col4: Market Data (.xlsx)
```

- Accepts `.xlsx` / `.xlsm`
- Optional — "Fire!" button enabled without it
- New progress step at ~12%: "📊 Reading the market data..."
- Error handling: if file can't be read, show warning and continue without it

### 7. Pipeline Integration

When market data file is provided:

```python
# After proforma extraction, before memo extraction
if market_data_file:
    market_text = extract_market_data(market_data_path, cfg)
    proforma_data += "\n\n" + market_text
```

No changes to `get_metric_mappings()`, `validate_mappings()`, or prompt
templates for text/table updates. The chart_updates array is a new addition
that Claude will populate when it sees chart objects in the memo and
corresponding data in the market data section.

### 8. Prompt Additions

Add to the mapping prompt (appended section, not modifying existing rules):

```
## Market Data & Chart Updates

When MARKET DATA is provided and the memo contains embedded charts:

1. Match market data tabs to memo charts by semantic similarity — chart titles,
   series names, and axis labels may differ slightly from the data tab names.
2. For each matched chart, generate a chart_update entry with the updated
   series values from the market data.
3. Preserve all series that exist in the chart. Do not add or remove series
   unless the market data clearly warrants it.
4. If a chart layout differs from the data structure (e.g., different year
   range, different comparison markets), adapt the data to fit the chart's
   existing structure — do not restructure the chart.
5. Flag any charts where the match is uncertain for human review.
```

### 9. Error Handling

| Scenario | Behavior |
|----------|----------|
| Market data file can't be opened | Warning → continue without market data |
| Missing dashboard tabs | Warn per tab → extract available tabs |
| No dashboard tabs found | Warn → skip market data entirely |
| Chart extraction fails on a shape | Warn → skip that chart, continue |
| Chart update application fails | Log error → skip that chart, continue |
| Market data increases prompt beyond batch threshold | Existing batching handles it |

### 10. Testing Plan

1. **Unit tests:**
   - `test_extract_market_data_happy_path` — 6 tabs extracted, correct format
   - `test_extract_market_data_missing_tabs` — warns, extracts available
   - `test_extract_market_data_no_dashboard_tabs` — returns empty string
   - `test_extract_market_data_invalid_file` — raises graceful error
   - `test_chart_extraction` — charts in memo correctly extracted
   - `test_chart_update_application` — chart data correctly updated

2. **Integration tests:**
   - Full pipeline with proforma + market data (no schedule)
   - Full pipeline with proforma + market data + schedule
   - Pipeline with market data but no charts in memo
   - Pipeline without market data (regression — existing behavior unchanged)

3. **Stress tests:**
   - Large market data file (all 15 tabs extracted by accident)
   - Market data with missing/null values
   - Memo with many charts on different pages
   - Chart with series names that don't match any market data
