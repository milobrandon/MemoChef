# Market Data Pipeline Step — Design Spec

**Date:** 2026-03-27
**Status:** Approved for implementation

---

## Problem

Market data from external workbooks (RealPage, Yardi, custom Excel) is currently bundled into the proforma mapping pass as concatenated text. This means Claude has to simultaneously map proforma numbers AND reason about market metrics, chart matching, and cross-slide narrative consistency — too many responsibilities in one call. Additionally, chart updates today only patch numeric series values; categories, axes, series add/remove, and narrative coherence are not handled.

---

## Goal

Add a dedicated market data pipeline stage that mirrors the proforma flow: extract → map → validate → apply. Claude receives the full deck context and workbook data, reasons across all slides to find related content (charts, tables, narrative) wherever they live in the deck, and applies a rich update set to produce the same cohesive PPTX output.

---

## Architecture

```
Market Workbook (any format)
        ↓
[Dynamic Extraction]         — scans all tabs, heuristic keyword scoring, passes candidates to Claude for semantic labeling
        ↓
[Memo Content Snapshot]      — existing extract_memo_content() (all slides: charts, tables, text frames)
        ↓
[User Directions]            — from "Directions for Claude" UI field on the market data source
        ↓
[Market Data Mapping Pass]   — new Claude call (market_mapping_v1.txt prompt)
        ↓
[Market Data Validation]     — new Claude call (market_validation_v1.txt prompt)
        ↓
[Enhanced Apply Layer]       — chart (values + categories + series add/remove/rename), table, narrative
        ↓
Output: same cohesive PPTX + changelog section for market data updates
```

### Pipeline Stage Sequence

```
Stage 1: proforma_mapping
Stage 2: proforma_validation
Stage 3: apply_proforma_updates
Stage 4: market_data_mapping    ← NEW
Stage 5: market_data_validation ← NEW
Stage 6: apply_market_updates   ← NEW
```

Stages 4–6 are skipped automatically when no `market_data_path` is provided. All stages are checkpointed via `CheckpointManager` so a failed run can resume from the last completed stage.

---

## Dynamic Extraction

Replace the hardcoded `_MARKET_DASHBOARD_TABS` list with a two-phase scan that works for any workbook format (RealPage, Yardi, custom internal Excel, etc.).

**Phase 1 — Heuristic pre-filter:**
- Scan all tab names and column headers for market keywords: rent, occupancy, comp, supply, demand, absorption, vacancy, cap rate, market rate, pipeline, lease-up, submarket, MSA, PSF, etc.
- Score each tab by keyword hit count. Tabs scoring at or above `market_data.keyword_threshold` (default: 2) are included as candidates.
- If `market_data.include_all_tabs: true`, skip scoring and send everything.
- Existing fallback behavior (scan all tabs if no dashboard tabs found) is replaced by this logic.

**Phase 2 — Claude semantic labeling:**
- Candidate tab summaries (tab name, column headers, row count, first 5 data rows) are included in the market mapping prompt as a preamble section.
- Claude assigns a semantic label to each tab (e.g., "rent comps", "occupancy trend", "supply pipeline", "comp set") before reasoning about matches — no separate API call required.
- This labeling step is format-agnostic and works for any workbook layout.

**Output:** Structured text per tab with label, headers, and full data rows up to `max_rows_per_tab`.

---

## Prompts

### `prompts/market_mapping_v1.txt`

Inputs to Claude:
1. Full memo content snapshot (all slides, all shapes with page numbers)
2. Extracted + labeled market workbook data
3. User directions (if any) from the "Directions for Claude" UI field

Instructions to Claude:
- **User directions are highest priority.** Follow them exactly. They constrain or focus which updates are made.
- Where no directions exist, use full judgment.
- Identify all market metrics present in the memo (chart titles, table headers, narrative references) by slide and shape.
- For each memo metric, find the best match in the workbook data using semantic similarity — names need not match exactly (e.g., "Florida" ≈ "UF", "Rent Growth by Market" ≈ "Rent Growth Comparison By Year").
- If a match exists and the data differs, generate updates across **all pages** where related content appears (chart on slide 5, narrative on slide 7, summary table on slide 3 — all updated).
- If a chart doesn't match 1:1 (different markets, different time range), adapt it: update categories, add/remove series, rewrite adjacent narrative to stay coherent with the new chart data.
- Include a `reasoning` field on every update explaining the match and any adaptation made.
- If a match is uncertain, include the update but mark `"confidence": "low"` so validation can flag it.

### `prompts/market_validation_v1.txt`

Inputs to Claude:
1. The proposed market data update set from the mapping pass
2. The original memo content snapshot (for cross-referencing)
3. User directions (repeated, so validation can check compliance)

Instructions to Claude:
- Flag updates where the match is uncertain or the reasoning is weak.
- Flag values outside plausible ranges (rents < $0 or > $10K/unit/month, occupancy outside 0–100%, etc.).
- Flag narrative that contradicts the chart it references (e.g., narrative says "improving" but chart shows decline).
- Flag missing updates: if a chart was updated but a narrative on another page still references the old data, flag it.
- Flag user direction non-compliance: if directions said "only update rent growth" but other updates were generated.
- Return a cleaned update set (dropping or correcting flagged entries) and a `warnings` list for the changelog.

---

## Update Schema

Claude returns a single JSON object from the mapping pass:

```json
{
  "market_data_updates": [
    {
      "type": "chart_update",
      "page": 5,
      "chart_name": "Rent Growth",
      "series": [
        {
          "name": "Market A",
          "new_values": [1200, 1350, 1480],
          "old_values": [1100, 1250, 1400]
        }
      ],
      "categories": ["2022", "2023", "2024"],
      "add_series": [
        { "name": "Market D", "values": [1050, 1150, 1280] }
      ],
      "remove_series": ["Market C"],
      "source": "Rent Growth Comparison By Year tab",
      "reasoning": "Matched by semantic similarity. Adapted from 3 to 4 markets; Market C not present in workbook.",
      "confidence": "high"
    },
    {
      "type": "narrative_update",
      "page": 7,
      "old_text": "Market A rents grew 13.6% from 2022 to 2024...",
      "new_text": "Market A rents grew 23.3% from 2022 to 2024...",
      "source": "Rent Growth Comparison By Year tab",
      "reasoning": "Updated figures to match new chart data on slide 5.",
      "confidence": "high"
    },
    {
      "type": "table_update",
      "page": 3,
      "slide_table": "Market Summary",
      "updates": [
        { "row": 2, "col": 3, "old_value": "94.1%", "new_value": "96.2%" }
      ],
      "source": "Tables tab",
      "reasoning": "Updated occupancy figure from market workbook Tables tab row 4.",
      "confidence": "high"
    }
  ],
  "unmatched_memo_metrics": ["Absorption Rate chart on slide 9"],
  "unmatched_workbook_tabs": ["Uncaptured Demand Comparison"]
}
```

`unmatched_memo_metrics` and `unmatched_workbook_tabs` surface in the changelog so the user knows what was intentionally left alone.

---

## Enhanced Application Layer

### Chart Updates (`_apply_chart_updates()` extended)

Current behavior: patches `<c:val>` numeric cache elements.

New behavior:
- **Category updates** — patch `<c:cat>` XML elements to update x-axis labels
- **Series add** — clone an existing `<c:ser>` XML node, update `<c:tx>` (name) and `<c:val>` (values), append to chart
- **Series remove** — find `<c:ser>` by name match, remove node
- **Series rename** — update `<c:ser><c:tx>` text without changing values
- All existing fuzzy-matching logic (chart by name, title, single-chart-on-page fallback) preserved

### Narrative Updates

Reuses existing `_apply_narrative_updates()` / `_replace_in_para()` infrastructure. The `source` field in each update distinguishes proforma vs. market data origin in the changelog.

### Table Updates

Reuses existing `apply_updates()` table path. No changes needed.

---

## User Directions Integration

The "Directions for Claude" UI field (already exists for proforma sources via PR #19) is extended to apply to the market data source as well. When a user uploads a market workbook, they can optionally enter free-text instructions:

> "Only update the rent growth chart. Do not touch occupancy or supply slides."
> "Add a narrative bullet about improving absorption rates if the data supports it."
> "The comp set should reflect only properties within 2 miles."

These directions are injected at the top of the market mapping prompt as highest-priority instructions. They are repeated in the validation prompt so the validator can check compliance. If no directions are provided, the field is omitted from the prompt.

---

## Config Changes

```yaml
market_data:
  enabled: true
  max_rows_per_tab: 250       # rows extracted per workbook tab
  max_cols_per_tab: 20        # columns extracted per workbook tab
  keyword_threshold: 2        # min keyword hits for tab inclusion
  include_all_tabs: false     # override: send all tabs regardless of score
```

The existing `proforma.max_rows_per_tab` and `proforma.max_cols_per_tab` keys remain unchanged (proforma extraction is separate).

---

## Backward Compatibility

- No `market_data_path` → stages 4–6 skipped, behavior identical to today
- Existing `extract_market_data()` function replaced by the new dynamic extractor; same return format (structured text) so existing callers are unaffected
- `_apply_chart_updates()` is extended, not replaced; existing call sites work unchanged

---

## Changelog / Before-After Report

The existing before-after report (`memo_chef/redline.py`) gains a `market_data` section:
- Metrics matched (source tab → memo shape, page)
- Updates applied per slide
- Unmatched memo metrics (no workbook data found)
- Unmatched workbook tabs (no memo content found)
- Validation warnings (uncertain matches, flagged values)

---

## Out of Scope

- Creating new slides for market data (slide insertion is a separate feature)
- Generating new charts from scratch
- Pulling live data from external APIs (FRED, CoStar, etc.)
- Chart visual styling (colors, fonts) — python-pptx chart styling is unreliable
