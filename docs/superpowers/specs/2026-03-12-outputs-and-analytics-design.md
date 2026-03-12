# Outputs & Analytics Feature Pack — Design Spec

**Created:** 2026-03-12
**Author:** @brandon + Claude
**Status:** Approved (design review complete)
**Scope:** 5 features — visual diff, comp slide builder, chart updating, run analytics, proforma drift detection

---

## Overview

Five features that expand Memo Automator's output capabilities and give admins visibility into pipeline health. All build on existing infrastructure (run history DB, slide insertion engine, proforma extraction). No changes to core mapping/validation prompts.

**Priority order for implementation:**
1. F10 — Proforma Drift Detection (highest reviewer value, lowest complexity)
2. F6 — Slide Image Diff with Redline Overlay (highest visual impact)
3. F9 — Run Analytics Dashboard (admin visibility, no pipeline changes)
4. F8 — Chart Data Updating from Market Workbooks (extends market data story)
5. F7 — Market Comp Slide Builder (most complex, builds on F8)

---

## F6: Slide Image Diff with Redline Overlay

### Purpose

After a pipeline run, render before/after slide images and display a redline-style visual diff in the Streamlit UI — showing reviewers exactly what changed on each slide with highlighted change regions.

### How It Works

1. **Slide export**: Use `win32com.client` to open both the backup PPTX (pre-changes) and the updated PPTX in PowerPoint. Export only slides that had changes as PNG images.

2. **Diff overlay**: Use Pillow to:
   - Compute a pixel diff between before/after images for each changed slide
   - Generate a redline composite: the "after" image with red-tinted bounding boxes around regions that differ
   - Generate an optional side-by-side view (before | after)

3. **Streamlit display**: New "Redline" expander in the results section:
   - Slide picker (dropdown or prev/next arrows) filtered to changed slides only
   - View modes: "Redline overlay", "Side by side", "Before only", "After only"
   - Numbered annotations on changed regions linking to the change log entry
   - Badge per slide showing change count

### New Files

- `memo_chef/redline.py` — slide export + diff image generation

### Key Functions

```python
def export_slides_as_images(
    pptx_path: str,
    slide_numbers: list[int],
    output_dir: str,
    dpi: int = 150,
) -> dict[int, Path]:
    """Export specific slides from a PPTX as PNG using PowerPoint COM.

    Args:
        pptx_path: Path to the .pptx file.
        slide_numbers: 1-based slide numbers to export.
        output_dir: Directory for output PNGs.
        dpi: Resolution (150 = good balance of quality and size).

    Returns:
        Dict mapping slide number to PNG file path.
    """

def generate_redline_image(
    before_img: bytes,
    after_img: bytes,
    threshold: int = 30,
) -> bytes:
    """Generate a redline overlay image highlighting pixel differences.

    Args:
        before_img: PNG bytes of the slide before changes.
        after_img: PNG bytes of the slide after changes.
        threshold: Pixel difference threshold for change detection.

    Returns:
        PNG bytes of the redline composite image.
    """

def generate_side_by_side(
    before_img: bytes,
    after_img: bytes,
    label_before: str = "Before",
    label_after: str = "After",
) -> bytes:
    """Generate a side-by-side comparison image.

    Returns:
        PNG bytes of the combined image.
    """
```

### Streamlit Integration

In `app.py` results section, after the existing metrics and download buttons:

```python
# Only show if changes were applied and images generated
if st.session_state.get("redline_images"):
    with st.expander("Redline View", expanded=True):
        changed_slides = sorted(st.session_state["redline_images"].keys())
        selected = st.selectbox("Slide", changed_slides,
                                format_func=lambda x: f"Slide {x} ({change_counts[x]} changes)")
        view = st.radio("View", ["Redline", "Side by side", "Before", "After"],
                        horizontal=True)
        # Display selected view
        st.image(...)
```

### Session State Keys

- `redline_images`: `dict[int, dict]` — `{slide_num: {"redline": bytes, "side_by_side": bytes, "before": bytes, "after": bytes}}`
- Generated after pipeline completion, not persisted to DB (too large)

### Dependencies

- `pywin32` (for `win32com.client`) — pip installable
- `Pillow` — pip installable
- PowerPoint must be installed on the host machine

### Deployment Constraints

- **Windows-local only**: `win32com.client` requires Windows + PowerPoint installed and licensed. This feature is unavailable on Streamlit Cloud, Linux containers, or any non-Windows host.
- **COM cleanup**: PowerPoint COM can leave zombie `POWERPNT.EXE` processes on failure. The export function must use a `try/finally` block that calls `app.Quit()` and kills any orphaned process after a 30-second timeout.
- **Thread safety**: COM apartment threading requires `pythoncom.CoInitialize()` at the start of the export function and `CoUninitialize()` on exit. Only one export can run at a time (use a threading lock).
- **Graceful degradation**: If PowerPoint is not available, skip redline generation silently, log a warning, and show a text-only diff fallback in the UI.

### Edge Cases

- If PowerPoint is not available, skip redline generation and fall back to text-based change summary
- If a slide has only chart data changes (no visual pixel diff), the redline may not highlight anything visible — note this in the UI
- Large decks (50+ changed slides): cap export to first 20 changed slides, show "N more not rendered" message
- Memory: each 150 DPI slide PNG is ~2-5 MB. For 20 slides, budget ~100 MB session state. Store images on disk in the run output directory; session state holds file paths only.

---

## F7: Market Comp Slide Builder

### Purpose

Auto-generate a formatted rent comparison slide from any structured comp data source — comp URLs, RealPage, CSV upload, or manual entry. Clones the existing comp slide format from the memo to preserve branding and layout.

### How It Works

1. **Unified comp schema**: All sources normalize into a common Pydantic model.

2. **Comp aggregation**: Collect from all provided sources, deduplicate by name/address fuzzy match (rapidfuzz, threshold > 85), merge fields with source priority: RealPage > CSV > URL scrape > manual entry (higher-priority source wins on conflicts).

3. **Slide generation**:
   - **Template clone** (preferred): Find the existing comp table slide in the memo, clone its layout via `clone_slide()`, repopulate cells with aggregated comp data. Subject property in leftmost column.
   - **From scratch** (fallback): Use `build_slide_from_scratch()` with a comp-specific table layout using Subtext brand theme.

4. **Optional Claude assistance** (Haiku): Select most relevant comp fields for this deal type, generate a 1-2 sentence narrative summarizing comp position.

### New Files

- `memo_chef/comp_builder.py` — comp aggregation, schema normalization, slide generation

### Data Model

```python
class UnitMixEntry(BaseModel):
    unit_type: str          # "Studio", "1BR", "2BR", etc.
    beds: int | None = None
    baths: int | None = None
    sf: int | None = None
    rent: float | None = None
    rent_per_sf: float | None = None

class CompProperty(BaseModel):
    name: str
    address: str | None = None
    distance_mi: float | None = None
    unit_mix: list[UnitMixEntry] = Field(default_factory=list)
    total_units: int | None = None
    occupancy_pct: float | None = None
    year_built: int | None = None
    concessions: str | None = None
    source: str              # "url", "realpage", "csv", "manual"
    source_detail: str = ""  # URL, filename, etc.

class CompSlideRequest(BaseModel):
    subject_property: CompProperty
    comps: list[CompProperty]
    sort_by: str = "distance"  # "distance", "rent", "units"
    max_comps: int = 6         # typical IC slide fits 5-6 comps
    include_narrative: bool = True
```

### Key Functions

```python
def normalize_comps_from_csv(csv_path: str) -> list[CompProperty]:
    """Parse a CSV/Excel file into CompProperty objects.
    Expected columns: Name, Address, Units, Unit Type, Beds, SF, Rent, Occupancy, Year Built.
    Flexible column matching (case-insensitive, partial match).
    """

def normalize_comps_from_urls(comp_urls: list[CompUrl], extracted_texts: dict) -> list[CompProperty]:
    """Convert scraped comp URL data into CompProperty objects.
    Uses existing extraction.py output.
    """

def normalize_comps_from_realpage(market_data: dict) -> list[CompProperty]:
    """Convert RealPage dashboard data into CompProperty objects."""

def deduplicate_comps(comps: list[CompProperty]) -> list[CompProperty]:
    """Fuzzy-match by name/address (rapidfuzz token_sort_ratio > 85).
    On conflict, prefer source with higher priority: realpage > csv > url > manual.
    Merge missing fields from lower-priority sources."""

def build_comp_slide(
    prs: Presentation,
    subject: CompProperty,
    comps: list[CompProperty],
    memo_sections: list[dict],
    narrative: str | None = None,
) -> None:
    """Clone existing comp slide or build from scratch, populate with comp data.
    Inserts at the appropriate position (after Competitive Landscape section).
    """
```

### Streamlit UI

In the run config tab, new "Comp Builder" section:

```
[x] Auto-generate comp slide

Comp data sources:
  [x] Comp URLs (3 provided above)
  [ ] RealPage market data (upload .xlsx)
  [ ] CSV comp data (upload)
  [ ] Manual entry

  [+ Add comp manually]
    Property name: ___________
    Total units: ___
    Avg rent: ___
    Avg SF: ___
    Occupancy: ___
    Year built: ___
    Distance (mi): ___
```

### Dependencies

- No new dependencies (python-pptx already available)

---

## F8: Chart Data Updating from Market Workbooks

### Purpose

When a market data workbook with charts is provided, extract the data and update corresponding charts in the memo. User provides instructions telling Claude what to extract and which memo charts to update.

### How It Works

1. **UI**: New input in run config:
   - Market data workbook upload (Excel)
   - Text box: "Chart instructions" — user tells Claude what to extract (e.g., "Update the rent trend chart on slide 12 with submarket rents from the 'Rent Growth' tab")

2. **Extraction**: Parse the uploaded workbook using **tabular data extraction only** (not openpyxl chart objects, which have unreliable read support). Extract all tab data as text using the same `extract_proforma_data()` pattern. Claude does the heavy lifting of interpreting which data maps to which chart — this is the same approach that already works for supplemental data.

3. **Claude mapping**: Haiku call that takes:
   - Extracted tabular data from the workbook (text representation of all tabs)
   - User's chart instructions (tells Claude which tabs contain which chart data)
   - List of existing memo charts (title, type, series names — already extracted during memo parsing)
   - Returns structured output: which memo chart to update, which series, new values, new category labels
   - Uses `tool_use` with forced tool choice for guaranteed structured output

4. **Application**: Feed structured output into existing `_apply_chart_updates()` — extended to also handle category labels and data labels.

### New Files

- `memo_chef/chart_extraction.py` — extract chart data from Excel workbooks

### Key Functions

```python
def extract_workbook_tables(workbook_path: str, tab_names: list[str] | None = None) -> str:
    """Extract tabular data from all (or specified) tabs as text representation.
    Same format as extract_proforma_data() for consistency:

        ======================================================================
        TAB: Rent Growth Comparison
        ======================================================================
        Row 1:	Year	Submarket	Subject	Comp Avg
        Row 2:	2022	3.2%	4.1%	3.5%
        ...

    If tab_names is None, extract all non-empty tabs.
    """

def map_market_charts(
    workbook_data: list[dict],
    memo_charts: list[dict],
    user_instructions: str,
    client: Any,
    model: str = "claude-haiku-4-5",
) -> list[dict]:
    """Use Claude to map workbook chart data to memo charts based on user instructions.

    Returns list of chart_update dicts compatible with _apply_chart_updates().
    """
```

### Prompt Pattern

Same as existing supplemental data flow: user provides instructions, Claude follows them:

```
You are a financial data analyst. The user has provided a market data workbook
and instructions for updating charts in an IC memo.

## User Instructions
{user_instructions}

## Workbook Charts and Data
{workbook_data_json}

## Existing Memo Charts
{memo_charts_json}

Return a JSON array of chart updates. Each update:
{
  "page": int,
  "chart_name": str,
  "series_name": str,
  "new_values": [numbers],
  "new_categories": [strings] | null,
  "source": str
}
```

### Extension to `_apply_chart_updates()`

Add support for:
- **Category label updates**: Modify `c:strRef/c:strCache` elements for x-axis labels
- **Data label sync**: After updating series values, find associated `c:dLbl` elements and update their text

### Streamlit UI

In run config, after existing file uploads:

```
Market Data (Charts)
  Upload workbook: [Choose file]  (.xlsx)
  Chart instructions:
  ┌─────────────────────────────────────────────────────┐
  │ Update the rent trend chart on slide 12 with        │
  │ submarket rents from the 'Rent Growth' tab.         │
  │ Update the pipeline chart with 'Construction        │
  │ Pipeline' tab data.                                 │
  └─────────────────────────────────────────────────────┘
```

### Dependencies

- `openpyxl` (already a dependency for proforma extraction)

---

## F9: Run Analytics Dashboard (Admin Only)

### Purpose

System-wide view of tool health, usage trends, and cost tracking. Lives in the existing admin panel tab.

### Data Source

All data comes from the existing `memo_chef_runs` table. No new data collection or storage. Pure aggregation queries.

### Dashboard Sections

1. **Summary cards** (top row, `st.metric()`):
   - Total runs (all time / last 30 days)
   - Average confidence score
   - Total API spend (formatted as USD)
   - Average run duration
   - Total changes applied

2. **Cost trend** (`st.line_chart()`):
   - Estimated API cost per run over time
   - Rolling 7-day average line
   - Validates that prompt optimizations reduce costs

3. **Accuracy trend** (`st.line_chart()`):
   - Confidence score over time
   - Rejection rate and miss rate as secondary series
   - Shows whether pipeline quality is stable or degrading

4. **Per-user breakdown** (`st.dataframe()`):
   - Columns: username, run count, avg confidence, total cost, last run date
   - Sortable, searchable

5. **Warning frequency** (`st.bar_chart()`):
   - Most common warnings across all runs
   - Surfaces systematic issues (e.g., truncation warnings spiking)

6. **Time savings estimate** (`st.metric()`):
   - Configurable baseline: "manual update time per memo" (default: 4 hours)
   - Formula: `total_runs * baseline_hours - sum(actual_duration_hours)`
   - Shows cumulative hours saved

### New Functions in `app_services.py`

```python
def get_run_analytics(days: int | None = None) -> dict:
    """Aggregate run statistics for the analytics dashboard.

    Args:
        days: Filter to last N days. None = all time.

    Returns:
        {
            "total_runs": int,
            "total_cost_usd": float,
            "avg_confidence": float,
            "avg_duration_sec": float,
            "total_changes": int,
            "cost_by_date": [{"date": str, "cost_usd": float}],
            "accuracy_by_date": [{"date": str, "confidence": float, "rejection_rate": float, "miss_rate": float}],
            "by_user": [{"username": str, "runs": int, "avg_confidence": float, "total_cost_usd": float, "last_run": str}],
            "warning_counts": [{"warning": str, "count": int}],
        }
    """
```

### Streamlit Integration

New section in the admin panel tab (after existing user management):

```python
st.subheader("Run Analytics")
period = st.selectbox("Period", ["Last 7 days", "Last 30 days", "All time"])
analytics = get_run_analytics(days=period_to_days(period))

# Summary cards
cols = st.columns(5)
cols[0].metric("Total Runs", analytics["total_runs"])
cols[1].metric("Avg Confidence", f"{analytics['avg_confidence']:.0f}/100")
# ... etc

# Charts
st.line_chart(cost_df, x="date", y="cost_usd")
st.line_chart(accuracy_df, x="date", y=["confidence", "rejection_rate", "miss_rate"])
```

### Dependencies

- None new — all native Streamlit components

---

## F10: Proforma Drift Detection

### Purpose

When a user uploads a proforma for a property that's been run before, automatically compare against the previous run's proforma and highlight what changed. Reviewers know exactly which numbers moved before they even look at the updated memo.

### How It Works

1. **Snapshot storage**: After each run, store the extracted proforma text (output of `extract_proforma_data()`) keyed by property name. Lightweight — reuses the already-generated text.

2. **Drift detection on upload**: When a new run starts for the same property:
   - Extract the new proforma
   - Retrieve the previous snapshot
   - Diff line-by-line and cell-by-cell

3. **Drift summary in UI**: Before the pipeline runs, show a "What Changed" panel:
   - "23 values changed since last run on 2026-03-08"
   - Grouped by tab: "Assumptions: 12 changes, Dev Budget: 8 changes"
   - Expandable detail: tab, row label, column, old value → new value

4. **Carry-through to change log**: After the run, CHANGE_LOG.md gets a "Proforma Drift" section at the top listing input changes. Reviewers cross-reference: "rent went from $1,325 to $1,350 in the proforma → memo updated in 6 places."

### New DB Table

```sql
CREATE TABLE IF NOT EXISTS proforma_snapshots (
    id TEXT PRIMARY KEY,  -- UUID generated in Python, matches run_id/job_id pattern
    property_name TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES memo_chef_runs(run_id),
    extracted_text TEXT NOT NULL,
    tab_hashes JSONB,          -- {"Assumptions": "abc123", "Dev Budget": "def456"}
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_proforma_snapshots_property
    ON proforma_snapshots(property_name, created_at DESC);
```

Keep last 3 snapshots per property (configurable). Older ones auto-pruned on insert.

**Property name normalization**: `property_name` is user-entered and inconsistent. Before storing and querying, normalize: lowercase, strip whitespace, remove common prefixes ("the ", "at "). In the UI, offer a dropdown of previously seen property names (populated from `memo_chef_runs`) alongside free-text entry.

### New Files

- `memo_chef/drift.py` — diff logic + snapshot comparison

### Proforma Text Format Spec

`extract_proforma_data()` produces a deterministic text format:

```
======================================================================
TAB: Assumptions
======================================================================
Row 1:	Property Name	The Reserve at Oak Creek
Row 2:	Total Units	250
Row 3:	Total Beds	510
Row 5:	Avg Rent/Bed	1325
Row 6:	Occupancy	0.94
...
======================================================================
TAB: Development Summary
======================================================================
Row 1:	Category	Amount	% of Total
Row 2:	Land	12500000	0.2403846
Row 3:	Hard Costs	28000000	0.5384615
...
```

**Grammar:**
- Tab headers: `={70}\nTAB: {tab_name}\n={70}`
- Data rows: `Row {row_num}:\t{val1}\t{val2}\t...` (tab-delimited, only non-empty cells)
- Empty rows are omitted entirely
- Cell values are stringified Python values (numbers lose formatting)

### Key Functions

```python
def parse_proforma_to_cells(text: str) -> dict[str, dict[int, list[str]]]:
    """Parse extract_proforma_data() text into structured cells.

    Parsing approach:
    1. Split on '={70}' delimiters to find tab boundaries
    2. Extract tab name from 'TAB: {name}' lines
    3. For each 'Row N:\\t...' line, split on tab to get cell values
    4. Key by (tab_name, row_number) for O(1) lookup

    Returns: {"TabName": {row_num: [val1, val2, ...], ...}, ...}
    """

def compute_proforma_diff(
    previous_text: str,
    current_text: str,
) -> dict:
    """Compare two proforma text extractions and return structured diff.

    Algorithm:
    1. Parse both texts with parse_proforma_to_cells()
    2. For each tab present in either version:
       - Match rows by row number (exact — row numbers are stable cell references)
       - For matched rows, compare cell-by-cell by position index
       - Rows in current but not previous → "added"
       - Rows in previous but not current → "removed"
       - Rows in both with different values → "changed" (per-cell diff)
    3. Aggregate counts by tab

    Row alignment note: Row numbers from Excel are stable identifiers
    (they correspond to actual spreadsheet row numbers, not sequential
    indices). This avoids the need for fuzzy row matching.

    Returns:
    {
        "total_changes": int,
        "by_tab": {
            "Assumptions": {
                "added": [{"row": 15, "values": ["Studio", "398", "1100"]}],
                "changed": [{"row": 5, "col_idx": 1, "old": "1325", "new": "1350"}],
                "removed": [{"row": 22, "values": ["..."]}],
            },
            ...
        },
        "summary": "23 values changed across 3 tabs",
    }
    """
```

### Caching to Avoid Double-Extraction

The drift check calls `extract_proforma_data()` before the run, and the pipeline calls it again during the run. To avoid double-extraction:

```python
# In app.py, before pipeline run:
proforma_text = extract_proforma_data(proforma_path, cfg)
st.session_state["cached_proforma_text"] = proforma_text

# Pipeline receives cached text via RunRequest or session state
# so it skips re-extraction if already available
```

### New Functions in `app_services.py`

```python
def store_proforma_snapshot(
    property_name: str,
    run_id: str,
    extracted_text: str,
    tab_hashes: dict[str, str] | None = None,
) -> None:
    """Store a proforma snapshot. Auto-prunes to keep last 3 per property."""

def get_previous_proforma_snapshot(property_name: str) -> dict | None:
    """Retrieve the most recent snapshot for a property.
    Returns: {"run_id": str, "extracted_text": str, "created_at": str} or None.
    """
```

### Streamlit Integration

In the run config, after file upload and before the "Run" button:

```python
# Check for drift if property name is set and files uploaded
if property_name and proforma_uploaded:
    prev = get_previous_proforma_snapshot(property_name)
    if prev:
        current_text = extract_proforma_data(proforma_path, cfg)
        diff = compute_proforma_diff(prev["extracted_text"], current_text)
        if diff["total_changes"] > 0:
            st.info(f"**Proforma drift detected:** {diff['summary']} "
                    f"(vs. run on {prev['created_at'][:10]})")
            with st.expander("View changes"):
                for tab, changes in diff["by_tab"].items():
                    n = len(changes["changed"]) + len(changes["added"]) + len(changes["removed"])
                    st.markdown(f"**{tab}**: {n} changes")
                    # ... detail table
```

### Change Log Integration

In `write_change_log()`, add a "Proforma Drift" section before "Applied Changes":

```markdown
## Proforma Drift (vs. run on 2026-03-08)

23 values changed across 3 tabs:

### Assumptions (12 changes)
| Row | Column | Previous | Current |
|-----|--------|----------|---------|
| Avg Rent/Bed | Monthly | $1,325 | $1,350 |
| Occupancy | Stabilized | 94.0% | 95.5% |
...

### Dev Budget (8 changes)
...
```

### Dependencies

- None new

---

## Shared Concerns

### New Dependencies Summary

| Feature | New Dependencies |
|---------|-----------------|
| F6 | `pywin32`, `Pillow` |
| F7 | `rapidfuzz` |
| F8 | None |
| F9 | None |
| F10 | None |

### DB Migrations

- F10: New `proforma_snapshots` table
- F9: No new tables (queries existing `memo_chef_runs`)
- Migration pattern: Follow existing `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` pattern in `get_db_conn()` (app_services.py). No migration tool needed.

### Model Updates (`memo_chef/models.py`)

`RunRequest` needs new fields for F7 and F8:

```python
# F7: Comp Builder
comp_csv_path: str | None = None
comp_manual_entries: list[dict] | None = None
auto_generate_comp_slide: bool = False
comp_max_comps: int = 6
comp_sort_by: str = "distance"

# F8: Chart Updating
market_workbook_path: str | None = None
chart_instructions: str | None = None
```

### Analytics Query Notes (F9)

- `warnings_json` is TEXT containing JSON — parse in Python, not SQL, for warning frequency aggregation
- `duration_seconds` is nullable — exclude nulls from average calculations
- Rejection/miss rates not stored directly — compute from `rejected_count / (change_count + rejected_count)` per row

### Testing Strategy

Each feature gets its own test file:
- `tests/test_redline.py` — mock COM calls, test Pillow diff logic with fixture PNGs
- `tests/test_comp_builder.py` — test normalization from CSV/URL/manual, dedup logic
- `tests/test_chart_extraction.py` — test workbook chart parsing with fixture Excel files
- `tests/test_run_analytics.py` — test aggregation queries with seeded DB data
- `tests/test_drift.py` — test proforma diff with fixture text pairs

### Implementation Order Rationale

1. **F10 (Drift)** first — lowest complexity, highest immediate reviewer value, no new dependencies
2. **F6 (Redline)** second — highest visual impact, validates the win32com integration pattern needed by other features
3. **F9 (Analytics)** third — pure UI, no pipeline changes, quick win for admin visibility
4. **F8 (Charts)** fourth — extends market data story, establishes workbook extraction pattern
5. **F7 (Comp Builder)** last — most complex, benefits from F8's extraction work and established patterns
