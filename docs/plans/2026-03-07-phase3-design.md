# Phase 3: Market Data & Intelligence — Design Document

> **Date:** 2026-03-07
> **Status:** Approved
> **Owner:** @brandon

---

## Scope

Phase 3 adds three capabilities to Memo Chef:

1. **Slide insertion** — generate new slides from supplemental data (PDF, URL, Excel)
2. **Run history dashboard** — view past runs with metrics in Streamlit
3. **Accuracy metrics + confidence scoring** — quantify pipeline reliability per run

Out of scope: batch processing (multi-memo), external API integrations (CoStar/Yardi).

---

## Feature 1: Slide Insertion (New Content Generation)

### Overview

Users upload supplemental data (PDF, URL, or Excel) that doesn't exist in the current memo. Claude analyzes the data, generates structured content (narrative + visual), and the pipeline inserts a new slide into the memo at the appropriate location, formatted to match the existing deck.

### Data Flow

```
Supplemental input (PDF / URL / Excel)
    |
    v
extract_supplemental()  -->  plain text
    |
    v
Claude: analyze_supplemental_content()
  Input: supplemental text + memo structure + optional user brief
  Output: JSON {data_points, narrative, visual_type, visual_data, target_section}
    |
    v
find_template_slide()  -->  best matching slide or None
    |
    +-- match found --> clone_and_replace()
    |                     deep-copy slide XML, replace chart/table data + narrative
    |
    +-- no match    --> build_slide_from_scratch()
                          create slide from master layout with title + chart/table + text
    |
    v
insert_slide_at_position()  -->  place after last slide in target section
```

### Components

#### `extract_supplemental(source, source_type)`
- **PDF**: Use `pdfplumber` (better table extraction than PyPDF2). Extract all text + detected tables. Return as structured text.
- **URL**: Use `requests` + `BeautifulSoup`. Extract visible text content, strip nav/footer. Return as plain text.
- **Excel**: Use `openpyxl` with `data_only=True`. Same extraction pattern as proforma (tab-delimited rows). Return as text.
- All paths return a unified plain-text representation.

#### `analyze_supplemental_content(supplemental_text, memo_structure, user_brief=None)`
- Claude API call (Sonnet).
- System prompt: "You are analyzing supplemental data for an investment committee memo. Given the memo structure and this new data, determine what content would be most valuable to add."
- Returns JSON:
  ```json
  {
    "slide_title": "Student Affluence Trends",
    "target_section": "Market Summary",
    "target_after_slide": 8,
    "narrative": "The University of Kentucky market shows...",
    "visual_type": "bar_chart",
    "visual_data": {
      "title": "Median Household Income by Zip Code",
      "categories": ["40502", "40503", "40508"],
      "series": [
        {"name": "Median HHI", "values": [62500, 58200, 45800]}
      ]
    },
    "data_points": [
      {"label": "Median HHI (3-mile)", "value": "$58,200"},
      {"label": "% Students with Parental Support", "value": "72%"}
    ]
  }
  ```

#### `find_template_slide(prs, target_section, visual_type)`
- Score each slide by:
  1. Section proximity (same section header = +10, adjacent = +5)
  2. Visual type match (chart→chart = +5, table→table = +5)
  3. Data density similarity (similar number of data points = +2)
- Return best match if score >= 10, else None.
- Section detection: look for section header slides (title-only slides, or slides matching known section names like "Market Summary", "Financial Summary").

#### `clone_and_replace(prs, template_slide_idx, content)`
- Deep-copy slide XML (lxml etree deepcopy of slide element + relationships).
- Walk shapes in cloned slide:
  - If chart: replace series data + categories in XML cache (same pattern as existing chart_updates).
  - If table: replace cell values row by row.
  - If text frame with narrative: replace paragraph text, preserving formatting of first paragraph.
- Replace slide title with `content.slide_title`.

#### `build_slide_from_scratch(prs, content)`
- Use the deck's first slide layout that has a title + content placeholder.
- Set title to `content.slide_title`.
- Based on `visual_type`:
  - `bar_chart` / `line_chart`: Create chart using python-pptx ChartData + add_chart().
  - `table`: Create table shape with rows/columns from visual_data.
  - `pie_chart`: Create pie chart.
- Add text box for narrative below the visual.
- Apply branding (Pragmatica font, Subtext colors) immediately.

#### `insert_slide_at_position(prs, new_slide, after_slide_idx)`
- Move slide element in presentation XML to position after `after_slide_idx`.
- Update slide numbering if page numbers are present.

### UI (Streamlit)

- New file uploader in the upload section: "Supplemental data (PDF, Excel, URL)"
  - Accept: `.pdf`, `.xlsx`, `.xlsm`, `.csv`
  - Separate text input for URL
- Optional text area: "Brief (optional)" — e.g., "Show student affluence trends"
- Change log entries: "SLIDE INSERTED: [title] after slide [N]"

### Prompt Design

System prompt for `analyze_supplemental_content`:
```
You are a real estate investment analyst creating content for an IC memo.

Given supplemental data and the current memo structure, generate a new slide.

Rules:
1. The slide must be relevant to the investment thesis
2. Choose the most impactful data to visualize
3. Write narrative in the same tone as the existing memo (professional, concise, data-driven)
4. Pick a visual type that best represents the data (bar_chart, line_chart, table, pie_chart)
5. Place the slide in the most contextually appropriate section
6. Keep narrative to 2-4 sentences
7. Include 3-8 data points in the visual

Return JSON only. No explanation outside JSON.
```

---

## Feature 2: Run History Dashboard

### Storage

Extend the existing SQLite DB (`memo_chef.db`) with enhanced run tracking.

#### Schema: `memo_chef_runs` table (extend existing)

```sql
CREATE TABLE IF NOT EXISTS memo_chef_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    project_name TEXT,
    memo_filename TEXT,
    proforma_filename TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'running',        -- running, completed, failed
    duration_seconds REAL,

    -- Counts
    changes_applied INTEGER DEFAULT 0,
    changes_rejected INTEGER DEFAULT 0,
    changes_missed INTEGER DEFAULT 0,
    slides_inserted INTEGER DEFAULT 0,

    -- Cost
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0.0,

    -- Accuracy
    confidence_score REAL,                -- 0-100 composite
    coverage_pct REAL,
    rejection_rate_pct REAL,
    correction_rate_pct REAL,

    -- Full data
    run_manifest_json TEXT,               -- full run_manifest.json blob
    change_log_html TEXT                  -- full change log HTML
);
```

#### Service functions (app_services.py)

- `save_run_record(user, manifest, change_log_html)` — insert after pipeline completes
- `get_run_history(user=None, limit=50)` — list runs, optionally filtered by user
- `get_run_detail(run_id)` — full manifest + change log for one run
- `compare_runs(run_id_a, run_id_b)` — diff two runs' change logs

### Streamlit Tab: "Run History"

- Table view: date, project, status, applied/rejected/missed, confidence, cost, duration
- Sortable + filterable by user, date range, project
- Click row to expand: full change log (rendered HTML), accuracy breakdown, cost details
- Compare button: select 2 runs → side-by-side diff of changes

---

## Feature 3: Accuracy Metrics + Confidence Scoring

### Per-Run Metrics

Computed after the validation pass, stored in `run_manifest.json` and DB.

| Metric | Formula | Weight |
|--------|---------|--------|
| **Coverage** | mappings_found / total_proforma_metrics | 30% |
| **Acceptance rate** | (total - rejected) / total | 25% |
| **Correction rate** | 1 - (corrections / total) | 20% |
| **Match quality** | non_degraded_matches / total_applied | 15% |
| **Miss rate** | 1 - (missed / total_proforma_metrics) | 10% |

**Confidence score** = weighted sum, scaled 0-100.

### Implementation

#### `compute_accuracy_metrics(mappings_raw, mappings_validated, apply_results)`

```python
def compute_accuracy_metrics(raw, validated, results) -> dict:
    total = len(raw.get("table_updates", [])) + len(raw.get("text_updates", [])) + ...
    rejected = len(validated.get("rejected", []))
    corrections = len(validated.get("corrections", []))
    missed = len(validated.get("missed", []))
    degraded = sum(1 for r in results if r.get("match_quality", "").startswith("degraded"))

    coverage = (total - missed) / max(total + missed, 1)
    acceptance = (total - rejected) / max(total, 1)
    correction_quality = 1 - corrections / max(total, 1)
    match_quality = (len(results) - degraded) / max(len(results), 1)
    miss_quality = 1 - missed / max(total + missed, 1)

    confidence = (
        coverage * 30 +
        acceptance * 25 +
        correction_quality * 20 +
        match_quality * 15 +
        miss_quality * 10
    )

    return {
        "confidence_score": round(confidence, 1),
        "coverage_pct": round(coverage * 100, 1),
        "rejection_rate_pct": round(rejected / max(total, 1) * 100, 1),
        "correction_rate_pct": round(corrections / max(total, 1) * 100, 1),
        "miss_rate_pct": round(missed / max(total + missed, 1) * 100, 1),
        "match_quality_pct": round(match_quality * 100, 1),
        "total_mappings": total,
        "rejected": rejected,
        "corrections": corrections,
        "missed": missed,
    }
```

### Display

- In run manifest JSON (machine-readable)
- In change log HTML (human-readable summary at top)
- In Streamlit run history table (confidence score column)
- Color-coded: green (80+), yellow (60-79), red (<60)

---

## Dependencies

- `pdfplumber` — PDF text + table extraction
- `beautifulsoup4` — URL content extraction
- `requests` — HTTP fetching (already in requirements)
- No new external API dependencies
