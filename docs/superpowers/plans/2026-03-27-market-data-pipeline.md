# Market Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated market data pipeline stage (map → validate → apply) that updates charts, tables, and narrative across the full deck using any Excel market workbook, driven by Claude's semantic reasoning.

**Architecture:** Market data is extracted separately from the proforma using a keyword-scoring tab scanner. Three new pipeline stages run after the proforma apply: `market_data_mapping` (Claude reasons across all slides), `market_data_validation` (Claude QA's matches), and `apply_market_updates` (enhanced PPTX XML patching). User directions from the existing "Directions for Claude" UI field are wired through to Claude.

**Tech Stack:** python-pptx (lxml XML access), openpyxl, anthropic SDK, Pydantic v2, Streamlit

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `memo_chef/models.py` | Modify | Add `MarketDataUpdateSet` and related models |
| `config.yaml` | Modify | Add `market_data:` section |
| `memo_automator.py` | Modify | Refactor `extract_market_data()`, add `MARKET_MAPPING_PROMPT`, `MARKET_VALIDATION_PROMPT`, `get_market_data_mappings()`, `validate_market_data_mappings()`, `_apply_market_chart_update()`, `apply_market_updates()` |
| `prompts/market_mapping_v1.txt` | Create | Market data mapping prompt |
| `prompts/market_validation_v1.txt` | Create | Market data validation prompt |
| `memo_chef/pipeline.py` | Modify | Separate market data from proforma concat, add 3 new stages after `apply` |
| `app.py` | Modify | Enable market data directive field, add Memo City News panel |
| `memo_city_news.json` | Create | Changelog feed for Memo City News panel |
| `tests/test_market_data.py` | Modify | Add tests for new extraction behavior and models |
| `tests/stress_test_market_data.py` | Modify | Add tests for enhanced chart apply |
| `test_pipeline_integration_mocked.py` | Modify | Add market data pipeline stage tests |

---

## Task 1: Pydantic Models for Market Data Updates

**Files:**
- Modify: `memo_chef/models.py`
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_market_data.py`:

```python
from memo_chef.models import (
    MarketChartUpdate, MarketNarrativeUpdate,
    MarketTableCellUpdate, MarketTableUpdate, MarketDataUpdateSet,
)


class TestMarketDataModels:
    def test_chart_update_validates(self):
        u = MarketChartUpdate(
            page=3,
            chart_name="Rent Growth",
            series=[{"name": "Market A", "new_values": [1200, 1350], "old_values": [1100, 1250]}],
            categories=["2023", "2024"],
            add_series=[{"name": "Market D", "values": [900, 950]}],
            remove_series=["Market C"],
            source="Rent Growth tab",
            reasoning="Semantic match",
            confidence="high",
        )
        assert u.type == "chart_update"
        assert u.page == 3
        assert u.confidence == "high"

    def test_narrative_update_validates(self):
        u = MarketNarrativeUpdate(
            page=7,
            old_text="Rents grew 5%",
            new_text="Rents grew 12%",
            source="Rent Growth tab",
            reasoning="Updated figures",
            confidence="high",
        )
        assert u.type == "narrative_update"

    def test_table_update_validates(self):
        u = MarketTableUpdate(
            page=3,
            slide_table="Market Summary",
            updates=[{"row": 2, "col": 1, "old_value": "94%", "new_value": "96%"}],
            source="Tables tab",
            reasoning="Occupancy updated",
            confidence="medium",
        )
        assert u.type == "table_update"
        assert u.updates[0].row == 2

    def test_update_set_validates_mixed(self):
        s = MarketDataUpdateSet(
            market_data_updates=[
                {"type": "chart_update", "page": 3, "series": [], "source": "x", "reasoning": "y", "confidence": "high"},
                {"type": "narrative_update", "page": 5, "old_text": "a", "new_text": "b", "source": "x", "reasoning": "y", "confidence": "high"},
            ],
            unmatched_memo_metrics=["Absorption chart p9"],
            unmatched_workbook_tabs=["Backend"],
            warnings=["Low confidence match on p3"],
        )
        assert len(s.market_data_updates) == 2
        assert s.warnings == ["Low confidence match on p3"]

    def test_empty_update_set(self):
        s = MarketDataUpdateSet()
        assert s.market_data_updates == []
        assert s.warnings == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "C:\Users\BrandonZmuda\Desktop\Claude\g. Memo Automator\v2"
python -m pytest tests/test_market_data.py::TestMarketDataModels -v
```

Expected: `ImportError` — models don't exist yet.

- [ ] **Step 3: Add models to `memo_chef/models.py`**

Add after the `DeckProfile` class (end of file):

```python
# ── Market Data Update Schema ─────────────────────────────────────────────────

class ChartSeriesUpdate(BaseModel):
    name: str
    new_values: list[float | int | None]
    old_values: list[float | int | None] = Field(default_factory=list)


class ChartSeriesAdd(BaseModel):
    name: str
    values: list[float | int | None]


class MarketChartUpdate(BaseModel):
    type: str = "chart_update"
    page: int
    chart_name: str | None = None
    chart_title: str | None = None
    series: list[ChartSeriesUpdate] = Field(default_factory=list)
    categories: list[str] | None = None
    add_series: list[ChartSeriesAdd] = Field(default_factory=list)
    remove_series: list[str] = Field(default_factory=list)
    source: str
    reasoning: str
    confidence: str = "high"  # "high" | "medium" | "low"


class MarketNarrativeUpdate(BaseModel):
    type: str = "narrative_update"
    page: int
    old_text: str
    new_text: str
    source: str
    reasoning: str
    confidence: str = "high"


class MarketTableCellUpdate(BaseModel):
    row: int
    col: int
    old_value: str
    new_value: str


class MarketTableUpdate(BaseModel):
    type: str = "table_update"
    page: int
    slide_table: str
    updates: list[MarketTableCellUpdate]
    source: str
    reasoning: str
    confidence: str = "high"


class MarketDataUpdateSet(BaseModel):
    market_data_updates: list[dict] = Field(default_factory=list)
    unmatched_memo_metrics: list[str] = Field(default_factory=list)
    unmatched_workbook_tabs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def chart_updates(self) -> list[MarketChartUpdate]:
        return [MarketChartUpdate(**u) for u in self.market_data_updates if u.get("type") == "chart_update"]

    def narrative_updates(self) -> list[MarketNarrativeUpdate]:
        return [MarketNarrativeUpdate(**u) for u in self.market_data_updates if u.get("type") == "narrative_update"]

    def table_updates(self) -> list[MarketTableUpdate]:
        return [MarketTableUpdate(**u) for u in self.market_data_updates if u.get("type") == "table_update"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_market_data.py::TestMarketDataModels -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/market-data-pipeline
git add memo_chef/models.py tests/test_market_data.py
git commit -m "feat: add market data update schema models"
```

---

## Task 2: Add market_data Config Section

**Files:**
- Modify: `config.yaml`
- Test: `test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `test_config.py`:

```python
def test_market_data_config_defaults():
    """config.yaml must parse with market_data section and correct defaults."""
    cfg = load_config("config.yaml")
    md = cfg.get("market_data", {})
    assert md.get("max_rows_per_tab") == 250
    assert md.get("max_cols_per_tab") == 20
    assert md.get("keyword_threshold") == 2
    assert md.get("include_all_tabs") is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest test_config.py::test_market_data_config_defaults -v
```

Expected: FAIL — `market_data` key missing.

- [ ] **Step 3: Add to `config.yaml`**

Add after the `schedule:` section:

```yaml
# --- Market data settings ---
market_data:
  # Maximum rows to read per tab (0 = read all rows).
  max_rows_per_tab: 250

  # Maximum columns to read per tab (0 = read all columns).
  max_cols_per_tab: 20

  # Minimum keyword hits for a tab to be included as a market data candidate.
  # Increase to be more selective; set include_all_tabs: true to bypass.
  keyword_threshold: 2

  # If true, send all tabs regardless of keyword score.
  # Use when the workbook has unusual tab names.
  include_all_tabs: false
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest test_config.py::test_market_data_config_defaults -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config.yaml test_config.py
git commit -m "feat: add market_data config section"
```

---

## Task 3: Dynamic Workbook Extraction

**Files:**
- Modify: `memo_automator.py` (lines 377–523, `extract_market_data`)
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_market_data.py`:

```python
class TestDynamicExtractMarketData:
    """Tests for the new keyword-scoring extractor."""

    def _make_workbook(self, sheets: dict) -> str:
        """Create a temp xlsx with given {tab_name: [[row], [row]]} data."""
        import openpyxl, tempfile, os
        wb = openpyxl.Workbook()
        first = True
        for name, rows in sheets.items():
            if first:
                ws = wb.active
                ws.title = name
                first = False
            else:
                ws = wb.create_sheet(name)
            for row in rows:
                ws.append(row)
        path = tempfile.mktemp(suffix=".xlsx")
        wb.save(path)
        return path

    def test_scores_rent_tab_above_threshold(self):
        path = self._make_workbook({
            "Rent Comparison": [["Market", "Rent", "Occupancy"], ["A", 1200, 0.95]],
            "Backend Raw": [["id", "code"], [1, "X"]],
        })
        cfg = {"market_data": {"max_rows_per_tab": 50, "max_cols_per_tab": 10, "keyword_threshold": 2, "include_all_tabs": False}}
        result = extract_market_data(path, cfg)
        assert "Rent Comparison" in result
        assert "Backend Raw" not in result
        os.remove(path)

    def test_include_all_tabs_bypasses_scoring(self):
        path = self._make_workbook({
            "XYZ": [["col1", "col2"], ["a", "b"]],
        })
        cfg = {"market_data": {"max_rows_per_tab": 50, "max_cols_per_tab": 10, "keyword_threshold": 2, "include_all_tabs": True}}
        result = extract_market_data(path, cfg)
        assert "XYZ" in result
        os.remove(path)

    def test_falls_back_to_proforma_config_if_no_market_data_section(self):
        path = self._make_workbook({
            "Comp Set": [["Name", "Rent"], ["A", 1200]],
        })
        cfg = {"proforma": {"max_rows_per_tab": 50, "max_cols_per_tab": 10},
               "market_data": {"max_rows_per_tab": 50, "max_cols_per_tab": 10, "keyword_threshold": 1, "include_all_tabs": False}}
        result = extract_market_data(path, cfg)
        assert "Comp Set" in result
        os.remove(path)

    def test_missing_file_returns_empty(self):
        cfg = {"market_data": {"max_rows_per_tab": 50, "max_cols_per_tab": 10, "keyword_threshold": 2, "include_all_tabs": False}}
        assert extract_market_data("/no/such/file.xlsx", cfg) == ""

    def test_tab_header_line_present(self):
        path = self._make_workbook({
            "Occupancy Trend": [["Year", "Occupancy"], [2023, 0.94], [2024, 0.96]],
        })
        cfg = {"market_data": {"max_rows_per_tab": 50, "max_cols_per_tab": 10, "keyword_threshold": 1, "include_all_tabs": False}}
        result = extract_market_data(path, cfg)
        assert "TAB: Occupancy Trend" in result
        os.remove(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_market_data.py::TestDynamicExtractMarketData -v
```

Expected: FAIL — current function uses hardcoded tabs, not keyword scoring.

- [ ] **Step 3: Replace `extract_market_data()` in `memo_automator.py`**

Delete the `_MARKET_DASHBOARD_TABS` constant (lines 377–384) and replace the entire `extract_market_data` function (lines 387–523) with:

```python
# Market keyword set used to score tabs for relevance.
_MARKET_KEYWORDS = frozenset({
    "rent", "occupancy", "comp", "supply", "demand", "absorption",
    "vacancy", "cap rate", "market rate", "pipeline", "lease-up",
    "submarket", "msa", "psf", "concession", "effective rent",
    "gross rent", "net rent", "growth", "prelease",
})


def _score_tab_for_market_relevance(sheet_name: str, header_cells: list) -> int:
    """Count how many market keywords appear in the tab name + column headers."""
    text = (sheet_name + " " + " ".join(str(h) for h in header_cells if h)).lower()
    return sum(1 for kw in _MARKET_KEYWORDS if kw in text)


def extract_market_data(market_data_path: str, cfg: dict) -> str:
    """
    Read any Excel market workbook and return a compact text representation
    of market-relevant tabs.

    Tab relevance is determined by keyword scoring against the tab name and
    column headers. Tabs scoring at or above ``market_data.keyword_threshold``
    (default 2) are included. Set ``market_data.include_all_tabs: true`` to
    bypass scoring and include all tabs.

    Returns empty string if the file is missing, unreadable, or contains no
    relevant tabs.
    """
    md_cfg = cfg.get("market_data", {})
    pf_cfg = cfg.get("proforma", {})
    max_rows = md_cfg.get("max_rows_per_tab", pf_cfg.get("max_rows_per_tab", 250))
    max_cols = md_cfg.get("max_cols_per_tab", pf_cfg.get("max_cols_per_tab", 20))
    threshold = md_cfg.get("keyword_threshold", 2)
    include_all = md_cfg.get("include_all_tabs", False)

    log.info("Opening market data workbook (data_only): %s", market_data_path)
    try:
        wb = openpyxl.load_workbook(market_data_path, data_only=True)
    except (InvalidFileException, zipfile.BadZipFile) as e:
        log.warning(
            "Unable to open market data '%s': %s. Continuing without market data.",
            market_data_path, e,
        )
        return ""
    except FileNotFoundError:
        log.warning("Market data file not found: %s", market_data_path)
        return ""

    log.info("Market workbook sheets: %s", wb.sheetnames)
    sections = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        end_row = ws.max_row if max_rows == 0 else min(ws.max_row or 0, max_rows)
        end_col = ws.max_column if max_cols == 0 else min(ws.max_column or 0, max_cols)

        if end_row == 0 or end_col == 0:
            continue

        # Read first row as headers for scoring
        header_row = next(
            ws.iter_rows(min_row=1, max_row=1, max_col=end_col, values_only=True),
            (),
        )
        headers = [str(c) if c is not None else "" for c in header_row]

        score = _score_tab_for_market_relevance(sheet_name, headers)
        if not include_all and score < threshold:
            log.debug("Skipping tab '%s' (score=%d < threshold=%d)", sheet_name, score, threshold)
            continue

        log.info("Including tab '%s' (score=%d)", sheet_name, score)
        lines = [
            f"\n{'=' * 70}",
            f"TAB: {sheet_name}",
            f"{'=' * 70}",
        ]
        for row in ws.iter_rows(min_row=1, max_row=end_row, max_col=end_col, values_only=False):
            row_data = [str(cell.value) for cell in row if cell.value is not None]
            if row_data:
                lines.append(f"Row {row[0].row}:\t" + "\t".join(row_data))

        sections.append("\n".join(lines))

    wb.close()

    if not sections:
        log.warning("No market-relevant tabs found in '%s'", market_data_path)
        return ""

    header = [
        f"\n{'=' * 70}",
        "MARKET DATA (from workbook)",
        f"{'=' * 70}",
    ]
    return "\n".join(header) + "\n" + "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_market_data.py::TestDynamicExtractMarketData -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Run existing extraction tests to confirm no regression**

```bash
python -m pytest tests/test_market_data.py -v
```

Expected: All previously passing tests still PASS. (Note: `TestExtractMarketData` tests using hardcoded RealPage tabs may now pass via keyword scoring — `"Tables"`, `"Comp Set"` will score ≥ 2 on `comp`, `"Rent Growth Comparison By Year"` on `rent` + `growth`.)

- [ ] **Step 6: Commit**

```bash
git add memo_automator.py tests/test_market_data.py
git commit -m "feat: replace hardcoded tab extraction with keyword-scoring dynamic extractor"
```

---

## Task 4: Write Market Data Prompts

**Files:**
- Create: `prompts/market_mapping_v1.txt`
- Create: `prompts/market_validation_v1.txt`

No unit tests — prompt content is validated by integration tests in Task 5.

- [ ] **Step 1: Create `prompts/market_mapping_v1.txt`**

```
You are an expert real estate analyst. You are updating an investment committee memo with current market data from a market workbook.

{source_directives_section}

## YOUR TASK

You will receive:
1. The full memo content (all slides, with page numbers, charts, tables, and text)
2. Market workbook data (tabs extracted from the workbook, with tab names and rows)

Complete these steps IN ORDER:

### Step 1 — Label each workbook tab
Before anything else, label each tab with its semantic category. Examples:
- "rent comps", "occupancy trend", "supply pipeline", "comp set", "submarket stats",
  "rent growth by year", "absorption rate", "market summary"

### Step 2 — Identify market metrics in the memo
List every chart, table, and narrative section that references market data
(rent growth, occupancy rates, comp properties, supply/demand, absorption, etc.)
with its page number and shape name.

### Step 3 — Match and generate updates
For each memo metric, find the best matching workbook tab using semantic similarity.
Tab and chart names do NOT need to match exactly. Examples of valid matches:
- "Florida" ≈ "UF" (abbreviation)
- "Rent Growth by Market" ≈ "Rent Growth Comparison By Year" (synonym)
- "Effective Rent" ≈ "Market Rate Rent" (equivalent concept)

For each match where the workbook data differs from the memo:
- Generate updates for ALL pages where related content appears. A rent chart on
  slide 5, a summary table on slide 3, and narrative on slide 7 are all updated.
- If a chart doesn't match 1:1 (e.g., different markets, different time range),
  ADAPT it: update categories, add/remove series as needed, then rewrite the
  narrative on any slide that references that chart to stay coherent.

IMPORTANT RULES:
- USER DIRECTIONS (if provided above) are HIGHEST PRIORITY. Follow them exactly.
  They override your judgment about what to update.
- Only generate updates where the data actually differs from the memo.
- Include a "reasoning" field on every update explaining the match and any adaptation.
- If a match is uncertain, include the update but set "confidence": "low".

## MEMO CONTENT (all slides)
{memo_content}

## MARKET WORKBOOK DATA
{market_data}

## OUTPUT FORMAT

Return a single JSON object. Do NOT wrap in markdown fences.

{{
  "market_data_updates": [
    {{
      "type": "chart_update",
      "page": <int, 1-based slide number>,
      "chart_name": "<shape name or null>",
      "chart_title": "<chart title or null>",
      "series": [
        {{"name": "<series name>", "new_values": [<numbers>], "old_values": [<numbers>]}}
      ],
      "categories": ["<x-axis label>", ...],
      "add_series": [
        {{"name": "<series name>", "values": [<numbers>]}}
      ],
      "remove_series": ["<series name to remove>"],
      "source": "<workbook tab name>",
      "reasoning": "<explanation of match and any adaptation>",
      "confidence": "high" | "medium" | "low"
    }},
    {{
      "type": "narrative_update",
      "page": <int, 1-based>,
      "old_text": "<exact text currently in the memo to replace>",
      "new_text": "<replacement text>",
      "source": "<workbook tab name>",
      "reasoning": "<explanation>",
      "confidence": "high" | "medium" | "low"
    }},
    {{
      "type": "table_update",
      "page": <int, 1-based>,
      "slide_table": "<table name or descriptive label>",
      "updates": [
        {{"row": <int, 0-based>, "col": <int, 0-based>, "old_value": "<text>", "new_value": "<text>"}}
      ],
      "source": "<workbook tab name>",
      "reasoning": "<explanation>",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "unmatched_memo_metrics": ["<description of memo metric with no workbook match>"],
  "unmatched_workbook_tabs": ["<tab name that had no memo match>"]
}}
```

- [ ] **Step 2: Create `prompts/market_validation_v1.txt`**

```
You are a quality assurance analyst reviewing proposed market data updates for an investment committee memo.

{source_directives_section}

## YOUR TASK

Review each proposed update and either keep it, correct it, or drop it based on the rules below.

### DROP these updates entirely:
- Matches where the reasoning is vague or the "confidence" is "low" with no clear justification
- Numeric values outside plausible real estate ranges:
  - Monthly rents: must be > $0 and < $10,000/unit/month
  - Occupancy rate: must be between 0 and 1 (or 0% and 100%)
  - Cap rates: must be between 0.02 and 0.15 (2%–15%)
  - Growth rates: must be between -0.5 and 2.0 (-50% to +200%)

### ADD to warnings (keep the update, but record the issue):
- Narrative that contradicts the chart it references (e.g., narrative says "improving"
  but updated chart values show year-over-year decline)
- A chart was updated on one page, but narrative on a different page still references
  the old figures (missed cross-slide update)
- Any update that appears to violate the user directions provided above

## ORIGINAL MEMO CONTENT
{memo_content}

## PROPOSED MARKET DATA UPDATES
{proposed_updates}

## OUTPUT FORMAT

Return a single JSON object. Do NOT wrap in markdown fences.

{{
  "market_data_updates": [<cleaned list — same schema as input, with dropped entries removed>],
  "unmatched_memo_metrics": [<pass through from mapping output>],
  "unmatched_workbook_tabs": [<pass through from mapping output>],
  "warnings": ["<warning description>"]
}}
```

- [ ] **Step 3: Verify prompt files load correctly**

```bash
python -c "
from memo_automator import _load_prompt_template
t = _load_prompt_template('market_mapping_v1.txt')
assert '{memo_content}' in t
assert '{market_data}' in t
print('market_mapping_v1.txt OK, len=', len(t))
t2 = _load_prompt_template('market_validation_v1.txt')
assert '{proposed_updates}' in t2
print('market_validation_v1.txt OK, len=', len(t2))
"
```

Expected: Both files print OK with non-zero lengths.

- [ ] **Step 4: Commit**

```bash
git add prompts/market_mapping_v1.txt prompts/market_validation_v1.txt
git commit -m "feat: add market data mapping and validation prompts"
```

---

## Task 5: Market Data API Functions

**Files:**
- Modify: `memo_automator.py`
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_market_data.py`:

```python
from unittest.mock import MagicMock, patch


class TestGetMarketDataMappings:
    """Tests for get_market_data_mappings() — mocks Anthropic."""

    def _make_client(self, response_text: str):
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text=response_text)]
        msg.stop_reason = "end_turn"
        msg.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        client.messages.stream.return_value.__enter__ = lambda s, *a: s
        client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)
        client.messages.stream.return_value.get_final_message = lambda: msg
        return client

    def test_returns_update_set_on_valid_json(self):
        from memo_automator import get_market_data_mappings
        response = '{"market_data_updates":[],"unmatched_memo_metrics":[],"unmatched_workbook_tabs":[]}'
        client = self._make_client(response)
        cfg = {"claude": {"model": "claude-sonnet-4-6", "max_tokens": 8192, "temperature": 0}}
        result = get_market_data_mappings(client, "market text", "memo text", cfg)
        assert result["market_data_updates"] == []

    def test_returns_empty_on_empty_market_data(self):
        from memo_automator import get_market_data_mappings
        cfg = {"claude": {"model": "claude-sonnet-4-6", "max_tokens": 8192, "temperature": 0}}
        client = MagicMock()
        result = get_market_data_mappings(client, "", "memo text", cfg)
        assert result == {"market_data_updates": [], "unmatched_memo_metrics": [], "unmatched_workbook_tabs": [], "warnings": []}


class TestValidateMarketDataMappings:
    def test_returns_cleaned_update_set(self):
        from memo_automator import validate_market_data_mappings
        from unittest.mock import MagicMock
        import json
        update_set = {"market_data_updates": [], "unmatched_memo_metrics": [], "unmatched_workbook_tabs": [], "warnings": []}
        response = json.dumps(update_set)
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text=response)]
        msg.stop_reason = "end_turn"
        msg.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        client.messages.stream.return_value.__enter__ = lambda s, *a: s
        client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)
        client.messages.stream.return_value.get_final_message = lambda: msg
        cfg = {"claude": {"model": "claude-sonnet-4-6", "validation_model": "claude-sonnet-4-6", "max_tokens": 8192, "temperature": 0}}
        result = validate_market_data_mappings(client, update_set, "memo text", cfg)
        assert "market_data_updates" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_market_data.py::TestGetMarketDataMappings tests/test_market_data.py::TestValidateMarketDataMappings -v
```

Expected: `ImportError` — functions don't exist yet.

- [ ] **Step 3: Add functions to `memo_automator.py`**

Find the line `VALIDATION_PROMPT = _load_prompt_template("validation_v1.txt")` (around line 1486). Add after it, in a new section:

```python
# ============================================================================
# 7b. CLAUDE API - MARKET DATA MAPPING & VALIDATION
# ============================================================================
MARKET_MAPPING_PROMPT = _load_prompt_template("market_mapping_v1.txt")
MARKET_VALIDATION_PROMPT = _load_prompt_template("market_validation_v1.txt")

_EMPTY_MARKET_UPDATE_SET = {
    "market_data_updates": [],
    "unmatched_memo_metrics": [],
    "unmatched_workbook_tabs": [],
    "warnings": [],
}


def get_market_data_mappings(
    client,
    market_data: str,
    memo_content: str,
    cfg: dict,
    source_directives: list[dict] | None = None,
) -> dict:
    """
    Send memo content + market workbook data to Claude and receive a structured
    JSON describing all market data updates (chart, narrative, table).

    Returns an empty update set if market_data is blank (no-op).
    """
    if not market_data.strip():
        return dict(_EMPTY_MARKET_UPDATE_SET)

    model = cfg["claude"]["model"]
    max_tokens = cfg["claude"]["max_tokens"]
    temperature = cfg["claude"]["temperature"]

    directives_section = format_source_directives(
        [d for d in (source_directives or []) if d.get("source_type") == "market_data"],
        scope="mapping",
    )

    system_text = MARKET_MAPPING_PROMPT.format(
        source_directives_section=directives_section,
        memo_content="(see user message below)",
        market_data=market_data,
    )
    user_text = f"## Memo Content (all slides)\n{memo_content}"

    log.info(
        "Market mapping: calling Claude (model=%s, prompt=%d chars)",
        model, len(system_text) + len(user_text),
    )

    message = _create_message(
        client,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_text}],
    )

    raw = next((b.text for b in message.content if b.type == "text"), "")
    log.info("Market mapping response: %d chars, stop_reason=%s", len(raw), message.stop_reason)

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        result.setdefault("warnings", [])
        return result
    except json.JSONDecodeError as e:
        log.warning("Market mapping JSON parse failed: %s. Raw: %s", e, raw[:200])
        return dict(_EMPTY_MARKET_UPDATE_SET)


def validate_market_data_mappings(
    client,
    update_set: dict,
    memo_content: str,
    cfg: dict,
    source_directives: list[dict] | None = None,
) -> dict:
    """
    QA pass for market data updates. Returns a cleaned update set with
    uncertain matches dropped and a warnings list added.
    """
    if not update_set.get("market_data_updates"):
        return update_set

    model = cfg["claude"].get("validation_model", cfg["claude"]["model"])
    max_tokens = cfg["claude"]["max_tokens"]
    temperature = cfg["claude"]["temperature"]

    directives_section = format_source_directives(
        [d for d in (source_directives or []) if d.get("source_type") == "market_data"],
        scope="mapping",
    )

    system_text = MARKET_VALIDATION_PROMPT.format(
        source_directives_section=directives_section,
        memo_content=memo_content,
        proposed_updates="(see user message below)",
    )
    user_text = "## Proposed Market Data Updates\n" + json.dumps(update_set, indent=2)

    log.info(
        "Market validation: calling Claude (model=%s, prompt=%d chars)",
        model, len(system_text) + len(user_text),
    )

    message = _create_message(
        client,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_text}],
    )

    raw = next((b.text for b in message.content if b.type == "text"), "")
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        result.setdefault("warnings", [])
        return result
    except json.JSONDecodeError as e:
        log.warning("Market validation JSON parse failed: %s. Returning original.", e)
        return update_set
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_market_data.py::TestGetMarketDataMappings tests/test_market_data.py::TestValidateMarketDataMappings -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add memo_automator.py
git commit -m "feat: add get_market_data_mappings and validate_market_data_mappings"
```

---

## Task 6: Enhanced Chart Apply Layer

**Files:**
- Modify: `memo_automator.py` (extend `_apply_chart_updates`, add `apply_market_updates`)
- Test: `tests/stress_test_market_data.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/stress_test_market_data.py`:

```python
class TestApplyMarketUpdates:
    """Tests for apply_market_updates() rich update schema."""

    def _make_simple_pptx_with_chart(self, tmp_path) -> str:
        """Create a minimal PPTX with one chart slide."""
        from pptx import Presentation
        from pptx.util import Inches
        from pptx.chart.data import ChartData
        from pptx import chart as pptx_chart
        from pptx.enum.chart import XL_CHART_TYPE

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])

        chart_data = ChartData()
        chart_data.categories = ["2022", "2023", "2024"]
        chart_data.add_series("Market A", (1100, 1250, 1400))
        chart_data.add_series("Market B", (900, 950, 1000))

        chart_shape = slide.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(6), Inches(4), chart_data
        )
        chart_shape.name = "Rent Growth"

        path = str(tmp_path / "test.pptx")
        prs.save(path)
        return path

    def test_chart_update_values(self, tmp_path):
        from memo_automator import apply_market_updates
        path = self._make_simple_pptx_with_chart(tmp_path)
        update_set = {
            "market_data_updates": [{
                "type": "chart_update",
                "page": 1,
                "chart_name": "Rent Growth",
                "series": [{"name": "Market A", "new_values": [1200, 1350, 1500], "old_values": [1100, 1250, 1400]}],
                "categories": None,
                "add_series": [],
                "remove_series": [],
                "source": "Rent tab",
                "reasoning": "Updated",
                "confidence": "high",
            }],
            "unmatched_memo_metrics": [],
            "unmatched_workbook_tabs": [],
            "warnings": [],
        }
        changes = apply_market_updates(path, update_set, dry_run=False)
        assert len(changes) == 1
        assert changes[0]["type"] == "chart"

    def test_chart_update_categories(self, tmp_path):
        from memo_automator import apply_market_updates
        path = self._make_simple_pptx_with_chart(tmp_path)
        update_set = {
            "market_data_updates": [{
                "type": "chart_update",
                "page": 1,
                "chart_name": "Rent Growth",
                "series": [],
                "categories": ["2023", "2024", "2025"],
                "add_series": [],
                "remove_series": [],
                "source": "Rent tab",
                "reasoning": "New year range",
                "confidence": "high",
            }],
            "unmatched_memo_metrics": [],
            "unmatched_workbook_tabs": [],
            "warnings": [],
        }
        changes = apply_market_updates(path, update_set, dry_run=False)
        assert len(changes) >= 1

    def test_narrative_update(self, tmp_path):
        from pptx import Presentation
        from pptx.util import Inches
        from memo_automator import apply_market_updates

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        txbox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(2))
        txbox.text_frame.text = "Rents grew 5% year over year"
        path = str(tmp_path / "narr.pptx")
        prs.save(path)

        update_set = {
            "market_data_updates": [{
                "type": "narrative_update",
                "page": 1,
                "old_text": "Rents grew 5% year over year",
                "new_text": "Rents grew 12% year over year",
                "source": "Rent tab",
                "reasoning": "Updated figure",
                "confidence": "high",
            }],
            "unmatched_memo_metrics": [],
            "unmatched_workbook_tabs": [],
            "warnings": [],
        }
        changes = apply_market_updates(path, update_set, dry_run=False)
        assert len(changes) == 1
        assert changes[0]["type"] == "narrative"

    def test_empty_update_set_no_changes(self, tmp_path):
        from memo_automator import apply_market_updates
        path = self._make_simple_pptx_with_chart(tmp_path)
        changes = apply_market_updates(path, {"market_data_updates": [], "unmatched_memo_metrics": [], "unmatched_workbook_tabs": [], "warnings": []}, dry_run=False)
        assert changes == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/stress_test_market_data.py::TestApplyMarketUpdates -v
```

Expected: `ImportError` — `apply_market_updates` doesn't exist yet.

- [ ] **Step 3: Add `_apply_market_chart_update()` and `apply_market_updates()` to `memo_automator.py`**

Find `_apply_chart_updates()` (line ~3025). Add these two new functions AFTER `_apply_chart_updates` ends (after line ~3150):

```python
def _apply_market_chart_update(prs, update: dict) -> list[dict]:
    """
    Apply a single market data chart update dict to an open Presentation object.

    Handles: series value updates, category (x-axis) label updates,
    add_series, remove_series. Returns list of change records.
    """
    from lxml import etree

    page = update.get("page", 1)
    chart_name = update.get("chart_name") or ""
    chart_title = update.get("chart_title") or ""
    source = update.get("source", "")
    changes = []

    _C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    ns = {"c": _C}

    try:
        slide = prs.slides[page - 1]
    except IndexError:
        log.warning("Market chart update SKIPPED: page %d does not exist", page)
        return []

    # Find target chart (same logic as existing _apply_chart_updates)
    target_chart = None
    target_shape = None
    for shape in slide.shapes:
        if not shape.has_chart:
            continue
        name_match = chart_name and _loose_match(chart_name, shape.name)
        title_match = False
        if shape.chart.has_title and shape.chart.chart_title:
            try:
                ct = shape.chart.chart_title.text_frame.text.strip()
                title_match = chart_title and _loose_match(chart_title, ct)
            except Exception:
                pass
        if name_match or title_match:
            target_chart = shape.chart
            target_shape = shape
            break

    if target_chart is None:
        chart_shapes = [s for s in slide.shapes if s.has_chart]
        if len(chart_shapes) == 1:
            target_chart = chart_shapes[0].chart
            target_shape = chart_shapes[0]
        else:
            log.warning("Market chart NOT FOUND: page=%d name='%s'", page, chart_name)
            return []

    chart_el = target_chart._element

    # 1. Update series values
    for ser_upd in update.get("series", []):
        s_name = ser_upd.get("name", "")
        new_values = ser_upd.get("new_values", [])
        old_values = ser_upd.get("old_values", [])
        for ser_el in chart_el.findall(f".//{{{_C}}}ser"):
            tx_vals = ser_el.findall(f".//{{{_C}}}tx//{{{_C}}}v")
            ser_label = tx_vals[0].text if tx_vals else ""
            if not _loose_match(s_name, ser_label):
                continue
            num_cache = ser_el.find(f".//{{{_C}}}numRef/{{{_C}}}numCache", ns)
            if num_cache is None:
                num_cache = ser_el.find(f".//{{{_C}}}numLit", ns)
            if num_cache is None:
                continue
            pts = num_cache.findall(f"{{{_C}}}pt", ns)
            for i, pt in enumerate(pts):
                if i < len(new_values):
                    v_el = pt.find(f"{{{_C}}}v", ns)
                    if v_el is not None:
                        v_el.text = str(new_values[i])
            changes.append({
                "page": page, "type": "chart",
                "location": f"{target_shape.name} / series '{s_name}'",
                "old": str(old_values[:3]),
                "new": str(new_values[:3]),
                "source": source,
            })
            break

    # 2. Update categories (x-axis labels)
    new_cats = update.get("categories")
    if new_cats:
        for ser_el in chart_el.findall(f".//{{{_C}}}ser"):
            str_cache = ser_el.find(f".//{{{_C}}}cat//{{{_C}}}strCache", ns)
            if str_cache is None:
                continue
            for pt in str_cache.findall(f"{{{_C}}}pt", ns):
                str_cache.remove(pt)
            pt_count = str_cache.find(f"{{{_C}}}ptCount", ns)
            if pt_count is not None:
                pt_count.set("val", str(len(new_cats)))
            for idx, cat in enumerate(new_cats):
                pt_el = etree.SubElement(str_cache, f"{{{_C}}}pt")
                pt_el.set("idx", str(idx))
                v_el = etree.SubElement(pt_el, f"{{{_C}}}v")
                v_el.text = str(cat)
            changes.append({
                "page": page, "type": "chart",
                "location": f"{target_shape.name} / categories",
                "old": "previous categories",
                "new": str(new_cats[:5]),
                "source": source,
            })
            break  # categories shared across series; only patch once

    # 3. Add new series (clone last existing series)
    import copy
    existing_sers = chart_el.findall(f".//{{{_C}}}ser")
    for add_ser in update.get("add_series", []):
        if not existing_sers:
            log.warning("Cannot add series — no existing series to clone on page %d", page)
            continue
        new_ser = copy.deepcopy(existing_sers[-1])
        new_idx = len(existing_sers)
        for el in new_ser.findall(f"{{{_C}}}idx"):
            el.set("val", str(new_idx))
        for el in new_ser.findall(f"{{{_C}}}order"):
            el.set("val", str(new_idx))
        # Set series name
        for v_el in new_ser.findall(f".//{{{_C}}}tx//{{{_C}}}v"):
            v_el.text = add_ser.get("name", f"Series {new_idx}")
        # Set values
        num_cache = new_ser.find(f".//{{{_C}}}numRef/{{{_C}}}numCache", ns)
        if num_cache is None:
            num_cache = new_ser.find(f".//{{{_C}}}numLit", ns)
        if num_cache is not None:
            for pt in num_cache.findall(f"{{{_C}}}pt", ns):
                num_cache.remove(pt)
            vals = add_ser.get("values", [])
            pt_count = num_cache.find(f"{{{_C}}}ptCount", ns)
            if pt_count is not None:
                pt_count.set("val", str(len(vals)))
            for i, val in enumerate(vals):
                pt_el = etree.SubElement(num_cache, f"{{{_C}}}pt")
                pt_el.set("idx", str(i))
                v_el = etree.SubElement(pt_el, f"{{{_C}}}v")
                v_el.text = str(val) if val is not None else ""
        parent = existing_sers[-1].getparent()
        parent.insert(list(parent).index(existing_sers[-1]) + 1, new_ser)
        existing_sers = chart_el.findall(f".//{{{_C}}}ser")  # refresh
        changes.append({
            "page": page, "type": "chart",
            "location": f"{target_shape.name} / add series '{add_ser.get('name')}'",
            "old": "", "new": str(add_ser.get("values", [])[:3]),
            "source": source,
        })

    # 4. Remove series
    for remove_name in update.get("remove_series", []):
        for ser_el in chart_el.findall(f".//{{{_C}}}ser"):
            tx_vals = ser_el.findall(f".//{{{_C}}}tx//{{{_C}}}v")
            ser_label = tx_vals[0].text if tx_vals else ""
            if _loose_match(remove_name, ser_label):
                ser_el.getparent().remove(ser_el)
                changes.append({
                    "page": page, "type": "chart",
                    "location": f"{target_shape.name} / remove series '{remove_name}'",
                    "old": remove_name, "new": "",
                    "source": source,
                })
                break

    return changes


def apply_market_updates(memo_path: str, update_set: dict, dry_run: bool = False) -> list[dict]:
    """
    Apply the full MarketDataUpdateSet to a PPTX file.

    Handles chart_update (values, categories, add/remove series),
    narrative_update (text replacement), and table_update (cell patching).
    Saves the file in-place unless dry_run is True.

    Returns a list of change records.
    """
    updates = update_set.get("market_data_updates", [])
    if not updates:
        return []

    prs = _load_presentation(memo_path)
    all_changes = []

    # Group by type for logging
    chart_upds = [u for u in updates if u.get("type") == "chart_update"]
    narr_upds = [u for u in updates if u.get("type") == "narrative_update"]
    tbl_upds = [u for u in updates if u.get("type") == "table_update"]

    log.info(
        "Applying market updates: %d chart, %d narrative, %d table",
        len(chart_upds), len(narr_upds), len(tbl_upds),
    )

    # Chart updates
    for upd in chart_upds:
        if dry_run:
            all_changes.append({
                "page": upd.get("page"), "type": "chart",
                "location": f"{upd.get('chart_name')} (dry run)",
                "old": "", "new": "", "source": upd.get("source", ""),
            })
        else:
            all_changes.extend(_apply_market_chart_update(prs, upd))

    # Narrative updates — reuse _replace_in_para across all text frames
    for upd in narr_upds:
        page = upd.get("page", 1)
        old_text = upd.get("old_text", "")
        new_text = upd.get("new_text", "")
        if not old_text or old_text == new_text:
            continue
        try:
            slide = prs.slides[page - 1]
        except IndexError:
            log.warning("Narrative update SKIPPED: page %d does not exist", page)
            continue
        replaced = False
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                if _replace_in_para(para, old_text, new_text):
                    replaced = True
                    break
            if replaced:
                break
        if replaced:
            all_changes.append({
                "page": page, "type": "narrative",
                "location": f"slide {page} narrative",
                "old": old_text[:80], "new": new_text[:80],
                "source": upd.get("source", ""),
            })
        else:
            log.warning("Narrative update NOT APPLIED on page %d: '%s'", page, old_text[:60])

    # Table updates — cell-based patching
    for upd in tbl_upds:
        page = upd.get("page", 1)
        slide_table = upd.get("slide_table", "")
        try:
            slide = prs.slides[page - 1]
        except IndexError:
            log.warning("Table update SKIPPED: page %d does not exist", page)
            continue
        for shape in slide.shapes:
            if not shape.has_table:
                continue
            if slide_table and not _loose_match(slide_table, shape.name):
                continue
            tbl = shape.table
            for cell_upd in upd.get("updates", []):
                r, c = cell_upd.get("row", 0), cell_upd.get("col", 0)
                old_val = cell_upd.get("old_value", "")
                new_val = cell_upd.get("new_value", "")
                if r >= len(tbl.rows) or c >= len(tbl.columns):
                    continue
                cell = tbl.cell(r, c)
                for para in cell.text_frame.paragraphs:
                    if _replace_in_para(para, old_val, new_val):
                        all_changes.append({
                            "page": page, "type": "table",
                            "location": f"{shape.name} [{r},{c}]",
                            "old": old_val, "new": new_val,
                            "source": upd.get("source", ""),
                        })
                        break

    if not dry_run and all_changes:
        prs.save(memo_path)
        log.info("Market updates saved: %d changes", len(all_changes))

    return all_changes
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/stress_test_market_data.py::TestApplyMarketUpdates -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Run full stress test suite to confirm no regression**

```bash
python -m pytest tests/stress_test_market_data.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add memo_automator.py tests/stress_test_market_data.py
git commit -m "feat: add _apply_market_chart_update and apply_market_updates with full schema support"
```

---

## Task 7: Pipeline Integration

**Files:**
- Modify: `memo_chef/pipeline.py`
- Test: `test_pipeline_integration_mocked.py`

- [ ] **Step 1: Write the failing tests**

First, check whether `test_pipeline_integration_mocked.py` already defines a mock Anthropic client helper. Search for `_make_mock_anthropic_client` or `mock_client` in that file. If one exists, use it. If not, add this helper at the top of the test class:

```python
def _make_mock_anthropic_client():
    """Minimal Anthropic client stub: always returns empty mapping JSON."""
    import json
    from unittest.mock import MagicMock
    client = MagicMock()
    empty = {"table_updates": [], "text_updates": [], "row_inserts": [], "narrative_updates": [], "table_structure_updates": []}
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text=json.dumps(empty))]
    msg.stop_reason = "end_turn"
    msg.usage = MagicMock(input_tokens=10, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    client.messages.stream.return_value.__enter__ = lambda s, *a: s
    client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)
    client.messages.stream.return_value.get_final_message = lambda: msg
    return client
```

Then add the test class:

```python
class TestMarketDataPipelineStages:
    """Market data stages are skipped when no market_data_path provided."""

    def test_market_stages_skipped_without_path(self, tmp_path, monkeypatch):
        """Pipeline runs to completion without market data; no market stages in manifest."""
        from memo_chef.models import RunRequest
        from pptx import Presentation
        import openpyxl

        memo = tmp_path / "memo.pptx"
        proforma = tmp_path / "proforma.xlsx"
        Presentation().save(str(memo))
        wb = openpyxl.Workbook()
        wb.active.title = "Executive Summary"
        wb.active.append(["Item", "Value"])
        wb.save(str(proforma))

        request = RunRequest(
            memo_path=str(memo),
            proforma_path=str(proforma),
            output_dir=str(tmp_path / "out"),
            api_key="test-key",
            config_path="config.yaml",
            run_id="test-mkt-skip",
            market_data_path=None,
            dry_run=True,
            skip_validation=True,
        )

        import anthropic
        monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: _make_mock_anthropic_client())

        from memo_chef.pipeline import run_memo_pipeline
        result = run_memo_pipeline(request)

        assert "market_data_mapping" not in result.manifest.stages
        assert "market_data_validation" not in result.manifest.stages
        assert "apply_market_updates" not in result.manifest.stages
```

- [ ] **Step 2: Run test to verify it fails (or passes — if it does, skip to Step 4)**

```bash
python -m pytest test_pipeline_integration_mocked.py::TestMarketDataPipelineStages -v
```

- [ ] **Step 3: Modify `memo_chef/pipeline.py`**

**3a. Add imports at top of pipeline.py** — add to the existing import block from `memo_automator`:

```python
from memo_automator import (
    ...
    apply_market_updates,           # ADD
    get_market_data_mappings,       # ADD
    validate_market_data_mappings,  # ADD
    ...
)
```

**3b. Separate market data from proforma in `extract_sources` stage**

Find this block inside `run_memo_pipeline` (around line 649):

```python
            if request.market_data_path:
                market_data = extract_market_data(request.market_data_path, cfg)
                if market_data:
                    proforma_data += "\n\n" + market_data   # ← REMOVE THIS LINE
                    market_extract_path = ...
```

Change to — extract and save to file but DO NOT concatenate into `proforma_data`:

```python
            market_data_text = ""
            if request.market_data_path:
                market_data_text = extract_market_data(request.market_data_path, cfg)
                if market_data_text:
                    market_extract_path = os.path.join(request.output_dir, "market_data_extract.txt")
                    Path(market_extract_path).write_text(market_data_text, encoding="utf-8")
                    checkpoint.set_output("market_data_extract", market_extract_path)
                else:
                    checkpoint.add_warning(
                        "extract_sources",
                        "Market data file loaded but no relevant tabs were extracted.",
                    )
```

**3c. Add market data stages after the `apply` stage**

Find the `apply` stage block (around line 759) and add IMMEDIATELY after `checkpoint.set_count("changes", len(changes))`:

```python
        # --- Market data pipeline stages (4-6) ---
        # Skipped automatically when no market_data_path is provided.
        if request.market_data_path and market_data_text and not request.dry_run:
            # Stage 4: Market data mapping
            _emit(callback, "market_data_mapping", "Map market data", 87)
            with checkpoint.stage("market_data_mapping", "Claude mapping market workbook to memo"):
                market_update_set = _retry(
                    get_market_data_mappings,
                    client,
                    market_data_text,
                    memo_content,
                    cfg,
                    source_directives=directives_dicts,
                    checkpoint=checkpoint,
                    stage="market_data_mapping",
                    callback=callback,
                    retry_percent=87,
                )
                raw_market_path = os.path.join(request.output_dir, "market_mappings_raw.json")
                _write_json(raw_market_path, market_update_set)
                checkpoint.set_output("market_mappings_raw", raw_market_path)
                checkpoint.set_count(
                    "market_updates_proposed",
                    len(market_update_set.get("market_data_updates", [])),
                )

            # Stage 5: Market data validation
            _emit(callback, "market_data_validation", "Validate market updates", 89)
            with checkpoint.stage("market_data_validation", "QA market data updates"):
                if request.skip_validation:
                    validated_market = market_update_set
                else:
                    validated_market = _retry(
                        validate_market_data_mappings,
                        client,
                        market_update_set,
                        memo_content,
                        cfg,
                        source_directives=directives_dicts,
                        checkpoint=checkpoint,
                        stage="market_data_validation",
                        callback=callback,
                        retry_percent=89,
                    )
                for warning in validated_market.get("warnings", []):
                    checkpoint.add_warning("market_data_validation", warning)
                validated_market_path = os.path.join(request.output_dir, "market_mappings_validated.json")
                _write_json(validated_market_path, validated_market)
                checkpoint.set_output("market_mappings_validated", validated_market_path)

            # Stage 6: Apply market updates
            _emit(callback, "apply_market_updates", "Apply market updates", 91)
            with checkpoint.stage("apply_market_updates", "Applying market data to deck"):
                market_changes = apply_market_updates(
                    request.memo_path, validated_market, dry_run=False
                )
                checkpoint.set_count("market_changes", len(market_changes))
                log.info("Market data applied: %d changes", len(market_changes))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest test_pipeline_integration_mocked.py::TestMarketDataPipelineStages -v
```

Expected: PASS.

- [ ] **Step 5: Run full pipeline integration suite to confirm no regression**

```bash
python -m pytest test_pipeline_integration_mocked.py -v
```

Expected: All previously passing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add memo_chef/pipeline.py test_pipeline_integration_mocked.py
git commit -m "feat: add market data pipeline stages 4-6 (map, validate, apply)"
```

---

## Task 8: Enable Market Data Directive in UI

**Files:**
- Modify: `app.py`

No automated tests — verify visually in the running app.

- [ ] **Step 1: Enable the market data directive text area**

In `app.py`, find this block (around line 641–648):

```python
        market_data_directive = directive_cols2[0].text_area(
            "Market data directions (coming soon)",
            key="market_data_directive",
            placeholder="e.g., Use for rent trend charts only",
            height=68,
            disabled=True,
        )
```

Replace with:

```python
        market_data_directive = directive_cols2[0].text_area(
            "Market data directions",
            key="market_data_directive",
            placeholder="e.g., Only update rent growth charts. Do not touch occupancy slides.",
            height=68,
        )
```

- [ ] **Step 2: Verify directive is wired through to source_directives**

Search app.py for `market_data_directive.strip()`. You will find a block like:

```python
    if market_data_directive.strip():
        source_directives.append({
            "source_id": "market_data",
            "source_type": "market_data",
            "directive": market_data_directive.strip(), "scope": "both",
        })
```

If this exists → no change needed. The directive flows through `RunRequest.source_directives` → `directives_dicts` → `get_market_data_mappings` → prompt automatically.

If it does NOT exist → find the block that handles `proforma_directive.strip()` (near it) and add the above block immediately after it in the same pattern.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: enable market data directions field in UI"
```

---

## Task 9: Memo City News Panel

**Files:**
- Create: `memo_city_news.json`
- Modify: `app.py`

- [ ] **Step 1: Create `memo_city_news.json`**

```json
[
  {
    "version": "2.11",
    "date": "2026-03-27",
    "title": "Market Data Pipeline",
    "bullets": [
      "Claude now has a dedicated step just for market data — it reads your market workbook and updates charts, tables, and slide narrative separately from the proforma pass.",
      "Works with any Excel format, not just RealPage — Claude figures out which tabs contain market data by reading the column headers.",
      "Charts that don't match exactly (different markets, different time ranges) are adapted automatically — categories and series are updated alongside the numbers.",
      "When Claude updates a rent chart on slide 5, it also finds and updates any related narrative on slide 7 or summary table on slide 3.",
      "The 'Market data directions' field is now live — type instructions like 'Only update the rent growth chart' to guide what Claude touches."
    ]
  },
  {
    "version": "2.10",
    "date": "2026-03-19",
    "title": "Stream Timeout & Log Downloads",
    "bullets": [
      "Fixed a bug where runs would freeze at 68% and never finish.",
      "Added a retry spinner so you can see when the app is recovering from a slow API response.",
      "Admins can now download the full run log from the Recent Activity tab."
    ]
  },
  {
    "version": "2.9",
    "date": "2026-02-15",
    "title": "Before & After Report",
    "bullets": [
      "Every run now produces a branded Before vs. After change report showing exactly what changed and where.",
      "API usage cut by ~60% — consistency check and final review passes are now disabled by default."
    ]
  }
]
```

- [ ] **Step 2: Add Memo City News to `app.py`**

Find the bottom of the Streamlit sidebar or the main page footer. Add a new expander for the news feed. Look for where the sidebar ends (search for `st.sidebar` or the last widget section) and add:

```python
# ── Memo City News ────────────────────────────────────────────────────────────
import json as _json
_news_path = Path(__file__).parent / "memo_city_news.json"
if _news_path.exists():
    _news = _json.loads(_news_path.read_text(encoding="utf-8"))
    with st.expander("📰 Memo City News — What's new in the app", expanded=False):
        for item in _news:
            st.markdown(f"**v{item['version']} · {item['date']} — {item['title']}**")
            for bullet in item["bullets"]:
                st.markdown(f"- {bullet}")
            st.markdown("---")
```

**Where to place it:** Search app.py for `st.divider()` or the first `st.header(` or `st.subheader(` in the main content area (outside of `st.sidebar`). Place the news expander BEFORE the first file uploader section so it's visible to all users without scrolling. The import for `json` may already exist in app.py as something else — use `import json as _json` to avoid conflicts.

- [ ] **Step 3: Verify the news panel loads**

```bash
streamlit run app.py
```

Open the app, confirm the "Memo City News" expander is visible and shows 3 entries.

- [ ] **Step 4: Commit**

```bash
git add memo_city_news.json app.py
git commit -m "feat: add Memo City News panel and v2.11 market data entry"
```

---

---

> **Out of scope for this plan:** The spec mentions adding a `market_data` section to `memo_chef/redline.py` (the before-after report). `market_changes` is tracked in `checkpoint.set_count("market_changes", ...)` and the raw/validated JSON files are saved to the output dir. Wiring those into redline's report is a follow-up task in a separate PR.

---

## Task 10: Final Integration Test & PR

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest -v --tb=short 2>&1 | tail -30
```

Expected: All tests PASS (or pre-existing failures only — no new failures introduced).

- [ ] **Step 2: Run ruff lint**

```bash
ruff check memo_automator.py memo_chef/pipeline.py memo_chef/models.py app.py
```

Fix any issues reported.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/market-data-pipeline
gh pr create \
  --title "feat: dedicated market data pipeline step with dynamic workbook support" \
  --body "$(cat <<'EOF'
## Summary
- Adds dedicated pipeline stages 4-6 (market_data_mapping → market_data_validation → apply_market_updates) that run after the proforma apply
- Replaces hardcoded RealPage tab names with a keyword-scoring dynamic extractor that works with any Excel format
- Claude reasons across the full deck to update charts (values + categories + add/remove series), tables, and narrative on any slide
- Enables the 'Market data directions' UI field so users can constrain what Claude updates
- Adds Memo City News expander panel visible to all users

## Test plan
- [ ] Run full test suite: `pytest -v`
- [ ] Ruff clean: `ruff check`
- [ ] Manual: upload a market workbook in the UI, confirm stages 4-6 appear in run log
- [ ] Manual: add market data directions and confirm Claude respects them
- [ ] Manual: verify Memo City News panel is visible and shows correct entries
EOF
)"
```
