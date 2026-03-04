# Market Data Workbook — Design Doc

> **Date:** 2026-03-04
> **Author:** @brandon
> **Status:** Accepted

---

## Problem

IC memos reference market data (rent comps, sale comps, submarket stats, macro
indicators) but this data is currently typed into PowerPoint manually. There is
no structured way to compile, validate, or visualize market data for a deal.

The market data workbook will be a **large Excel file** with multiple tabs and
charts that serves as both:
1. A standalone market analysis deliverable
2. A structured data source that Memo Chef can ingest to enrich IC memos

---

## Goals

| # | Goal | Success Metric |
|---|------|----------------|
| M1 | Generate a multi-tab Excel workbook from market data | Workbook opens cleanly in Excel with formatted tables + charts |
| M2 | Include rent comps, sale comps, submarket stats, macro data | All 4 data categories present with source attribution |
| M3 | Auto-generate charts (bar, line, scatter) from data | At least 5 chart types render correctly |
| M4 | Support both manual data entry and API-sourced data | Manual JSON/CSV input works; API integration is additive |
| M5 | Prepare infrastructure only — full integration is Phase 3 | Scaffold + generator + sample output; no API calls yet |

---

## Workbook Structure

### Tabs (Worksheets)

| Tab | Contents | Charts |
|-----|----------|--------|
| **Cover** | Property name, address, date, prepared by | — |
| **Rent Comps** | Comp table: name, distance, unit types, rents, occupancy, year built | Bar chart: asking rent by comp; Scatter: rent vs distance |
| **Sale Comps** | Comp table: name, date, price, $/unit, cap rate, units | Bar chart: $/unit by comp; Line: cap rate trend |
| **Submarket Overview** | Stats: avg rent, vacancy, absorption, pipeline, rent growth | Bar: rent growth YoY; Pie: vacancy vs occupancy |
| **Macro Indicators** | SOFR, 10yr Treasury, CPI, unemployment (time series) | Line chart: rate trends over 12 months |
| **Unit Mix Summary** | Unit types, count, SF, rent, rent/SF | Stacked bar: unit count by type; Bar: rent by type |
| **Sources & Notes** | Data sources, as-of dates, disclaimers, methodology notes | — |

### Estimated Size

- 7 tabs, ~50-200 rows per data tab
- 8-10 embedded charts
- Formatted headers, number formats, conditional formatting
- Estimated file size: 500KB - 2MB depending on data density

---

## Data Schema

Reuses the schema from `DATA_SOURCES.md` with additions:

```python
@dataclass
class MarketDataWorkbook:
    property: PropertyInfo
    rent_comps: list[RentComp]
    sale_comps: list[SaleComp]
    submarket: SubmarketStats
    macro: MacroIndicators
    unit_mix: list[UnitMixEntry]
    metadata: WorkbookMetadata  # prepared_by, date, sources

@dataclass
class PropertyInfo:
    name: str
    address: str
    city: str
    state: str
    submarket: str
    msa: str
    property_type: str  # multifamily, mixed-use, etc.

@dataclass
class RentComp:
    name: str
    address: str
    distance_mi: float
    units: int
    year_built: int
    occupancy_pct: float
    unit_types: dict[str, float]  # {"Studio": 1450, "1BR": 1800, ...}
    concessions: str
    as_of_date: str
    source: str

@dataclass
class SaleComp:
    name: str
    address: str
    sale_date: str
    sale_price: float
    units: int
    price_per_unit: float
    cap_rate: float
    source: str

@dataclass
class SubmarketStats:
    name: str
    avg_rent: float
    vacancy_pct: float
    absorption_units: int
    pipeline_units: int
    rent_growth_yoy_pct: float
    median_hh_income: float
    population: int
    as_of_date: str
    source: str

@dataclass
class MacroIndicators:
    entries: list[MacroEntry]  # time series

@dataclass
class MacroEntry:
    date: str
    sofr_rate: float
    treasury_10yr: float
    cpi_yoy: float
    unemployment: float
    source: str

@dataclass
class UnitMixEntry:
    unit_type: str        # Studio, 1BR, 2BR, 3BR
    unit_count: int
    avg_sf: float
    avg_rent: float
    rent_per_sf: float

@dataclass
class WorkbookMetadata:
    prepared_by: str
    prepared_date: str
    property_name: str
    notes: str
```

---

## Chart Specifications

### Rent Comps Tab
1. **Bar Chart — Asking Rent by Comp**: X = comp name, Y = avg asking rent, grouped by unit type
2. **Scatter — Rent vs Distance**: X = distance (mi), Y = rent, bubble size = units

### Sale Comps Tab
3. **Bar Chart — Price per Unit**: X = comp name, Y = $/unit
4. **Line Chart — Cap Rate Trend**: X = sale date, Y = cap rate

### Submarket Tab
5. **Bar Chart — Rent Growth YoY**: Single bar showing submarket rent growth %
6. **Pie Chart — Vacancy Split**: Occupied vs vacant percentage

### Macro Tab
7. **Line Chart — Rate Trends**: Multi-series line (SOFR, 10yr Treasury) over time
8. **Line Chart — CPI Trend**: CPI YoY over 12 months

### Unit Mix Tab
9. **Stacked Bar — Unit Count by Type**: X = unit type, Y = count
10. **Bar — Avg Rent by Type**: X = unit type, Y = rent

---

## Implementation Components

### `market_workbook.py` (new module)

```python
def generate_workbook(data: MarketDataWorkbook, output_path: str) -> str:
    """Generate the full market data Excel workbook."""

def _write_cover_sheet(wb, data) -> None:
    """Write the cover/title sheet."""

def _write_rent_comps(wb, comps, charts=True) -> None:
    """Write rent comps table + charts."""

def _write_sale_comps(wb, comps, charts=True) -> None:
    """Write sale comps table + charts."""

def _write_submarket(wb, stats, charts=True) -> None:
    """Write submarket overview + charts."""

def _write_macro(wb, macro, charts=True) -> None:
    """Write macro indicators + trend charts."""

def _write_unit_mix(wb, units, charts=True) -> None:
    """Write unit mix summary + charts."""

def _write_sources(wb, metadata) -> None:
    """Write sources and methodology notes."""

def _apply_formatting(ws, header_row=1) -> None:
    """Apply standard formatting: headers, number formats, column widths."""

def _add_bar_chart(ws, title, categories, values, anchor, ...) -> None:
    """Helper to create a bar chart."""

def _add_line_chart(ws, title, categories, series, anchor, ...) -> None:
    """Helper to create a line chart."""

def _add_scatter_chart(ws, title, x_values, y_values, anchor, ...) -> None:
    """Helper to create a scatter chart."""

def _add_pie_chart(ws, title, categories, values, anchor, ...) -> None:
    """Helper to create a pie chart."""
```

### Dependencies

- `openpyxl` (already in requirements.txt) — supports chart creation via
  `openpyxl.chart` module
- No new dependencies needed

### Sample Data

A `sample_market_data.json` file will be provided for testing and demos,
containing anonymized but realistic data for a fictional multifamily property.

---

## Integration Points (Future — Phase 3)

1. **Streamlit UI**: "Generate Market Workbook" button alongside "Fire"
2. **Memo enrichment**: Memo Chef reads the workbook as a data source
3. **API ingestion**: FRED/Census data auto-populates macro + demographics
4. **Batch mode**: Generate workbooks for multiple properties

---

## Formatting Standards

- **Headers**: Bold, dark green fill (#16352E), white text, frozen top row
- **Numbers**: Currency ($#,##0), percentages (0.0%), integers (#,##0)
- **Dates**: MM/DD/YYYY format
- **Column widths**: Auto-fit with minimum 12 characters
- **Charts**: Subtext brand colors (accent1-accent6 from theme)
- **Source attribution**: Every data tab has a "Source" column

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Chart rendering varies across Excel versions | Test on Excel 365, LibreOffice, Google Sheets |
| Large datasets slow down generation | Limit to 50 comps per category; paginate if needed |
| openpyxl chart limitations | Stick to basic chart types (bar, line, scatter, pie) |
| File size bloat from embedded charts | Charts reference data ranges, not embedded copies |

---

## Scope for This Phase

**In scope (prepare/scaffold):**
- Data models (dataclasses)
- Workbook generator with all tabs and charts
- Sample data file for testing
- Formatting and branding
- Unit tests for generation

**Out of scope (Phase 3):**
- API data fetching (FRED, Census, CoStar)
- Streamlit UI integration
- Memo enrichment pipeline
- Batch generation
