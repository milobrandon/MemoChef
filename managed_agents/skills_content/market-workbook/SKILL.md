---
name: market-workbook
description: Read user-approved tabs from a Subtext student-housing Market Analysis Workbook and apply rent-growth, occupancy, prelease, demand, and enrollment data to the memo's market-research slides. Use whenever a market-analysis xlsx is in supplemental files or the run brief lists "Market analysis workbook — approved tabs". Enforces the tab allowlist, the 3-panel layout convention, the standard benchmark universe (Subject / Power 4 / Subtext 30), semantic name matching, plausibility bounds, cross-slide propagation, and the slide-generation toggle.
---

# Market Workbook

Market data reaches you as supplemental workbook tabs (rent surveys, comp sets, occupancy/absorption trends, supply pipelines, submarket stats) or broker/appraisal PDFs. These drive charts, tables, AND narrative on the market, comps, and demand slides.

## User-controlled tab allowlist

When the user uploads a Market Analysis Workbook as supplemental data, the run message will list an explicit set of **approved tabs** under `Market analysis workbook — approved tabs`. You MUST:

1. Read ONLY the tabs in that allowlist. Do not open other tabs even if they look relevant. Back-end source tabs (raw IPEDS, RealPage dumps, manual enrollment data, raw supply/demand) are intentionally excluded — they are source data, not presentation-ready.
2. If no allowlist is provided, do not read the workbook tabs heuristically. Surface a changelog warning asking the deal team to specify tabs.
3. If a tab you'd need to answer a memo metric isn't in the allowlist, surface a changelog warning and skip the update rather than reading a non-approved tab.

## Standard output-tab layout

Output tabs follow a consistent shape. Typical names and their meaning:

| Tab | Contains |
|---|---|
| `Tables` | University snapshot, FTE/supply/demand grid, deliveries |
| `Competitive Set` | Finalized comp set (Year Built, Beds, Rent/Bed, YoY Rent Growth, Occupancy, Prelease) |
| `Rent Growth Comp. By Year` | Time series by benchmark set |
| `Occ. Comp. By Year` | Time series by benchmark set |
| `Prelease Comp. By Year` | Time series by benchmark set |
| `Uncap. Demand Comp` | Uncapitalized demand by year |
| `Applications` | Apps, application index, admit rate |
| `Enrollment` | FTE by benchmark set |

Every time-series tab shares a **3-panel layout**:

- **Panel 1** (starts around col Q): Market radius — columns for `Subject University`, `Power 4`, `Subtext 30`.
- **Panel 2** (further right): One Mile radius — same 3 benchmarks.
- **Panel 3**: tab-specific extra cuts (e.g. Beds Delivered).

The standard benchmark universe is:

- **Subject university** — the deal's host school.
- **Power 4** — 15 peer universities (see the `Power 4 & Subtext 30` tab).
- **Subtext 30** — institutional 30-property universe.

When extracting data: iterate the tab, find the row containing the benchmark names ("Kentucky", "Power 4", "Subtext 30"), then read the year + value columns beneath. Do NOT assume fixed row/column addresses.

## Slide generation toggle

The run message carries a `Generate new market research slides` flag:

- **ON** — if the memo's market research section is thinner than the approved tabs support, insert new slides into that section using the house style from `/mnt/examples/`. Insert adjacent to related existing content; do not append at the end. Narrative must cite specific years and deltas.
- **OFF (default)** — update existing slides only. Do NOT insert new slides even if the workbook has data the memo doesn't currently show.

Respect this flag strictly. Creating slides the user didn't ask for is worse than leaving workbook data unused.

## Semantic matching — tab/chart names do NOT need to be exact

Valid matches include:

- `"Florida"` ≈ `"UF"` (abbreviation)
- `"Rent Growth by Market"` ≈ `"Rent Growth Comparison By Year"` (synonym)
- `"Effective Rent"` ≈ `"Market Rate Rent"` (equivalent concept)
- `"Occupancy"` ≈ `"Trailing 3-Yr Avg Occ"` (aggregate of the same series)

When a workbook chart does not match 1:1 (different markets, different time range, different unit of measure), ADAPT it: update categories, add/remove series as needed, then rewrite any narrative on any slide that references the chart so it stays coherent.

## Cross-slide propagation — update ALL references, not just the first

A single market metric often appears on multiple slides. When a rent chart changes on slide 12, the summary table on slide 3 and the "Market Context" narrative on slide 7 likely reference the same number. Update all of them in one pass. Missing a cross-slide update is a quality defect, not a judgment call.

## Full-row consistency

When you update ANY column in a table row, update every OTHER column in that row that has corresponding workbook data. Never update demand figures without also updating the enrollment figures in the same row. The table must be internally consistent after updates.

## Narrative–heading coherence

If updated data contradicts a section heading or framing — e.g., a heading that reads "Consistent Growth" but the new data shows a year-over-year decline — reword the heading to match the new data. Do not leave stale framing above fresh numbers.

## Narrative–table coherence

When you update a table, read adjacent narrative text on the same or nearby slides. If it references the old values ("rents remain roughly flat", "details remain unclear", "pipeline of ~200 beds"), rewrite the narrative to match the new values. Do not let narrative contradict the tables it describes.

## Plausibility bounds — drop values outside these ranges

Treat the following as hard sanity checks. Any workbook value outside these ranges almost certainly reflects a unit-of-measure error (e.g., annual rent pasted into a monthly field, basis points treated as percents). Drop the update entirely rather than writing a nonsensical number to the memo, and mention it in the changelog warnings.

- Monthly rent per unit: > $0 and < $10,000
- Occupancy: between 0 and 1 (or 0% and 100%)
- Cap rates: between 2% and 15% (0.02 to 0.15)
- Growth rates: between -50% and +200% (-0.5 to 2.0)

## Confidence tracking

For each market-data update, self-assess confidence:

- `high` — exact semantic match, unambiguous mapping.
- `medium` — clear match but adaptation required (series renamed, categories realigned).
- `low` — judgment call, likely correct but worth flagging.

Include low-confidence updates anyway — downstream review catches them — but call them out as warnings in the changelog so the deal team can spot-check them first.

## Structural constraints (reminders)

- NEVER insert rows into side-by-side comp tables. Update existing rows only; add comps by replacing, not appending.
- Preserve existing series structure on charts unless the workbook clearly has a different shape (then adapt + rewrite narrative).
- Chart titles, axis labels, and series names on a chart should match what the underlying data actually represents after your update.
