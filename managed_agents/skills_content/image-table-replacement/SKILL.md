---
name: image-table-replacement
description: Replace image-only data slides (Cash Flow, Unit Mix, Development Budget) with freshly-built editable python-pptx tables populated from the proforma. Use when a slide whose title contains "Underwriting", "Cash Flow", "Proforma", "Unit Mix", "Development Budget", or "Dev Budget" has only Picture shapes and no editable table. Includes per-table column structures, the explicit-text-color rule for body cells, and the position-preservation procedure.
---

# Image-Table Replacement

After applying all standard cell updates per the `memo-table-updates` skill, scan every slide for the following content types that may be embedded as images (Picture shapes) instead of editable tables:

- **Cash Flow / Underwriting Projections** — slide title contains "Underwriting", "Cash Flow", or "Proforma" and the slide has no editable table, only Picture shapes.
- **Unit Mix** — slide title contains "Unit Mix" with only Picture shapes.
- **Development Budget** — slide title contains "Development Budget" or "Dev Budget" with only Picture shapes.

For each such slide, **replace the image with a freshly-built python-pptx table** populated entirely from the proforma data extracted earlier in the run.

## Data sources by proforma variant

The replacement tables are sourced from the same proforma the rest of the run already extracted. Tab names differ by variant:

- **Vanilla proforma** — Cash Flow, Assumptions (unit mix), Development Summary.
- **P3 proforma** (filename starts with "P3 ") — `Presentation Cash Flow`, `Presentation Exec Summary` (the unit mix breakdown is a sub-table here), and `Presentation Dev Budget`. These are the only tabs ingested for P3 runs and they carry the line-item detail the replacement tables need. **Do not skip the rebuild on P3 because the Assumptions tab is absent — read the `Presentation*` tabs instead.**

## Procedure (do this for every match)

1. Identify the Picture shape(s) on the slide. Record their position (left, top) and size (width, height) — the new table should occupy the same bounding box.
2. Remove the Picture shape from the slide.
3. Build the table using `slide.shapes.add_table(rows, cols, left, top, width, height)`. Use the row/column structure from the example memo's equivalent table as your structural template (number of rows, column widths, header labels).
4. Populate every cell with the corresponding proforma value.
5. Apply formatting to match the example memo exactly (see Formatting rules below).
6. Log the replacement in the changelog as "Image replaced with editable table — [slide title]".

## Formatting rules

- **Header row**: bold, white text, dark background (match RGB from the example memo).
- **Body + subtotal rows — explicit text color (CRITICAL)**: freshly built tables default to **black text**, which is invisible on Subtext's dark-theme memos. Before populating body cells, read `run.font.color.rgb` from a reference body cell in any existing editable data table on the same memo (scan all slides for a table whose body cells have non-black text and use its color — the Proforma Comparison and end-of-memo Proforma / Underwriting Projections tables are typical sources, but slide numbers vary across templates so locate them by content, not page number). Apply that RGB to EVERY body and subtotal cell you populate. If no reference body table exists on the memo, read the color from the equivalent table in the example memo under `/mnt/examples/`. If neither is available, default to `RGBColor(0xFF, 0xFF, 0xFF)` (white) rather than leaving black.
- **Alternating row shading** where used in the example.
- **Font family, size, alignment** per column type (text=left, numbers=right or center, headers=center).
- **Number formatting**: $ with commas, % with one decimal, SF with commas.
- **Section subtotal rows**: bold, lightly shaded background, **same text color as body rows** (explicitly set — do not rely on the python-pptx default).

## Cash Flow table structure

Two columns per year shown in the example; at minimum include Year 1 and Year 2/Stabilized.

- **Revenue section**: Gross Potential Rent, (Vacancy Loss), Parking Revenue, Other Income, Utility Income, Total EGR
- **Expense section**: Management Fee, Admin, Maintenance, Landscaping, Insurance, Utilities, Total Controllable OpEx, RE Taxes, Total OpEx
- **NOI section**: NOI (before reserves), (Replacement Reserves), NOI (less reserves)
- **Returns**: Return on Cost (Yr 2), Untrended Return on Cost
- Include a `$/Bed` column if present in the example memo.

## Unit Mix table structure

- **Columns**: Unit Type, Avg SF, Beds/Unit, # Units, # Beds, % of Units, % of Beds, Rent/Bed (untrended)
- One row per unit sub-type (S1, B1, B2, etc.) — see Data sources above for the variant-specific tab.
- A bold **Total** row at the bottom with summed/weighted-average values.

## Development Budget table structure

- **Columns**: Line Item, Total Cost, % of Total, Cost/Bed
- **Sections**: Acquisition (land, closing costs), Hard Costs (site work, construction, contingency), Soft Costs (line items), Total
- Section header rows: bold with shaded background.
- `% of Total` = line total / grand total × 100, formatted as `"XX.X%"`.
- `Cost/Bed` = line total / total beds.
