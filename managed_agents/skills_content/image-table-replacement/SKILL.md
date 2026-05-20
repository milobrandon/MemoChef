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

## Procedure (do this for every match)

The original image is the **structural ground truth** for what rows and columns the rebuilt table must contain. The example memo is now only a *formatting* reference (fonts, colors, alignment, alternating shading) — it is NOT the row/column template. Dropping a line item that appeared in the original image is a regression, not a judgment call.

1. **Identify and inventory the Picture shape.** Record position (`left`, `top`) and size (`width`, `height`) — the new table must occupy the same bounding box. Capture the picture's binary content via python-pptx: `blob = picture.image.blob` and `ext = picture.image.ext`.

2. **Vision-extract the original image BEFORE removing the shape (CRITICAL).** Write the blob to a temp file (e.g. `/tmp/orig_table_slide{N}.{ext}`) and read it with the file tool so its contents become visible to you. From the image, extract and record:
   - The full ordered list of **row labels** (every line item AND every section header, exactly as they appear).
   - The full ordered list of **column headers**.
   - Each visible **cell value** — these are used only as a fallback if the updated proforma has no matching line (see step 5).
   This inventory is your structural ground truth for steps 4–5.

3. **Remove the Picture shape** from the slide.

4. **Build the table** with `slide.shapes.add_table(rows, cols, left, top, width, height)` using the row count and column count from the step-2 inventory — NOT from the example memo's table.

5. **Populate every cell from the UPDATED proforma.** For each row label captured in step 2, look up the corresponding value in the proforma data already extracted earlier in the run. For each row:
   - If the proforma has a clean matching line → use the proforma value.
   - If the proforma has NO matching line → retain the image-OCR'd value from step 2, append a footnote indicator to the cell text (e.g. `¹`), and log a changelog warning of the form `[retained from image — no proforma source: <row label>]`. **Never silently drop an image-only row.**
   - If the proforma has the line but the value is blank/zero where the image showed a real number → treat as no-match (retain + warn), do not overwrite a real number with a blank.

6. **Apply formatting** per the Formatting rules below (header style, body text color via the explicit-text-color rule, alternating shading, number formats, subtotal styling) — copy these from the example memo's equivalent table.

7. **Log the replacement** in the changelog as `Image replaced with editable table — [slide title]`, and include: (a) total row count, (b) the list of row labels preserved from the image, (c) any rows whose value was retained-from-image rather than proforma-sourced.

## Formatting rules

- **Header row**: bold, white text, dark background (match RGB from the example memo).
- **Body + subtotal rows — explicit text color (CRITICAL)**: freshly built tables default to **black text**, which is invisible on Subtext's dark-theme memos. Before populating body cells, read `run.font.color.rgb` from a reference body cell in any existing editable data table on the same memo (scan all slides for a table whose body cells have non-black text and use its color — the Proforma Comparison and end-of-memo Proforma / Underwriting Projections tables are typical sources, but slide numbers vary across templates so locate them by content, not page number). Apply that RGB to EVERY body and subtotal cell you populate. If no reference body table exists on the memo, read the color from the equivalent table in the example memo under `/mnt/examples/`. If neither is available, default to `RGBColor(0xFF, 0xFF, 0xFF)` (white) rather than leaving black.
- **Alternating row shading** where used in the example.
- **Font family, size, alignment** per column type (text=left, numbers=right or center, headers=center).
- **Number formatting**: $ with commas, % with one decimal, SF with commas.
- **Section subtotal rows**: bold, lightly shaded background, **same text color as body rows** (explicitly set — do not rely on the python-pptx default).

## Per-table minimum coverage (sanity check, not template)

The sections below describe the **minimum** row/column coverage each table type usually contains. Use them as a sanity check against the step-2 image inventory:

- If the image inventory includes additional rows or columns beyond what's listed → include them in the rebuilt table.
- If the image inventory is missing rows from this canonical list → the source image may be a partial view or cropped. Still rebuild from the image inventory, but log a changelog warning naming the missing rows so the analyst can review.
- These lists are NOT the structural template — the image is.

## Cash Flow table structure

Two columns per year shown in the example; at minimum include Year 1 and Year 2/Stabilized.

- **Revenue section**: Gross Potential Rent, (Vacancy Loss), Parking Revenue, Other Income, Utility Income, Total EGR
- **Expense section**: Management Fee, Admin, Maintenance, Landscaping, Insurance, Utilities, Total Controllable OpEx, RE Taxes, Total OpEx
- **NOI section**: NOI (before reserves), (Replacement Reserves), NOI (less reserves)
- **Returns**: Return on Cost (Yr 2), Untrended Return on Cost
- Include a `$/Bed` column if present in the example memo.

## Unit Mix table structure

- **Columns**: Unit Type, Avg SF, Beds/Unit, # Units, # Beds, % of Units, % of Beds, Rent/Bed (untrended)
- One row per unit sub-type from the Assumptions tab (S1, B1, B2, etc.).
- A bold **Total** row at the bottom with summed/weighted-average values.

## Development Budget table structure

- **Columns**: Line Item, Total Cost, % of Total, Cost/Bed
- **Sections**: Acquisition (land, closing costs), Hard Costs (site work, construction, contingency), Soft Costs (line items), Total
- Section header rows: bold with shaded background.
- `% of Total` = line total / grand total × 100, formatted as `"XX.X%"`.
- `Cost/Bed` = line total / total beds.
