---
name: image-table-replacement
description: Replace pasted proforma-screenshot pictures (Executive Summary, Cash Flow / Underwriting Projections, Unit Mix, Development Budget) with freshly-built editable python-pptx tables populated from the proforma. Converting the screenshot to a real table is the DEFAULT treatment for proforma data ranges. Use when a slide has a Picture shape whose content is a grid of proforma rows/columns and the slide title contains "Underwriting", "Cash Flow", "Proforma", "Project Assumptions", "Unit Mix", "Development Budget", "Dev Budget", "Development Cost", or "Executive Summary". Includes detection cues, per-table column structures, the explicit-text-color rule for body cells, the row-tier styling map, and the position-preservation procedure.
---

# Image-Table Replacement (DEFAULT for proforma data tables)

After applying all standard cell updates per the `memo-table-updates` skill, scan every slide for proforma data that was pasted as an **image (Picture shape)** instead of an editable table. **The default action is to replace that picture with a freshly-built, editable python-pptx table** populated from the proforma — not to leave it as a picture.

Only leave a range as a picture when it does **not** tabularize cleanly — a merged-cell waterfall **Returns Summary**, a chart, or a range the user explicitly wants kept as a faithful screenshot. When in doubt for a rectangular row/column proforma range, convert it.

## Non-tabularizable ranges — you cannot refresh a picture in this environment

This sandbox has **no Excel or PowerPoint COM**, so a pasted proforma-range screenshot cannot be re-rendered from the updated model and swapped in place. That changes how you handle a range that does **not** tabularize cleanly (a merged-cell waterfall **Returns Summary**, an embedded **chart**, or a stylized block):

1. **Prefer a best-effort native table.** If the range is even loosely row/column shaped, rebuild it as a native table (procedure below) rather than leaving a stale screenshot — a readable editable table beats out-of-date numbers.
2. **Otherwise leave the picture and WARN — never silently.** If it genuinely cannot be tabularized, leave the Picture untouched and emit a prominent changelog warning, e.g. `[stale picture — could not refresh: <slide title> — no Excel render in this environment; refresh this screenshot manually before distribution]`. List every such picture so the deal team knows which exhibits still show old numbers. Never ship a proforma screenshot as if it were current when its underlying numbers changed this run.

## What counts as a "proforma Excel-table picture"

Detect by BOTH cues:

- **Title cue** — slide title contains "Underwriting", "Cash Flow", "Proforma", "Project Assumptions", "Unit Mix", "Development Budget", "Dev Budget", "Development Cost", or "Executive Summary".
- **Shape cue** — a `Picture` shape (`shape.shape_type == 13`) whose content, when viewed, is a grid of proforma rows/columns (numbers in a tabular layout).

A slide may hold a proforma picture **alongside** narrative text or other shapes (e.g. a "Project Assumptions" exec-summary picture next to a "Return Summary" picture). Convert **each** table-like picture independently — do NOT require the slide to be picture-only.

## The original image is the structural ground truth

The pasted screenshot defines which rows and columns the rebuilt table must contain. The example memo is only a *formatting* reference (fonts, colors, alignment, alternating shading, row tiers) — it is NOT the row/column template. Dropping a line item that appeared in the original image is a regression, not a judgment call.

## Procedure (do this for every matching picture)

1. **Identify and inventory the Picture shape.** Record position (`left`, `top`) and size (`width`, `height`) — the new table must occupy the same bounding box. Capture the picture's binary content via python-pptx: `blob = picture.image.blob` and `ext = picture.image.ext`.

2. **Vision-extract the original image BEFORE removing the shape (CRITICAL).** Write the blob to a temp file (e.g. `/tmp/orig_table_slide{N}.{ext}`) and read it with the file tool so its contents become visible to you. From the image, extract and record, in order:
   - every **row label** (every line item AND every section header / subtotal / total, exactly as written, including indentation),
   - every **column header**,
   - each visible **cell value** — used only as a fallback if the updated proforma has no matching line (see step 5),
   - which rows are **section headers**, **subtotals**, **totals**, and the **grand total** (you map these to row tiers in step 6).
   This inventory is your structural ground truth for steps 4–6.

3. **Remove the Picture shape** from the slide: `picture._element.getparent().remove(picture._element)`.

4. **Build the table** with `slide.shapes.add_table(rows, cols, left, top, width, height)` using the row count and column count from the step-2 inventory — NOT from the example memo's table.

5. **Populate every cell from the UPDATED proforma.** For each row label captured in step 2, look up the corresponding value in the proforma data already extracted earlier in the run. For each row:
   - If the proforma has a clean matching line → use the proforma value.
   - If the proforma has NO matching line → retain the image-OCR'd value from step 2, append a footnote indicator to the cell text (e.g. `¹`), and log a changelog warning of the form `[retained from image — no proforma source: <row label>]`. **Never silently drop an image-only row.**
   - If the proforma has the line but the value is blank/zero where the image showed a real number → treat as no-match (retain + warn); do not overwrite a real number with a blank.
   Pre-format every value as its final **display string** (commas, `$`, `%`, parentheses for negatives) before writing it — you are placing text into cells, not formatting numbers in-cell.

6. **Apply formatting** per the Formatting rules below (header style, body text color via the explicit-text-color rule, row-tier styling, alternating shading, number formats, subtotal/total styling) — copy these from the example memo's equivalent table.

7. **Verify with python-pptx (no COM in the sandbox).** Re-open the saved deck and confirm: the table has the expected row/column counts; every cell that should have a value is non-empty; no body/subtotal run has `font.color.rgb` of `None` or black (the invisible-text regression); and the table's bounding box still fits inside the slide and does not collide with the title/footer (compare against `prs.slide_width`/`prs.slide_height` and the original picture box recorded in step 1). Fix any failure before saving.

8. **Log the replacement** in the changelog as `Image replaced with editable table — [slide title]`, and include: (a) total row count, (b) the list of row labels preserved from the image, (c) any rows whose value was retained-from-image rather than proforma-sourced.

## Formatting rules

Build the table with `add_table` and style it explicitly — a freshly built python-pptx table defaults to **black text** and a banded table style that overpaints your fills, which produces invisible text on Subtext's dark-theme memos. Set fills and font colors explicitly on every cell.

- **Header / column-header row**: bold, white text, dark background (match RGB from the example memo's equivalent table; use the heading font, centered).
- **Body rows — explicit text color (CRITICAL)**: before populating body cells, read `run.font.color.rgb` from a reference body cell in any existing editable data table on the same memo (scan all slides for a table whose body cells have non-black text and use its color — the Proforma Comparison and end-of-memo Proforma / Underwriting Projections tables are typical sources, but slide numbers vary across templates so locate them by content, not page number). Apply that RGB to EVERY body and subtotal cell you populate. If no reference body table exists on the memo, read the color from the equivalent table in the example memo under `/mnt/examples/`. If neither is available, default to `RGBColor(0xFF, 0xFF, 0xFF)` (white) rather than leaving black.
- **Row tiers** — map each image row to one tier and style it to match that tier in the example memo (read the tier's fill + font color from the example/reference table; never hardcode brand hex):
  - `title` — table caption ("Development Budget"): dark band, white, bold.
  - `subtitle` — the unit/bed-count line under the caption: dark band, white.
  - `colheader` — the column-header row: dark band, white, bold, heading font, centered.
  - `section` — section header rows (EFF. GROSS REVENUE, HARD COSTS).
  - `subtotal` — section subtotals (Eff. Gross Revenue, Total Hard Costs): same text color as body, lightly shaded.
  - `total` — major totals (TOTAL EFF. GROSS REVENUE, NET OPERATING INCOME): bold band.
  - `grandtotal` — the single grand total (TOTAL DEV. COSTS): accent band, bold.
  - `body` — ordinary line items.
  - `bodybold` — an emphasized line (RETURN ON COST).
  Cash-flow section/total bands are dark with white text; development-budget section/subtotal bands are light with dark text — read the right pair from the example so the two flavors don't get crossed.
- **Alternating row shading** where the example uses it.
- **Font family, size, alignment** per column type (text=left, numbers=right or center, headers=center). Middle vertical anchor, tight cell margins.
- **Number formatting**: `$` with commas, `%` with one decimal, SF with commas.

## Per-table minimum coverage (sanity check, not a template — the image is the template)

If the image inventory has rows/columns beyond these lists → INCLUDE them. If the inventory is missing rows from a list → the source image may be cropped; rebuild from the image inventory anyway and log a changelog warning naming the missing rows.

### Cash Flow / Underwriting Projections
- Columns: a **Per Bed** column, **Year 1 … Year 5** (or as many years as the picture shows), and an **Untrended** column — match the picture.
- Revenue: Gross Potential Rent, (General Vacancy), (Collection / Bad Debt), Parking Revenue, Other Income, Utility Income, Eff. Gross Residential, any commercial rows present, **TOTAL EFF. GROSS REVENUE**.
- Controllable expenses: G&A, Payroll, Leasing & Marketing, Maintenance & Repairs, Contract Services, Turnover. Non-controllable: Utilities, Property Taxes, Insurance, Management Fees. **TOTAL OPERATING EXPENSES**.
- **NET OPERATING INCOME**, capital / replacement-reserve rows, **NOI (LESS RESERVES)**, **RETURN ON COST** row.

### Unit Mix
- Columns: Unit Type, Avg SF, Bed SF, Beds/Unit, # Units, # Beds, % Units, % Beds, $/Bed (Rent/Bed untrended), Rent/PSF — match the picture.
- One row per unit sub-type (1BR/1BA, 2BR/2BA, 4BR/4BA, etc.); a bold **TOTAL / AVG** row with summed counts and weighted-average sizes/rents.

### Development Budget
- A `title` row (caption) and a `subtitle` row (## Units | ## Beds), then a `colheader` row: Line Items, % Total, $ Amount, $/Unit, $/Bed, $/GSF, $/NRSF — match the picture.
- Sections (`section`): ACQUISITION COSTS, HARD COSTS, SOFT COSTS, CAPITAL STRUCTURE — each followed by its line items and a `subtotal` (Total Acquisition, Total Hard Costs, Total Soft Costs). `grandtotal` row: TOTAL DEV. COSTS.
- `% Total` = line $ / grand-total $ × 100. `$/Unit`, `$/Bed`, `$/GSF`, `$/NRSF` = line $ / the respective denominator. Recompute all of these from the new figures.

### Executive Summary
- A compact metrics block (units, beds, GSF/NSF, cost/bed, IRR, equity multiple, yield). Mirror the picture's rows/columns; `colheader` / `body` plus a `total` row if the picture has one.

## Common mistakes
- **Leaving it a picture.** Conversion is the default. Only keep a picture for non-tabularizable ranges (a merged-cell waterfall returns summary, a chart) or an explicit user request.
- **Black-on-dark text.** Never use `cell.text =`, and never set fills without also setting font color. Read every populated run's `font.color.rgb` afterward and re-apply the reference body color to any run that came back `None` or black.
- **Dropping image rows** not found in the proforma — retain + footnote + warn instead.
- **Wrong section tier** — cash-flow section/total bands are dark with white text; development-budget section/subtotal bands are light with dark text. Read the correct tier styling from the example memo.
- **Not verifying** — always re-open the saved deck and check values, colors, and fit before finishing.
