---
name: memo-table-updates
description: Update existing PowerPoint table cells in Subtext IC memos with proforma-sourced values while preserving run-level font formatting (color, size, family, bold). Use whenever you need to modify cell text in any data table on the memo template — Pipeline, Comp Summary, Side-by-Side Comp, Unit Mix, Cash Flow, Development Budget, Proforma Comparison, or any other tabular content. Includes the canonical set_cell_value helper, font-color regression check, range/parenthetical conventions, and rules for filling empty cells without losing formatting.
---

# Memo Table Updates

This skill governs every cell-level edit you make to existing tables in a Subtext IC memo. It does **not** cover building a fresh table to replace an embedded image — see the `image-table-replacement` skill for that. It does **not** cover changing the row/column structure of a table either; that's a deliberate ask.

## Cardinal rule — preserve run-level formatting

NEVER write `cell.text = "new value"` or `tf.text = "..."`. That pattern deletes the cell's existing runs along with their `<a:rPr>` (font family, size, color, bold/italic), and python-pptx rebuilds the cell with default theme formatting. On Subtext's dark-theme memos this produces **black/dark text on a dark background** that looks like the cell was never updated. This is the single most common formatting regression — treat `cell.text =` as forbidden for any cell that already has content.

Instead, modify the existing run's `.text` in-place so its `rPr` (and therefore font color) is preserved. Use this helper for EVERY numeric or text update to an existing table cell:

```python
def set_cell_value(cell, new_value) -> None:
    # Write new_value into cell, preserving the existing run's font
    # color, family, size, and bold/italic attributes (its rPr).
    new_value = str(new_value)
    tf = cell.text_frame
    # Collect runs ONCE. python-pptx hands back a fresh _Run proxy on every
    # paragraph.runs access, so a second pass that re-iterates tf.paragraphs
    # and checks `run is not target_run` will blank the target's own run
    # (it's a different proxy wrapping the same element) — emptying the cell.
    runs = [run for para in tf.paragraphs for run in para.runs]
    if not runs:
        # Truly empty cell — see "Font size when filling empty cells" below;
        # copy rPr from an adjacent non-empty cell after writing.
        para = tf.paragraphs[0] if tf.paragraphs else tf.add_paragraph()
        para.add_run().text = new_value
        return
    target_run = next((r for r in runs if r.text.strip()), runs[0])
    target_el = target_run._r
    target_run.text = new_value
    # Empty any OTHER runs so stale fragments don't reappear, but keep the run
    # elements themselves. Compare the element (run._r), NOT proxy identity.
    for run in runs:
        if run._r is not target_el:
            run.text = ""
```

**Always re-read at least one edited cell** — re-open the saved deck with python-pptx and read `cell.text` to confirm the value is actually present. A silently-emptied `<a:t/>` looks fine in code but renders blank in the deck.

If you build your own helper, spot-check `run.font.color.rgb` on a few updated cells before AND after the write. The values must match. If the color is `None` or `RGBColor(0x00,0x00,0x00)` after but was a light color before, your helper is stripping rPr — go back to `set_cell_value`.

## Number, text, and date formatting

For each table cell whose value comes from the proforma, replace the old text with the new value. Preserve formatting: commas in numbers, dollar signs, percent signs, decimal precision matching the memo's existing style. When a source value changes, also update ALL derived values: totals, subtotals, ratios (parking ratio, cost per bed/unit), per-bed/per-unit metrics, summed pipeline beds/units.

## Pipeline / Comp Summary tables (row-oriented)

Tables where each ROW is a property. The **first data row is always the subject property**. Update the subject row to match the proforma. If the subject appears under a prior project name, update that row too.

## Competitive Set Side-by-Side tables (column-oriented)

Tables where each COLUMN is a property. Find the subject property column by matching header text to the property name. Do NOT assume it's the leftmost column.

- **Unit mix source**: Use the detailed unit mix from the Assumptions tab top (individual unit rows, NOT the summary section). Each row in the detailed mix represents a distinct sub-type (e.g. S1, S2, B2, B3, D1, D2, D3, D4).
- **Populate ALL unit type rows**: If the subject property column has empty cells for a unit type block that other properties have data for, you MUST fill those cells using proforma data. Empty cells in the subject column are gaps that need to be filled, not intentional blanks.
- **Range formatting**: When multiple proforma sub-types map to the same bedroom block (e.g. two Studio types S1 and S2, or two 4BR/2BA types D1 and D2), show RANGES in the subject column:
  - Unit Size: "356 - 419 sf" (min to max across sub-types)
  - Market Rent: "$1,650 - $1,750" (min to max across sub-types)
  - # of Beds: SUM of beds across sub-types (e.g. S1 beds + S2 beds)
  If all sub-types for a block have the same value, show a single value (not a range). Follow the range formatting style already used in the table (e.g. "1,050 - 1,071 sf" or "$1,205 - $1,210").
- **Bed count per unit type**: The bed count row is for THAT specific type, not total property beds. Calculate from units × beds-per-unit, then sum across sub-types within the block.
- **Parenthetical notation**: "4BR/4BA (212)" means 212 = total beds for that type (53 units × 4 beds = 212). Not the unit count.
- **Split bedroom blocks**: When a bedroom type spans multiple rows, each row's metrics must reflect ONLY the unit types assigned to that row. Never mix data across split blocks.
- **DO NOT insert rows** into side-by-side comp tables. This breaks column alignment for all other properties.
- **Font size when filling empty cells**: When populating an empty cell in a comp table, match the font size and font name of the adjacent non-empty cells in the same row (or the row above/below if the entire row is empty). Never leave the default font size — always explicitly set it to match the table's existing style. Use python-pptx to read the font size from a reference cell before writing the new value.

## Row inserts for missing unit types

When the proforma has unit types not in the memo's unit mix table, add new rows. But NEVER insert rows into side-by-side comp tables.

## Table structure changes

Only restructure tables when explicitly instructed by the user or when the proforma structure has fundamentally changed.

## Formatting verification pass (REQUIRED after edits)

After applying all data updates, compare the output memo's formatting against the example memos:

- Check that fonts, sizes, and colors match the example style.
- **Font color regression check (CRITICAL — run this on EVERY table whose cells you modified, identified by content not slide number: Proforma Comparison, the end-of-memo cash-flow / Underwriting Projections / Proforma table, Unit Mix, Development Budget, comp side-by-sides, and any other data tables you touched)**: iterate all runs in each such table and flag any run whose `font.color.rgb` is `None` or `RGBColor(0x00,0x00,0x00)` (default black). If the surrounding unmodified cells in the same table use a light color (anything close to white), the flagged cell had its `rPr` stripped by a `cell.text =` overwrite — re-apply the neighboring cell's `font.color.rgb` to the flagged run, preserving its text content. Do NOT skip this check for subtotal rows; they are the most commonly affected because agents treat them as "summary values" and rewrite them wholesale.
- Verify number formatting consistency (decimal places, $ signs, commas).
- Ensure table cell alignment matches (left/center/right per column type).
- Fix any formatting drift introduced during updates (e.g. a cell that lost its bold or changed font size after text replacement).

Log any formatting corrections in the changelog.
