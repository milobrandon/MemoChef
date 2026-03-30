# Run Manifest Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 issues found in the Knoxville-Cumberland run manifest — table font corruption, mapping accuracy gaps, consistency check false positives, formatting detection, and typo flagging.

**Architecture:** Three independent workstreams: (A) code changes to branding/layout to stop modifying table fonts, (B) prompt text additions to four prompt files, (C) one prompt addition for typo detection. All changes are additive or deletions — no architectural changes.

**Tech Stack:** python-pptx, pytest, Anthropic prompt templates

---

### Task 1: Stop `apply_branding()` from modifying table cell fonts

**Files:**
- Modify: `memo_automator.py:3667-3685`
- Modify: `test_updates_formatting.py:51-76`

- [ ] **Step 1: Write failing test — table fonts preserved after branding**

Add to `test_updates_formatting.py` after the existing `test_apply_branding` function:

```python
def test_apply_branding_preserves_table_fonts(sample_pptx):
    """apply_branding must NOT change font family or size in table cells."""
    theme_path = os.path.join(os.path.dirname(__file__), "Subtext Brand Theme.thmx")
    if not os.path.exists(theme_path):
        pytest.skip("Theme file not found for branding test")

    # Set distinctive fonts on table cells before branding
    from pptx.util import Pt
    prs = Presentation(sample_pptx)
    table = None
    for shape in prs.slides[0].shapes:
        if shape.has_table:
            table = shape.table
            break
    assert table is not None

    # Header cell: set to Arial Bold 14pt
    header_run = table.cell(0, 0).text_frame.paragraphs[0].runs[0]
    header_run.font.name = "Arial"
    header_run.font.size = Pt(14)
    header_run.font.bold = True

    # Body cell: set to Calibri 10pt
    body_run = table.cell(1, 1).text_frame.paragraphs[0].runs[0]
    body_run.font.name = "Calibri"
    body_run.font.size = Pt(10)

    prs.save(sample_pptx)

    cfg = {
        "branding": {
            "heading_size_threshold": 18,
            "color_distance_threshold": 80,
        }
    }
    apply_branding(sample_pptx, theme_path, cfg)

    prs2 = Presentation(sample_pptx)
    table2 = None
    for shape in prs2.slides[0].shapes:
        if shape.has_table:
            table2 = shape.table
            break

    header_run2 = table2.cell(0, 0).text_frame.paragraphs[0].runs[0]
    body_run2 = table2.cell(1, 1).text_frame.paragraphs[0].runs[0]

    # Fonts must be unchanged
    assert header_run2.font.name == "Arial"
    assert header_run2.font.size == Pt(14)
    assert body_run2.font.name == "Calibri"
    assert body_run2.font.size == Pt(10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_updates_formatting.py::test_apply_branding_preserves_table_fonts -v`
Expected: FAIL — font names will be "Pragmatica Bold"/"Pragmatica Book" instead of "Arial"/"Calibri"

- [ ] **Step 3: Remove table cell processing from `apply_branding()`**

In `memo_automator.py`, delete the table cell processing block (lines 3667-3685). Replace with a comment:

```python
            # Table cells: skip font changes entirely — preserve original
            # font families and sizes set by the analyst.
```

The full block to remove is:

```python
            # Process table cells (conservative: font only, preserve alignment & color)
            if shape.has_table:
                table = shape.table
                for row_idx, row in enumerate(table.rows):
                    for cell in row.cells:
                        for para in cell.text_frame.paragraphs:
                            for run in para.runs:
                                # Determine heading from existing bold state or row 0
                                is_cell_heading = (
                                    row_idx == 0
                                    or run.font.bold is True
                                )
                                _reformat_run(
                                    run, is_cell_heading, heading_threshold,
                                    heading_font, body_font,
                                    color_threshold,
                                    skip_color=True,  # tables use deliberate colors
                                )
                                runs_reformatted += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_updates_formatting.py::test_apply_branding_preserves_table_fonts -v`
Expected: PASS

- [ ] **Step 5: Verify existing branding test still passes**

Run: `pytest test_updates_formatting.py::test_apply_branding -v`
Expected: PASS — this test only checks non-table shapes (SubheaderBox), so it should still pass.

**Note:** If this test checks table cell font names and fails, update the assertions to remove the table cell font checks since we no longer modify table fonts.

- [ ] **Step 6: Commit**

```bash
git add memo_automator.py test_updates_formatting.py
git commit -m "fix: stop apply_branding from changing table cell fonts"
```

---

### Task 2: Remove table font size normalization from `normalize_layout()`

**Files:**
- Modify: `memo_automator.py:4079-4144`
- Modify: `test_updates_formatting.py:79-97`

- [ ] **Step 1: Write failing test — table font sizes preserved after layout normalization**

Add to `test_updates_formatting.py`:

```python
def test_normalize_layout_preserves_table_font_sizes(tmp_dir):
    """normalize_layout must NOT change font sizes in table cells."""
    from pptx.util import Pt

    path = os.path.join(tmp_dir, "font_size_test.pptx")
    prs = Presentation()

    # Slide 1: table with mixed font sizes (intentional)
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    table_shape = slide.shapes.add_table(
        3, 2, Inches(0.5), Inches(1.0), Inches(6.0), Inches(2.0),
    )
    table = table_shape.table
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Rent"
    table.cell(1, 1).text = "$1,500"
    table.cell(2, 0).text = "Notes"
    table.cell(2, 1).text = "See appendix"

    # Set header to 12pt, body to 8pt (intentionally different)
    for cell in [table.cell(0, 0), table.cell(0, 1)]:
        for run in cell.text_frame.paragraphs[0].runs:
            run.font.size = Pt(12)
    for row_idx in [1, 2]:
        for col_idx in [0, 1]:
            for run in table.cell(row_idx, col_idx).text_frame.paragraphs[0].runs:
                run.font.size = Pt(8)

    prs.save(path)

    cfg = {
        "layout": {
            "margin_left": 0.50,
            "margin_right": 0.50,
            "margin_top": 0.25,
            "margin_bottom": 0.50,
            "snap_tolerance": 0.05,
        }
    }
    summary = normalize_layout(path, cfg)
    assert summary["table_font_size_normalized"] == 0

    # Verify sizes are unchanged
    prs2 = Presentation(path)
    table2 = None
    for shape in prs2.slides[0].shapes:
        if shape.has_table:
            table2 = shape.table
            break

    body_run = table2.cell(1, 0).text_frame.paragraphs[0].runs[0]
    assert body_run.font.size == Pt(8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test_updates_formatting.py::test_normalize_layout_preserves_table_font_sizes -v`
Expected: FAIL — `table_font_size_normalized` will be > 0 and font size will be changed from 8pt

- [ ] **Step 3: Delete the font size normalization block**

In `memo_automator.py`, replace lines 4079-4144 (the entire section starting from `# --- 1h. Cross-slide formatting consistency ---` through the table font normalization) with just the counter set to 0. Specifically, find and delete:

1. The `table_size_usage` collection loop (scanning all tables to build the frequency map)
2. The `dominant_table_size` computation
3. The normalization loop that overwrites font sizes

Replace with:

```python
    # Table font sizes: preserved as-is (analyst-set formatting is intentional)
    summary["table_font_size_normalized"] = 0
```

Keep any non-table-font parts of section 1h intact (e.g., text frame formatting checks if any exist in that block).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test_updates_formatting.py::test_normalize_layout_preserves_table_font_sizes -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest test_updates_formatting.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add memo_automator.py test_updates_formatting.py
git commit -m "fix: remove table font size normalization from normalize_layout"
```

---

### Task 3: Add SF verification rule to mapping prompt

**Files:**
- Modify: `prompts/mapping_v1.txt`

- [ ] **Step 1: Add the SF verification rule**

In `prompts/mapping_v1.txt`, after rule 3 (the "Derived / calculated values" paragraph ending around line 29), add as a new sub-rule:

```
   - **SF verification** — For every square footage value you emit (GSF, NSF,
     amenity SF, leasing SF, retail SF, exterior SF), trace it to a specific
     proforma row and include that row reference in the `source` field (e.g.
     "Executive Summary Row 21: Amenity & Leasing SF"). If no proforma row
     matches the SF value, do NOT emit the update — leave the memo's existing
     value in place. Never propagate a memo value that has no proforma source.
```

- [ ] **Step 2: Add unit/bed final check reminder**

In `prompts/mapping_v1.txt`, after the last numbered rule (rule 24 / table structure changes) and before the "CRITICAL: Return ONLY the raw JSON" line, add:

```
**UNIT vs BED COUNTS — FINAL CHECK:** Before emitting your JSON, review every
unit mix update. Parenthetical numbers like "(84)" = total beds for that unit
type (units × beds_per_unit). The separate "Units" or "# Units" column = raw
unit count. The "# Beds" or "Bed Count" row = beds for that type. These are
DIFFERENT numbers. Getting them backwards is a critical error.
```

- [ ] **Step 3: Verify prompt loads without error**

Run: `python -c "from memo_automator import MAPPING_PROMPT; print(len(MAPPING_PROMPT), 'chars loaded')"`
Expected: prints char count without error

- [ ] **Step 4: Commit**

```bash
git add prompts/mapping_v1.txt
git commit -m "fix: add SF verification and unit/bed check to mapping prompt"
```

---

### Task 4: Add false positive guard to consistency check prompt

**Files:**
- Modify: `prompts/consistency_check_v1.txt`

- [ ] **Step 1: Add the false positive instruction**

In `prompts/consistency_check_v1.txt`, after the 5-point audit list (after item 5 "Narrative accuracy" ending around line 41) and before the "CRITICAL: Return ONLY the raw JSON" line, add:

```
**Do NOT report false positives.** Before adding any discrepancy to your
output, compare your `expected` and `found` values character-by-character.
If they are identical (same text, same formatting), do NOT include that item
in the discrepancies array. Only report genuine mismatches where the values
actually differ. A discrepancy where expected equals found is a bug in YOUR
output, not a bug in the memo.
```

- [ ] **Step 2: Run existing prompt test**

Run: `pytest test_prompts.py -v`
Expected: PASS (prompt template loads and format placeholders are valid)

- [ ] **Step 3: Commit**

```bash
git add prompts/consistency_check_v1.txt
git commit -m "fix: add false positive guard to consistency check prompt"
```

---

### Task 5: Add column format inheritance rule to validation prompt

**Files:**
- Modify: `prompts/validation_v1.txt`

- [ ] **Step 1: Add the column format inheritance instruction**

In `prompts/validation_v1.txt`, after the formatting check instruction (item 3 "Is the formatting consistent" around line 21), add as a new domain rule at the end of the domain rules section (before the `{property_name_section}` placeholder):

```
- **Column format inheritance** — For each table column in the proposed updates,
  check whether values in the same column consistently use a formatting prefix
  (e.g., "$") or suffix (e.g., "%"). If a proposed `new_value` omits a prefix
  or suffix that every other cell in that column uses, emit a `correction`
  entry that adds the missing prefix/suffix. Likewise, if an existing memo
  value already violates the column formatting convention and is NOT being
  updated by any proposed change, emit it as a `missed` entry.
```

- [ ] **Step 2: Run existing prompt test**

Run: `pytest test_prompts.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add prompts/validation_v1.txt
git commit -m "fix: add column format inheritance rule to validation prompt"
```

---

### Task 6: Add typo detection to final review prompt

**Files:**
- Modify: `prompts/final_review_v1.txt`

- [ ] **Step 1: Add typo detection sub-bullet**

In `prompts/final_review_v1.txt`, in section "### 4. READABILITY" (around line 41), after the bullet "- Narrative text is coherent and grammatically correct", add:

```
- Flag obvious typos and spelling errors (e.g., "on" instead of "one",
  "teh" instead of "the", missing words, repeated words like "the the").
  These are high-visibility errors in an IC-distributed document.
```

- [ ] **Step 2: Run existing prompt test**

Run: `pytest test_prompts.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add prompts/final_review_v1.txt
git commit -m "fix: add typo detection to final review prompt"
```

---

### Task 7: Run full test suite and verify

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest -x -v`
Expected: All tests PASS

- [ ] **Step 2: Verify branding still works on non-table text**

Run: `pytest test_updates_formatting.py::test_apply_branding -v`
Expected: PASS — SubheaderBox bold run gets "Pragmatica Bold", body run gets "Pragmatica Book"

- [ ] **Step 3: Verify layout normalization still works (margins, titles)**

Run: `pytest test_updates_formatting.py::test_normalize_layout -v`
Expected: PASS — shapes_clamped_to_margins >= 1

- [ ] **Step 4: Verify all prompts load cleanly**

Run:
```bash
python -c "
from memo_automator import MAPPING_PROMPT, VALIDATION_PROMPT, CONSISTENCY_PROMPT, FINAL_REVIEW_PROMPT
print('mapping:', len(MAPPING_PROMPT))
print('validation:', len(VALIDATION_PROMPT))
print('consistency:', len(CONSISTENCY_PROMPT))
print('final_review:', len(FINAL_REVIEW_PROMPT))
"
```
Expected: Four char counts printed, no errors

- [ ] **Step 5: Push branch and create PR**

```bash
git push -u origin fix/run-manifest-issues
gh pr create --title "fix: table font preservation + prompt hardening" --body "..."
```
