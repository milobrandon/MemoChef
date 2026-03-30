# Run Manifest Fixes — Design Spec

**Date:** 2026-03-30
**Trigger:** Knoxville-Cumberland run (run_id `5289f303`) scored 62 on final review with 7 actionable issues.

## Issues Addressed

| # | Issue | Stage | Workstream |
|---|-------|-------|------------|
| 1 | Table font sizes forced to global dominant size (Pages 13, 16) | layout | A |
| 2 | Table font families changed (Bold → Book on subheaders) | branding | A |
| 3 | GSF mismatch passed through — 14,460 vs 12,663 (Page 3) | mapping | B |
| 4 | Unit/bed count confusion in unit mix (Page 17) | mapping | B |
| 5 | Consistency check false positives (4 of 10 warnings) | consistency_check | B |
| 6 | Missing $ sign not caught (Page 25) | validation | B |
| 7 | Pre-existing typo not flagged — "on" → "one" (Page 15) | final_review | C |

## Workstream A: Table Font Preservation

### Problem

Two code paths modify table cell fonts when they shouldn't:

1. **`normalize_layout()`** (lines 4113-4144): Computes a single global dominant font size across ALL tables, then forces every table cell within 0.5-4pt of that size to match. Dense comp tables on Pages 13/16 use intentionally smaller fonts that get overwritten.

2. **`apply_branding()`** (lines 3667-3685): Calls `_reformat_run()` on every table cell run, which remaps font families. Subheaders using Pragmatica Bold get changed to Pragmatica Book because the bold-detection heuristic misclassifies them.

### Design

**Rule: Never modify font family or font size inside table cells.**

#### Change 1: Remove table font size normalization

Delete the entire block at lines 4113-4144 in `normalize_layout()`. Set `summary["table_font_size_normalized"] = 0` as a no-op so downstream code that reads this counter doesn't break.

#### Change 2: Skip `_reformat_run()` for table cells

In `apply_branding()`, remove the table cell processing loop (lines 3667-3685). Tables already use `skip_color=True` to preserve colors — now we extend that principle to skip font changes entirely.

### Files Changed

- `memo_automator.py`: `normalize_layout()`, `apply_branding()`

### Risk

Low. Both changes are deletions of behavior. No new logic introduced.

## Workstream B: Prompt Hardening

### Issue 3: GSF Mismatch (mapping prompt)

**Root cause:** The mapping prompt says "preserve formatting conventions" but doesn't require tying every SF figure back to a specific proforma row. Claude mapped 14,460 GSF from somewhere (possibly a stale memo value or a different proforma line) without cross-referencing the Executive Summary row 21 value of 12,663.

**Fix:** Add instruction to `prompts/mapping_v1.txt` after rule 3 (derived values):

> **SF verification** — For every square footage value you emit (GSF, NSF, amenity SF, leasing SF, retail SF), trace it to a specific proforma row and include that row reference in the `source` field. If no proforma row matches, do NOT emit the update — leave the memo's existing value in place. Never propagate a memo value that has no proforma source.

### Issue 4: Unit/Bed Confusion (mapping prompt)

**Root cause:** Despite rules 5 and the parenthetical guidance, Claude still confuses total bed counts with unit counts in unit mix tables. The Page 17 error shows 4BR/2BA (84) which is `21 units × 4 beds = 84 beds` used where the memo expected the unit count (21).

**Fix:** The existing rules 5 and the parenthetical sub-rules already address this. The issue is reinforcement — Claude occasionally drops these rules on long prompts. Add a boxed reminder at the end of the instruction section (before the JSON schema):

> **UNIT vs BED COUNTS — FINAL CHECK:** Before emitting your JSON, review every unit mix update. Parenthetical numbers like "(84)" = total beds for that type (units × beds_per_unit). The separate "Units" or "# Units" column = raw unit count. The "# Beds" or "Bed Count" row = beds for that type. These are DIFFERENT numbers. Getting them backwards is a critical error.

### Issue 5: Consistency Check False Positives

**Root cause:** The consistency check prompt tells Claude to verify values "match EXACTLY" but doesn't tell it to skip reporting when they do match. Claude is over-reporting — listing matches as discrepancies when expected and found are identical (e.g., `expected '$8,953,746', found '$8,953,746'`).

**Fix:** Add instruction to `prompts/consistency_check_v1.txt` after the 5-point audit list:

> **CRITICAL: Do NOT report false positives.** Before adding any discrepancy, compare your `expected` and `found` values character-by-character. If they are identical (same text, same formatting), do NOT include that item in the discrepancies array. Only report genuine mismatches where the values differ. A discrepancy where expected equals found is a bug in YOUR output, not a bug in the memo.

### Issue 6: Missing $ Sign (validation prompt)

**Root cause:** The validation prompt checks for duplicate detection and format consistency but doesn't flag cells that break their column's formatting convention. On Page 25, "22,960,805" appeared in a column where every other cell had a "$" prefix.

**Fix:** Add instruction to `prompts/validation_v1.txt` in the formatting verification section:

> **Column format inheritance** — For each table column, check whether values in the same column consistently use a formatting prefix (e.g., "$") or suffix (e.g., "%"). If a proposed `new_value` omits a prefix/suffix that every other cell in that column uses, add a `correction` entry that includes the prefix/suffix. Likewise, if an existing memo value already violates column formatting and is NOT being updated, emit it as a `missed` entry.

### Files Changed

- `prompts/mapping_v1.txt`: Add SF verification rule and unit/bed final check
- `prompts/consistency_check_v1.txt`: Add false positive guard
- `prompts/validation_v1.txt`: Add column format inheritance rule

### Risk

Medium. Prompt changes affect Claude's behavior probabilistically. These are all additive instructions (no existing rules removed), so regression risk is low. The false positive fix is the highest-confidence improvement.

## Workstream C: Typo Detection in Final Review

### Problem

Pre-existing typo "represents on of the few" on Page 15 was not flagged. The final review prompt's "Readability" section says "Narrative text is coherent and grammatically correct" but doesn't explicitly call out typos and spelling errors.

### Fix

Add to the Readability section (item 4) in `prompts/final_review_v1.txt`:

> - Flag obvious typos and spelling errors (e.g., "on" instead of "one", "teh" instead of "the", missing words, repeated words). These are high-visibility errors in an IC-distributed document.

### Files Changed

- `prompts/final_review_v1.txt`: Expand readability checklist

### Risk

Low. This only adds a sub-bullet to an existing checklist item.

## Out of Scope

- **Cover page project name** — intentionally shortened by analyst, not a bug.
- **Streaming fix** — already merged in PR #37.
- **Market data pipeline stages 4-6** — already merged in PR #37.

## Testing

1. Re-run the Knoxville-Cumberland files after changes and compare:
   - `table_font_size_normalized` should be 0
   - Font families on Pages 13 and 16 should be unchanged
   - GSF value on Page 3 should match proforma (12,663)
   - Consistency check false positives should drop from 4 to 0
   - Page 25 Soft Costs should include "$" prefix
   - Page 15 typo should appear in final review warnings
2. Run existing test suite (`pytest`) to verify no regressions
3. Spot-check that non-table text still gets branded correctly
