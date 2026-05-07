---
name: memo-changelog
description: Author the per-run changelog.md and run the seven-point self-consistency audit before finalizing it. Use at the end of every Memo Chef run before writing /mnt/session/uploads/changelog.md. Covers the required structure (totals, per-change entries with full before/after text, warnings, summary stats), the formatting rules that prevent truncated diffs, and the audit checks (numbered changes, header tally, source attribution, cross-slide consistency, layout-fixes subsection).
---

# Memo Changelog

Write a detailed changelog to `/mnt/session/uploads/changelog.md` with:

- Total updates applied (by category: table, text, narrative, chart, row insert).
- List of each change: page number, what changed, old value → new value, source.
- Any warnings (skipped sensitivity tables, unmatched metrics, etc.).
- Summary statistics.

## Changelog formatting rules (IMPORTANT)

- **Before/after text must be COMPLETE, not truncated.** For every narrative or text change, show the full old text and the full new text. Do NOT cut off mid-sentence. Do NOT end a quoted value with a dangling conjunction ("as", "and", "which", "the", "a"). Do NOT use ellipses ("...") to shorten a before/after diff. If the new value is a 60-word paragraph, all 60 words appear in the changelog.
- For multi-sentence narrative updates, use a fenced markdown quote block or a multi-line code block so the full text renders cleanly rather than trying to inline it on a single line.
- The short header describing a change may be brief (e.g. "Entitlements narrative — TRC meeting added"), but the before/after body values must be complete text.
- A reviewer reading only the changelog should be able to reconstruct exactly what changed in the deck without having to open the pptx.

## Self-Consistency Audit (REQUIRED before finalizing)

Before producing the final `changelog.md`, audit it. If any check below fails, re-do the affected portion of the changelog before finalizing.

1. **Numbered changes consistent.** Every "Change N of M" header must have N ≤ M, and M must equal the actual count of changes in the body. If you added or split a change after the first numbering pass, re-number from 1. There must never be a "Change 7 of 6".

2. **Header tally matches body.** The opening "Total Updates: X text blocks across Y slides" line must equal:
   - X = the number of "Change N of M" sections in the body, and
   - Y = the count of distinct slide numbers cited across those sections.

3. **Summary statistics table.** The closing summary's "Slides modified: K (slides A, B, C, ...)" — K must equal the number of distinct slide numbers in the parenthesized list. Recount the list by hand before finalizing.

4. **Source attribution complete.** Every change must cite either (meeting name + date) for transcript-sourced edits or a specific proforma cell / sheet location for proforma-sourced edits. If you cannot cite, do not make the change.

5. **Numeric preservation.** Original financial figures (dollar amounts, percentages, bed counts, IRRs, equity multiples) that you did not explicitly change because of new evidence must appear unchanged in the output deck and unchanged in any "Before" text in the changelog.

6. **Cross-slide consistency.** If two slides reference the same fact (e.g., "stucco premium ~$1M", "Schematic Design starts June 2026"), they must use the same number / date / phrasing and the same hedging language ("approximately", "expected to", "targeting") across slides.

7. **Layout fixes logged.** Any slide split, image resize, or continuation slide created during the layout-integrity check must appear in a `## Layout fixes` subsection of the changelog.

Run the audit explicitly: walk each numbered check above, state PASS or FAIL inline in your scratch reasoning, and only emit the final `changelog.md` once all seven are PASS.
