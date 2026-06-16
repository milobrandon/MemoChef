---
name: layout-integrity
description: Validate slide layout after editing — check for text overflow, text-image collisions, off-canvas content, native data tables that overflow their box, and proper continuation-slide formatting. Use before saving the deck on every run, especially when narrative additions, transcript-sourced bullets, or new/rebuilt tables may have changed shape sizes. Provides the procedure for tightening, splitting slides, resizing images, and fitting rebuilt native tables, plus the bounding-box validation pass that must run on every modified slide.
---

# Layout Integrity

Before saving the deck, walk through every slide you modified and fix any of the following. None of these are acceptable in the final output.

## 1. Text overflowing its container

If narrative additions push text past the bottom or sides of its placeholder, do NOT truncate, do NOT rely on auto-shrink to a sub-10pt size. Either:

a. Tighten the text you added (preserve every fact, just remove filler) if the overflow is under ~2 lines, OR
b. Split the slide: duplicate it, leave the cleanly-fitting content on the original, and move the overflowing section onto a new slide immediately after, titled `<Section> (cont.)`. Carry the same section banner, footer text, page-number style, and layout master onto the continuation slide.

## 1b. Native data tables overflowing their box

A table rebuilt from a proforma screenshot (per the `image-table-replacement` skill) can run taller or wider than the picture it replaced — long row labels wrap to two lines, or wide numeric columns push past the slide edge. For each rebuilt table:

a. **Too wide / numbers wrapping**: PowerPoint table cells wrap to the column width (word-wrap can't be disabled for cells). Re-size columns to fit the widest formatted value by setting each column's width explicitly (`table.columns[i].width = Inches(...)`), or step the body font down one size (e.g. 8 pt → 7 pt) for very large deals by reducing `run.font.size` on every body cell. Numbers must never split across two lines.
b. **Too tall**: reduce row label wrapping (widen the label column / abbreviate long labels to match the source), or drop the body font one step as in (a). If it still overflows, split the table at a section boundary onto a `(cont.)` slide as in (1) — keep whole sections together; never split mid-section.
c. The table must sit fully inside the original picture's bounding box (the `left`/`top`/`width`/`height` recorded when the Picture shape was removed) and clear the title and footer. Confirm fit with the bounding-box validation pass in section 5 before saving.

## 2. Text colliding with images, charts, or other shapes

If a text shape's bounding box overlaps a Picture or chart shape on the same slide after your edit:

a. First try shrinking the image — preserve aspect ratio, reduce until at least a 0.25" clear margin separates text and image.
b. If shrinking the image to less than ~60% of its original area would make it illegible, instead split the slide as in (1).
c. **Never** move an image off-slide, behind a text box, or off-canvas as a workaround.

## 3. Text running off the slide canvas

If any character is positioned outside the slide rectangle (off the left/right edges or below the bottom), the slide fails review. Resize the shape, reflow the text, or split the slide. Off-canvas text must never ship.

## 4. Continuity on split slides

Continuation slides MUST inherit from the original: section banner / category label, footer text, page-number style, layout master, and brand colors. Update the TOC to include any new continuation slide (see the `toc-maintenance` skill).

## 5. Validation before saving

For each slide you touched, programmatically iterate every shape on the slide:

- **Text shape**: check whether its rendered text fits its frame given its autofit setting. If python-pptx cannot give exact rendered metrics, be conservative — assume ~14 chars/inch at body size and leave 0.25" margins.
- **Picture / chart shape**: check bounding-box overlap against every text shape on the slide.
- **Table shape (including rebuilt native tables from section 1b)**: confirm the table's bounding box (`shape.left`, `shape.top`, `shape.width`, `shape.height`) sits fully inside the slide rectangle and inside the original picture's recorded box, and clears the title and footer placeholders.

Compare every shape's bounding box against the slide dimensions (`prs.slide_width`, `prs.slide_height`) and against the other shapes on the slide. If unsure, split rather than crowd. Never ship a slide you have not inspected after editing.

## Logging

Log every layout fix (slide split, image resize, table resize/font step-down, continuation insert) in the changelog under a dedicated `## Layout fixes` subsection so the reviewer can see what changed structurally, not just textually.
