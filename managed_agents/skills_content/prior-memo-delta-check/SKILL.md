---
name: prior-memo-delta-check
description: Reconcile every comparative reference in the IC memo against the PRIOR IC memo the user supplied. Use whenever the deck contains "increased/decreased from X to Y", "up/down from X", "cost per bed is up from", "yield improved N bps since the last update", "previously X", "vs. prior", "remains unchanged at X", or any "since the last update / approval" language — and whenever the user asks to "check the memo against the last IC memo" or "verify the from/to references". The FROM side of each claim must tie to the prior memo (PDF or PPTX under /mnt/session/uploads/), the TO side must tie to the current proforma, and the direction word (increased/decreased/improved/up/down) must match the math. Fix mismatched FROM values, TO values, direction words, and derived deltas. Run this AFTER all other table/text edits are applied, before the changelog.
---

# Prior-Memo Delta Cross-Check

IC project-update memos constantly compare against the previous update: "total beds have increased from 489 to 556", "cost per bed is down from $198K", "yield improved 15 bps since the last update". When the deck gets refreshed from a new proforma, the TO side gets updated but the FROM side silently drifts — it was typed against whatever the *previous* memo said, and nobody re-checks it. IC members DO re-check it, with the old memo in hand. That is why this step exists: every comparative claim must reconcile to two sources at once — the prior memo (FROM) and the current proforma (TO).

## Inputs

- **The prior IC memo**, uploaded by the user and mounted under `/mnt/session/uploads/` (typically a PDF or, less often, a PPTX). If no prior memo was uploaded, skip this skill — see the Changelog section.
- **The current proforma values** already extracted earlier in the run (the same numbers used to update the deck's tables and narratives).
- **The updated deck** — run this check AFTER all other table and text edits are applied, so you are checking the finished slides.

## Step 1 — Extract the prior memo's metrics

Find the prior memo in `/mnt/session/uploads/` and read it according to its type.

**PDF** — use `pdfplumber`: `extract_text()` per page for the prose, plus `extract_tables()` for the summary/budget tables. The metrics you need are concentrated on the project-update / executive-summary slides (usually the first ~10 pages) and the big end-of-memo data tables.

```python
import pdfplumber

prior_text = []
prior_tables = []
with pdfplumber.open("/mnt/session/uploads/PRIOR_IC_MEMO.pdf") as pdf:
    for page in pdf.pages:
        prior_text.append(page.extract_text() or "")
        prior_tables.extend(page.extract_tables() or [])
```

**PPTX** — use `python-pptx`: walk every slide, pull text frames and table cells.

```python
from pptx import Presentation

prior = Presentation("/mnt/session/uploads/PRIOR_IC_MEMO.pptx")
prior_text, prior_tables = [], []
for slide in prior.slides:
    for shape in slide.shapes:
        if shape.has_text_frame:
            prior_text.append(shape.text_frame.text)
        if shape.has_table:
            prior_tables.append([[c.text for c in row.cells] for row in shape.table.rows])
```

Build a metric map for at least these (capture more if present):

| Metric | Typical labels |
|---|---|
| Total beds | "beds", "bed count" |
| Total units | "units" |
| GSF / NSF | "gross SF", "net rentable SF" |
| Amenity SF | "amenity" |
| Total development cost | "TDC", "total project cost", "development budget" |
| Cost per bed | "cost/bed", "per bed" |
| Yield / ROC | "yield on cost", "untrended yield", "return on cost" |
| Levered IRR (3-yr) | "IRR" |
| Equity multiple | "MOIC", "equity multiple" |
| Avg rent per bed | "rent/bed", "chunk rent" |
| Parking stalls | "parking" |
| Key dates | closing, construction start, delivery/CO |

Where a metric appears in several places in the prior memo, prefer the executive-summary table value — that's what an IC reader will quote back.

## Step 2 — Find every comparative reference in the updated deck

Scan all slide text (and table cells) in the updated deck for comparative language. Patterns to hunt — case-insensitive, numbers may be formatted ($, commas, %, "K"/"M"):

- "increased/decreased/grew/declined/rose/fell from **X** to **Y**"
- "up/down from **X**", "up/down by **N**", "**N** higher/lower than"
- "previously **X**", "was **X** at the last update", "vs. **X** prior"
- "since the last update/approval", "compared to the prior memo"
- "remains/unchanged at **X**" (this is a comparison too — verify against BOTH sources; "unchanged" is wrong if the proforma moved)

Slides titled "Project Update", "Changes Since…", "Summary of Changes", or similar are dense with these — read them line by line, not just by regex.

## Step 3 — Verify each reference against BOTH sources

For each comparative claim:

1. **FROM side = prior memo.** The old value quoted must match the prior memo's value for that metric (respect rounding: if the prior memo said "$68.8M", a FROM of "$68.8M" is right even if its underlying number was $68,769,750).
2. **TO side = current proforma.** Must match the current proforma value you extracted earlier in the run (it usually already matches, since the table/text pass updated it — this catches the ones that pass missed because they live inside prose).
3. **Direction word matches the math.** "decreased from $1,349 to $1,389" is an increase — fix the verb. Same for improved/worsened, higher/lower.
4. **Derived deltas recompute.** "up 67 beds" must equal TO − FROM; "a 12% increase" must equal (TO − FROM) / FROM at the deck's rounding.

Fix what fails. Mismatched FROM values get replaced with the prior memo's number; mismatched TO values with the proforma's; wrong verbs get flipped; wrong deltas recomputed. When you edit a cell or run, preserve run-level font formatting per the `memo-table-updates` skill (never `cell.text =` / `tf.text =`). If a sentence's meaning changes materially, apply the narrative-rewriting rules from the main Memo Chef workflow.

## Step 4 — Handle the gaps honestly

- **Metric not in the prior memo** (e.g. the deck claims "amenity SF up from 5,100" but the prior memo never stated amenity SF): leave the text, log a WARNING in the changelog naming the metric and what you searched for.
- **Ambiguous prior value** (two candidate numbers, e.g. trended vs untrended yield): apply the main workflow's untrended-default rule; if still ambiguous, warn instead of guessing.
- **Sensitivity tables stay untouched**, same as everywhere else.

## Step 5 — Changelog

Add a "Prior-memo delta cross-check" section to the changelog listing:

- the prior memo file used (full `/mnt/session/uploads/` path),
- every comparative reference checked (slide #, metric, FROM/TO, verdict),
- each fix applied (before → after, full text, not truncated),
- warnings for unverifiable references.

If no prior memo was provided, the section is one line: "Skipped — no prior memo provided."
