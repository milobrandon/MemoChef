#!/usr/bin/env python3
"""
Memo Automator
==============
Automatically updates an Investment Committee (IC) PowerPoint memo with
metrics from an Excel proforma.

Workflow
--------
1. Create a backup of the original memo.
2. Extract data from specified proforma tabs (openpyxl, data_only).
3. Extract text / tables from ALL slides in the memo (python-pptx).
4. Send both datasets to the Claude API, which identifies every metric that
   should be updated and returns structured JSON mappings.
5. Apply the text / table updates to the memo (python-pptx).
6. Save the updated memo and write a detailed change-log.

Usage
-----
    python memo_automator.py <memo.pptx> <proforma.xlsx>
    python memo_automator.py <memo.pptx> <proforma.xlsx> --config my_config.yaml
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import anthropic
import openpyxl
import yaml
from dotenv import load_dotenv
from pptx import Presentation

# ============================================================================
# LOGGING SETUP
# ============================================================================
LOG_FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("memo_automator")


# ============================================================================
# 1. ARGUMENT PARSING
# ============================================================================
def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Update an IC memo with metrics from a proforma."
    )
    p.add_argument("memo", help="Path to the PowerPoint memo (.pptx)")
    p.add_argument("proforma", help="Path to the Excel proforma (.xlsx / .xlsm)")
    p.add_argument(
        "--config", "-c",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to YAML config file (default: config.yaml beside this script)",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory for output artifacts (default: same folder as memo)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying the memo",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the Claude validation pass for faster runs",
    )
    p.add_argument(
        "--property-name",
        default="",
        help="Property name as shown in the proforma (helps match rebranded names)",
    )
    return p.parse_args()


# ============================================================================
# 2. CONFIGURATION
# ============================================================================
def load_config(config_path: str) -> dict:
    """Load and validate the YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Apply defaults for missing keys
    cfg.setdefault("proforma", {})
    cfg["proforma"].setdefault("tabs", [
        "Executive Summary", "Development Summary", "Cash Flow",
        "Assumptions", "Proforma Comparison"
    ])
    cfg["proforma"].setdefault("max_rows_per_tab", 250)
    cfg["proforma"].setdefault("max_cols_per_tab", 30)
    cfg.setdefault("memo", {})
    cfg["memo"].setdefault("pages", "all")
    cfg.setdefault("schedule", {})
    cfg["schedule"].setdefault("max_tasks", 500)
    cfg.setdefault("claude", {})
    cfg["claude"].setdefault("model", "claude-sonnet-4-6")
    cfg["claude"].setdefault("validation_model", cfg["claude"]["model"])
    cfg["claude"].setdefault("max_tokens", 16000)
    cfg["claude"].setdefault("temperature", 0)
    return cfg


# ============================================================================
# 3. BACKUP
# ============================================================================
def create_backup(memo_path: str, output_dir: str) -> str:
    """Copy the original memo to a timestamped backup file."""
    stem = Path(memo_path).stem
    ext = Path(memo_path).suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{stem}_BACKUP_{ts}{ext}"
    backup_path = os.path.join(output_dir, backup_name)
    shutil.copy2(memo_path, backup_path)
    log.info("Backup created: %s", backup_path)
    return backup_path


# ============================================================================
# 4. PROFORMA DATA EXTRACTION  (openpyxl, data_only=True)
# ============================================================================
def extract_proforma_data(proforma_path: str, cfg: dict) -> str:
    """
    Read the proforma workbook and return a compact text representation
    of every non-empty cell on the specified tabs.

    Uses data_only=True so formulas resolve to their cached values.
    """
    tabs = cfg["proforma"]["tabs"]
    max_rows = cfg["proforma"]["max_rows_per_tab"]
    max_cols = cfg["proforma"]["max_cols_per_tab"]

    log.info("Opening proforma (data_only): %s", proforma_path)
    wb = openpyxl.load_workbook(proforma_path, data_only=True)
    log.info("Available sheets: %s", wb.sheetnames)

    lines = []
    for tab_name in tabs:
        if tab_name not in wb.sheetnames:
            log.warning("Tab '%s' not found in proforma - skipping", tab_name)
            continue
        ws = wb[tab_name]
        lines.append(f"\n{'='*70}")
        lines.append(f"TAB: {tab_name}")
        lines.append(f"{'='*70}")

        end_row = ws.max_row if max_rows == 0 else min(ws.max_row, max_rows)
        end_col = ws.max_column if max_cols == 0 else min(ws.max_column, max_cols)

        for row in ws.iter_rows(min_row=1, max_row=end_row, max_col=end_col,
                                values_only=False):
            row_data = []
            for cell in row:
                if cell.value is not None:
                    row_data.append(str(cell.value))
            if row_data:
                lines.append(f"Row {row[0].row}:\t" + "\t".join(row_data))

    wb.close()
    proforma_text = "\n".join(lines)
    log.info("Proforma extraction complete (%d lines, %d chars)",
             len(lines), len(proforma_text))
    return proforma_text


# ============================================================================
# 4b. SCHEDULE DATA EXTRACTION  (mpxj via jpype)
# ============================================================================
def _ensure_jvm():
    """Start the JVM once for mpxj access. No-op if already running."""
    import jpype
    if jpype.isJVMStarted():
        return

    import mpxj
    jars_dir = os.path.join(os.path.dirname(mpxj.__file__), "lib")
    # Build classpath from all jars in the mpxj lib directory
    jars = [os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")]
    classpath = os.pathsep.join(jars)

    jpype.startJVM(classpath=[classpath])
    log.info("JVM started with %d jars from %s", len(jars), jars_dir)


def extract_schedule_data(schedule_path: str, cfg: dict) -> str:
    """
    Read a Microsoft Project (.mpp) schedule and return a hierarchical text
    representation of tasks with dates and durations.

    Uses mpxj (via jpype) to parse the .mpp file.
    """
    _ensure_jvm()

    import jpype
    from java.io import File as JavaFile

    max_tasks = cfg.get("schedule", {}).get("max_tasks", 500)

    log.info("Opening schedule: %s", schedule_path)
    reader = jpype.JClass("net.sf.mpxj.reader.UniversalProjectReader")()
    project = reader.read(JavaFile(schedule_path))

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("SCHEDULE DATA (from Microsoft Project)")
    lines.append(f"{'='*70}")

    task_count = 0
    for task in project.getTasks():
        if task_count >= max_tasks:
            lines.append(f"... (truncated at {max_tasks} tasks)")
            break

        name = str(task.getName()) if task.getName() else ""
        # Skip L0 unnamed separator tasks (grouping containers)
        outline_level = task.getOutlineLevel()
        if outline_level is not None:
            level = int(str(outline_level))
        else:
            level = 0
        if level == 0 and not name.strip():
            continue

        # Get dates and duration
        start = task.getStart()
        finish = task.getFinish()
        duration = task.getDuration()

        start_str = str(start).split("T")[0] if start else "N/A"
        finish_str = str(finish).split("T")[0] if finish else "N/A"

        if duration:
            dur_str = str(duration)
        else:
            dur_str = "0d"

        # Milestone detection
        is_milestone = task.getMilestone() if task.getMilestone() is not None else False
        milestone_tag = "  [MILESTONE]" if is_milestone else ""

        indent = "  " * max(level - 1, 0)
        level_tag = f"[L{level}]"

        lines.append(
            f"{indent}{level_tag} {name}{milestone_tag}  |  "
            f"Start: {start_str}  |  Finish: {finish_str}  |  Dur: {dur_str}"
        )
        task_count += 1

    schedule_text = "\n".join(lines)
    log.info("Schedule extraction complete (%d tasks, %d chars)",
             task_count, len(schedule_text))
    return schedule_text


# ============================================================================
# 5. MEMO CONTENT EXTRACTION  (python-pptx)
# ============================================================================
def extract_memo_content(memo_path: str, cfg: dict) -> str:
    """
    Read the PowerPoint memo and return a structured text representation
    of shapes (tables, text boxes) on the target pages.

    If cfg["memo"]["pages"] is "all", scans every slide in the deck.
    Otherwise expects a list of 1-based page numbers.
    """
    prs = Presentation(memo_path)
    total_slides = len(prs.slides)

    # Determine which pages to scan
    pages_cfg = cfg["memo"]["pages"]
    if pages_cfg == "all":
        page_numbers = list(range(1, total_slides + 1))
    else:
        page_numbers = [int(p) for p in pages_cfg]

    log.info("Scanning %d pages (out of %d total slides)", len(page_numbers), total_slides)

    lines = []
    for page_num in page_numbers:
        idx = page_num - 1
        if idx >= total_slides:
            log.warning("Page %d does not exist (only %d slides)", page_num, total_slides)
            continue

        slide = prs.slides[idx]
        lines.append(f"\n{'='*70}")
        lines.append(f"PAGE {page_num}  (slide index {idx})")
        lines.append(f"{'='*70}")

        for si, shape in enumerate(slide.shapes):
            lines.append(f"\n--- Shape {si}: type={shape.shape_type}, "
                         f"name='{shape.name}' ---")
            lines.append(f"    Position: left={shape.left}, top={shape.top}, "
                         f"width={shape.width}, height={shape.height}")

            # Text frames (text boxes, placeholders)
            if shape.has_text_frame:
                for pi, para in enumerate(shape.text_frame.paragraphs):
                    text = para.text.strip()
                    if text:
                        lines.append(f"    Para {pi}: '{text}'")

            # Tables
            if shape.has_table:
                tbl = shape.table
                lines.append(f"    Table: {len(tbl.rows)} rows x "
                             f"{len(tbl.columns)} cols")
                for ri, row in enumerate(tbl.rows):
                    cells = []
                    for ci, cell in enumerate(row.cells):
                        ct = cell.text.strip()
                        if ct:
                            cells.append(f"[{ci}]={ct}")
                    if cells:
                        lines.append(f"    Row {ri}: {' | '.join(cells)}")

    memo_text = "\n".join(lines)
    log.info("Memo extraction complete (%d lines, %d chars)",
             len(lines), len(memo_text))
    return memo_text


# ============================================================================
# 6. MEMO CONTENT CHUNKING
# ============================================================================
def chunk_memo_by_pages(memo_content: str, pages_per_chunk: int = 10) -> list:
    """
    Split memo content into chunks of up to pages_per_chunk pages each.

    Used when the full prompt would exceed the model's output token limit.
    Each chunk is processed in a separate API call and the results are merged.
    """
    # Each page block starts with the === PAGE N === header
    page_blocks = re.split(r"(?=\n={60,}\nPAGE \d+)", memo_content)
    chunks = []
    for i in range(0, len(page_blocks), pages_per_chunk):
        chunk = "".join(page_blocks[i:i + pages_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ============================================================================
# 7. CLAUDE API — METRIC MAPPING
# ============================================================================
MAPPING_PROMPT = """\
You are a financial analyst assistant. Your task is to compare data from an
Excel proforma with content from a PowerPoint IC memo and identify EVERY
metric in the memo that should be updated with new values from the proforma.

## Proforma Data (from Excel)
{proforma_data}

## Memo Content (from PowerPoint)
{memo_content}

## Instructions

1. **Table updates** — For each table cell in the memo whose value comes
   from the proforma, emit an entry with the exact old text to find and the
   exact new text to replace it with.  Preserve formatting conventions
   (commas in numbers, dollar signs, percent signs, decimal precision).

2. **Text updates** — For narrative / paragraph text that embeds a proforma
   metric (e.g. "Bed count has increased ... to 510 beds"), emit an entry with
   the minimal old text snippet and its replacement.

3. **Derived / calculated values** — When a source value changes, check
   whether derived values in the memo should also change: totals, subtotals,
   ratios (e.g. parking ratio, cost per bed/unit), per-bed or per-unit
   metrics, summed pipeline beds/units, and sensitivity-table outputs.
   Perform the arithmetic and emit updates for each derived value.

4. **Pipeline / comp summary tables (row-oriented)** — Tables where each
   ROW is a property (columns = metrics like units, beds, year, rent).
   The first data row is always the subject property.  Update the subject
   property's row to match the proforma — units, beds, delivery year,
   address, rents, ratios.  If the subject appears under a prior project
   name (e.g. a rebranded project in the same row position), update that
   row too.

5. **Competitive set side-by-side tables (column-oriented)** — Tables
   where each COLUMN is a property and rows are metrics (unit size, beds,
   market rent per unit type).  The subject property is ALWAYS the
   leftmost data column.  Update EVERY cell in that column — header
   metrics (units, beds, parking, ratios) AND each unit-type sub-section
   (unit size, bed count, market rent).
   - **Unit mix source:** Use the unit mix near the TOP of the Assumptions
     tab (the detailed breakdown), NOT the unit mix summary.
   - **Row matching:** Place each proforma unit type in the row whose
     bedroom label matches (e.g. a "1BR" proforma unit goes in the "1BR"
     row).  If the table has multiple rows for one bedroom type (e.g. two
     "4BR" rows), distribute proforma units to the row whose competitive-
     set values (size, rent) are the closest match.
   - **Overflow (more proforma types than slots):** When there are more
     proforma unit types than the table has rows for that bedroom count,
     combine the two most similar in size into a range for both size and
     rent (e.g. "490 - 550 sf" and "$1,325 - $1,435").
   - When the memo shows a range but the proforma has a single value,
     replace the range with the proforma value.

6. **Address and name consistency** — If the proforma's property address or
   name differs from the memo, update every occurrence in both tables and
   narrative text to match the proforma.

7. **IRR selection** — When the executive summary or other sections reference
   an IRR (e.g. "3 YR IRR", "3-Yr IRR"), use the **3-year holding-period
   Levered IRR** from the proforma.  Do NOT use the 1-year or 4-year IRR.
   The 1-year IRR is typically much higher (>30%); the 3-year IRR is usually
   in the 20–28% range.  Only use a different horizon if the memo cell
   explicitly labels itself as "1 YR" or "4 YR".

8. **"Chunk rents" and untrended values** — When the memo refers to
   "chunk rents per month" or simply "rents" without a trend qualifier,
   it means today's **untrended** rents from the proforma.  Likewise,
   "controllable OpEx per bed" and "OpEx ratio" default to **untrended**
   values unless the memo explicitly says "trended."

9. **Comp summary / comp rollup (row-oriented)** — The **top row** (first
   data row) is always the subject property and MUST be updated from the
   proforma.  If the name in that row differs from the proforma's property
   name, replace it so the name matches the proforma.

10. **Pipeline table** — Same rule: the **top row** is the subject property.
    Update all its metrics and swap its name to match the proforma if they
    differ.

11. **Underwriting projections / end-of-memo tables** — Pages near the end
    of the memo that contain tables WITHOUT narrative text (pure data
    tables under headings like "Underwriting Projections") MUST also be
    updated.  Do NOT skip these pages — scan and update every metric that
    has a proforma source.

12. **Sensitivity analysis tables — DO NOT UPDATE** — Any table whose heading
    or surrounding text contains "Sensitivity" (e.g. "Untrended YOC Sensitivity
    Analysis", "IRR Sensitivity") must be SKIPPED entirely. These require
    matrix recalculation the tool cannot perform. Emit zero updates for them.

13. **Preserve strategic / aspirational language** — Narrative text containing
    phrases like "targeting", "driving towards", "our goal is", "we are
    aiming for", "we anticipate", or "we expect" represents outside knowledge
    or team targets, NOT proforma data points. Do NOT overwrite these values.
    Only update values presented as factual statements of the current
    underwriting (e.g. "the project contains 510 beds").

14. **Pipeline delivery year and land acquisition date** — In pipeline tables,
    update the subject property's **delivery year** from the proforma's
    construction completion / certificate of occupancy date (year portion),
    and the **land acquisition / closing date** from the proforma's land
    closing date. Emit table_updates for both.

15. **Development budget "% of Total" columns** — When a dollar amount in the
    development budget changes, ALSO recalculate and update the corresponding
    "% Total" or "% of Total" column: new_pct = (new_dollar / total_budget)
    × 100, formatted to match the memo's decimal precision (e.g. "1.3%").

16. **Chunk rents — average rent per bed** — The "Chunk Rents Per Month" row
    in comparison tables AND any standalone "Avg Rent/Bed" or "Average Rent
    Per Bed" reference in tables or narrative MUST be mapped to the proforma's
    untrended average rent per bed. Do NOT skip this metric.

17. **Missing unit types — row_inserts** — When the proforma has unit types
    (e.g. Studio, 2BR/1BA) that do NOT exist in the memo's unit mix table
    or side-by-side comp table, emit a `row_inserts` entry to add them.
    - **Unit mix table:** One row per missing type. Populate ALL columns
      from the proforma. Place after the most similar existing row
      (e.g. "Studio" before "1BR", "2BR/1BA" after "1BR").
    - **Side-by-side comp table:** Insert a 3-row section (unit size,
      # of beds, market rent) for each missing type. Populate ONLY the
      subject property column; leave all other columns EMPTY (comps
      have no data for these types). Place after the last unit section.
    - Set `insert_after_row_label` to column 0 text of the row to insert AFTER.

18. **Schedule milestones and dates** — When schedule data is provided below the
    proforma data (under "SCHEDULE DATA"), update project timeline tables and
    narrative date references in the memo:
    - **Entitlement approval** start/finish dates (e.g., "Level II Development Plan",
      "Governmental Approvals", "Zoning", any task containing "entitlement")
    - **Building permit** start/finish dates (tasks containing "permit")
    - **Project closing / anticipated closing** dates
    - **Construction start/end** — including sub-phases: abatement, demolition,
      foundation, and vertical construction dates
    - **CO / Certificate of Occupancy** and **substantial completion** dates
    - **Move-in / delivery** dates
    - Dates in narrative text (e.g., "construction is expected to begin in Q2 2027")
      should be updated to match the schedule, using the same format as the memo
      (e.g., "Q2 2027", "April 2027", "4/5/2027")
    - For contract/earnest money milestones: update deposit amounts and when money
      goes hard (non-refundable) dates if they appear in the memo
    - Use the `source` field format: "Schedule: [task name]"

CRITICAL: Return ONLY the raw JSON object below. Do NOT include any analysis,
reasoning, explanation, or commentary before or after the JSON. Your entire
response must be parseable as JSON. Start with {{ and end with }}.

Schema:

{{
  "table_updates": [
    {{
      "page": <int>,
      "table_name": "<shape name>",
      "row_label": "<text in column 0 of the row>",
      "column_index": <int, 0-based>,
      "old_value": "<exact substring to find>",
      "new_value": "<exact replacement>",
      "source": "<tab name + cell ref>"
    }}
  ],
  "text_updates": [
    {{
      "page": <int>,
      "old_text": "<exact substring to find>",
      "new_text": "<replacement>",
      "source": "<tab name + cell ref>"
    }}
  ],
  "row_inserts": [
    {{
      "page": <int>,
      "table_name": "<shape name>",
      "insert_after_row_label": "<text in column 0 of reference row>",
      "cells": ["<col 0>", "<col 1>", ...],
      "source": "<tab + cell ref>"
    }}
  ]
}}

Important:
- ONLY include entries where the new value DIFFERS from the old value.
  If a memo metric already matches the proforma, skip it entirely.
- old_value / old_text must be the EXACT text currently in the memo.
- new_value / new_text must match the memo's formatting style.
- For dollar amounts use commas: $68,769,750 not $68769750.
- For percentages keep the same decimal places as the memo uses.
- Do NOT fabricate data, but DO perform arithmetic to update derived values
  whose inputs have changed (totals, ratios, per-unit metrics, etc.).
- Scan ALL pages including large data tables (e.g. executive summary,
  cash flow, unit mix, development budget tables that may span full pages).
- For row_inserts, cells must have exactly as many elements as the target
  table has columns. Use "" for empty cells.
{property_name_section}"""


def _salvage_truncated_json(raw: str) -> dict | None:
    """
    Attempt to recover valid mappings from a truncated JSON response.

    When Claude hits max_tokens mid-JSON, the response ends abruptly.
    This function tries to close the JSON by finding the last complete
    entry boundary (a '},' or '}]' pattern) and appending the missing
    closing brackets.

    Returns a valid mappings dict if salvageable, None otherwise.
    """
    text = raw.strip()
    if not text:
        return None

    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Must start with a JSON object
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    # Strategy: try a few common closing patterns to complete the JSON.
    # The response is typically:
    #   {"table_updates": [..., {last_complete}, {partial   <-- cut here
    # or:
    #   {"table_updates": [...], "text_updates": [..., {partial   <-- cut here

    closings_to_try = [
        # Already inside a text_updates array — close entry + array + object
        (r'\}\s*,?\s*$', '}], "text_updates": []}'),       # mid table_updates array entry boundary
        (r'\}\s*\]\s*,?\s*$', '], "text_updates": []}'),   # end of table_updates array
        (r'\}\s*,?\s*$', '}]}'),                            # mid text_updates array entry boundary
        (r'\}\s*\]\s*,?\s*$', ']}'),                        # end of text_updates array
        (r'\}\s*\]\s*\}\s*$', None),                        # already complete
    ]

    # Find the last complete entry: look for the last '}' that's followed
    # by ',' or ']' or is at a natural boundary
    # Walk backwards to find last complete "}" entry
    last_good = -1
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == '}':
            if depth == 0:
                last_good = i
                break
            depth -= 1
        elif ch == '{':
            depth += 1

    if last_good == -1:
        return None

    truncated = text[:last_good + 1]

    # Count unmatched brackets to determine what closings are needed
    open_braces = truncated.count('{') - truncated.count('}')
    open_brackets = truncated.count('[') - truncated.count(']')

    if open_braces < 0 or open_brackets < 0:
        return None

    # Build closing sequence: close all open brackets then braces
    closing = ']' * open_brackets + '}' * open_braces
    candidate = truncated + closing

    try:
        mappings = json.loads(candidate)
        mappings.setdefault("table_updates", [])
        mappings.setdefault("text_updates", [])
        mappings.setdefault("row_inserts", [])
        n = len(mappings["table_updates"]) + len(mappings["text_updates"]) + len(mappings["row_inserts"])
        if n > 0:
            log.info("Salvaged %d updates from truncated response", n)
            return mappings
        return None
    except json.JSONDecodeError:
        pass

    # Fallback: try trimming back to last '},' boundary (complete array entry)
    last_comma_boundary = truncated.rfind('},')
    if last_comma_boundary == -1:
        return None

    truncated2 = truncated[:last_comma_boundary + 1]
    open_braces2 = truncated2.count('{') - truncated2.count('}')
    open_brackets2 = truncated2.count('[') - truncated2.count(']')
    if open_braces2 < 0 or open_brackets2 < 0:
        return None

    closing2 = ']' * open_brackets2 + '}' * open_braces2
    candidate2 = truncated2 + closing2

    try:
        mappings = json.loads(candidate2)
        mappings.setdefault("table_updates", [])
        mappings.setdefault("text_updates", [])
        mappings.setdefault("row_inserts", [])
        n = len(mappings["table_updates"]) + len(mappings["text_updates"]) + len(mappings["row_inserts"])
        if n > 0:
            log.info("Salvaged %d updates from truncated response (fallback)", n)
            return mappings
        return None
    except json.JSONDecodeError:
        log.debug("Could not salvage truncated JSON response")
        return None


def _parse_json_response(raw: str) -> dict | None:
    """
    Parse a JSON object from a Claude API response, handling common
    noise: markdown fences, trailing commentary, whitespace.

    Returns the parsed dict, or None if no valid JSON could be extracted.

    Parsing strategy (in order):
    1. Strip markdown fences and whitespace.
    2. Return None for empty / null / trivially empty responses.
    3. Try json.loads() on the full text.
    4. Fall back to raw_decode() at the first '{' — log at DEBUG.
    5. Fall back to brace-matching (first '{' to last '}') — log at WARNING.
    6. Return None on total failure.
    """
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Empty / trivial
    if not text or text in ("{}", "[]", "null"):
        return None

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: raw_decode from first '{'
    start = text.find("{")
    if start != -1:
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text, start)
            log.debug("Parsed JSON via raw_decode (response had extra data)")
            return obj
        except json.JSONDecodeError:
            pass

        # Strategy 3: brace-matching (first '{' to last '}')
        end = text.rfind("}")
        if end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                log.warning("Extracted JSON using brace matching")
                return obj
            except json.JSONDecodeError:
                pass

    log.error("Failed to parse JSON from Claude response. "
              "First 500 chars: %s", text[:500])
    return None


def get_metric_mappings(
    client: anthropic.Anthropic,
    proforma_data: str,
    memo_content: str,
    cfg: dict,
    property_name: str = "",
) -> dict:
    """
    Send proforma data + memo content to Claude and receive structured
    JSON describing every metric update.

    This is the core reasoning step — Claude analyzes which memo values
    correspond to which proforma cells and determines the correct
    replacements, preserving formatting conventions.
    """
    model = cfg["claude"]["model"]
    max_tokens = cfg["claude"]["max_tokens"]
    temperature = cfg["claude"]["temperature"]
    use_thinking = "opus" in model.lower()

    if property_name:
        pn_section = (
            f"\n## Property Name — CRITICAL TARGETING OVERRIDE\n"
            f"The proforma data corresponds to the property named **\"{property_name}\"** "
            f"in the memo. The proforma's own internal name may differ (e.g. an old or "
            f"rebranded name) — IGNORE the proforma's internal project name for targeting "
            f"purposes.\n\n"
            f"Apply all proforma data to the column, row, or section labeled "
            f"\"{property_name}\" in the memo:\n"
            f"- In **side-by-side comparison tables** (column-oriented), update the "
            f"  \"{property_name}\" column, NOT any other property's column.\n"
            f"- In **row-oriented tables** (comp summary, pipeline), update the row "
            f"  for \"{property_name}\".\n"
            f"- In **narrative text**, update metrics that describe \"{property_name}\".\n"
            f"- Do NOT rename \"{property_name}\" to the proforma's internal name. "
            f"  Keep the memo's name as-is.\n"
            f"- Do NOT update columns/rows for other properties (those belong to "
            f"  different proformas).\n"
        )
    else:
        pn_section = ""

    prompt = MAPPING_PROMPT.format(
        proforma_data=proforma_data,
        memo_content=memo_content,
        property_name_section=pn_section,
    )

    log.info("Calling Claude API (model=%s, thinking=%s, prompt=%d chars)...",
             model, use_thinking, len(prompt))

    api_kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if use_thinking:
        api_kwargs["thinking"] = {"type": "adaptive"}
    else:
        api_kwargs["temperature"] = temperature

    message = client.messages.create(**api_kwargs)

    # Extract text from response (skip thinking blocks)
    raw = ""
    for block in message.content:
        if block.type == "text":
            raw = block.text
            break
    log.info("Claude response received (%d chars, %s stop_reason)",
             len(raw), message.stop_reason)

    if message.stop_reason == "max_tokens":
        log.warning(
            "Claude's response was cut off (hit max_tokens=%d). "
            "Attempting to salvage partial entries...",
            max_tokens,
        )
        salvaged = _salvage_truncated_json(raw)
        if salvaged is not None:
            n = len(salvaged["table_updates"]) + len(salvaged["text_updates"]) + len(salvaged["row_inserts"])
            log.info("Salvaged %d updates from truncated response", n)
            salvaged["_truncated"] = True
            return salvaged
        else:
            log.warning("Could not salvage truncated response — "
                        "caller will retry with smaller chunks")
            return {"table_updates": [], "text_updates": [], "row_inserts": [], "_truncated": True}

    # Parse JSON using consolidated helper
    empty_mappings = {"table_updates": [], "text_updates": [], "row_inserts": []}
    mappings = _parse_json_response(raw)
    if mappings is None:
        # Retry once — Claude sometimes returns analysis text instead of JSON
        log.warning("Claude returned non-JSON response — retrying with stricter prompt...")
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: You MUST respond with ONLY the JSON object. "
            "Do NOT include any analysis, explanation, or reasoning. "
            "Start your response with { and end with }."
        )
        api_kwargs["messages"] = [{"role": "user", "content": retry_prompt}]
        retry_msg = client.messages.create(**api_kwargs)
        retry_raw = ""
        for block in retry_msg.content:
            if block.type == "text":
                retry_raw = block.text
                break
        log.info("Retry response received (%d chars, %s stop_reason)",
                 len(retry_raw), retry_msg.stop_reason)
        mappings = _parse_json_response(retry_raw)
        if mappings is None:
            log.info("Retry also returned unparseable response — no updates for this batch")
            return empty_mappings

    # Ensure expected keys exist
    mappings.setdefault("table_updates", [])
    mappings.setdefault("text_updates", [])
    mappings.setdefault("row_inserts", [])

    n_table = len(mappings["table_updates"])
    n_text = len(mappings["text_updates"])
    n_row_ins = len(mappings["row_inserts"])
    log.info("Parsed mappings: %d table updates, %d text updates, %d row inserts",
             n_table, n_text, n_row_ins)
    return mappings


# ============================================================================
# 7. CLAUDE API — VALIDATION PASS
# ============================================================================
VALIDATION_PROMPT = """\
You are a QA reviewer for financial document updates. You have been given a
list of proposed changes (numbered by index) to an IC memo. Your job is to
find ONLY the problems — do NOT echo back entries that are correct.

## Proposed Changes (JSON, each entry has an "idx" field)
{mappings_json}

## Original Memo Content
{memo_content}

## Proforma Data
{proforma_data}

## Instructions

Review each proposed update and check:
1. Does the old_value / old_text EXACTLY match text in the memo? (character-for-character)
2. Does the new_value / new_text correctly reflect the proforma data?
3. Is the formatting consistent (dollar signs, commas, decimals, percent signs)?
4. Are there any DUPLICATE updates (same cell targeted twice)?
5. Are there any metrics in the memo that were MISSED?

IMPORTANT: Only return entries that FAIL validation or are MISSED.
Entries not mentioned are assumed to pass. This keeps the response compact.

Return ONLY valid JSON (no markdown fences) matching this schema:

{{
  "rejected": [
    {{
      "idx": <int, the index of the rejected entry>,
      "type": "table" or "text",
      "reason": "<why it was rejected>"
    }}
  ],
  "corrections": [
    {{
      "idx": <int, the index of the entry to correct>,
      "type": "table" or "text",
      "corrected_entry": {{ ... the full corrected entry ... }},
      "reason": "<what was wrong and how it was fixed>"
    }}
  ],
  "missed": [
    {{
      "page": <int>,
      "description": "<what metric was missed and its correct value>",
      "source": "<proforma cell>"
    }}
  ]
}}

If everything passes validation, return: {{"rejected": [], "corrections": [], "missed": []}}

Be strict: reject any update where old_value does not exactly match the memo,
or where the new_value formatting is inconsistent with the memo's style.

Additional domain rules to enforce:
- IRR values should use the **3-year holding-period Levered IRR**, not the
  1-year or 4-year IRR, unless the cell explicitly labels a different horizon.
  The 1-year IRR is typically >30%; if a proposed IRR new_value exceeds 30%
  for a "3 YR" label, **correct** it to the 3-year IRR (usually 20–28%).
  Do NOT just reject — emit a correction with the right value.
- "Chunk rents", unqualified "rents per month", "controllable OpEx per bed",
  and "OpEx ratio" all refer to **untrended** proforma values unless
  explicitly labeled "trended." Correct any that used trended values.
- The **top row** of comp summary/rollup and pipeline tables is the subject
  property — it MUST be updated. Flag as missed if it was skipped.
- End-of-memo data tables (underwriting projections pages with tables but
  no narrative) MUST be updated. Flag as missed if skipped.
- **Sensitivity analysis tables** must NEVER be updated. Reject any update
  targeting a table whose heading contains "Sensitivity."
- **Strategic/aspirational language** ("targeting", "driving towards", "our
  goal is", "we are aiming for") must NOT be overwritten with proforma
  values. Reject any text_update replacing text within such phrases.
- **Dev budget "% of Total"** must be recalculated when dollar amounts
  change. Flag as missed if a dollar amount changed but the corresponding
  percentage was not updated.
- For **row_inserts** in side-by-side comp tables, verify ONLY the subject
  property column is populated — all other columns must be empty strings.
  Reject any row_insert that fills comparable property columns.
- **Schedule dates** must match the schedule data exactly. If a date was updated
  from the schedule, verify the correct task was referenced. Date formatting must
  match the memo's existing style (Q2 2027 vs April 2027 vs 4/5/2027). Flag as
  missed if the memo references timeline milestones that were not updated.
{property_name_section}"""


def _call_validation_api(
    client: anthropic.Anthropic,
    indexed_mappings: dict,
    proforma_data: str,
    memo_content: str,
    cfg: dict,
    property_name: str = "",
) -> dict:
    """
    Single validation API call. Returns the parsed JSON result from Claude.
    Extracted as a helper so validate_mappings can batch multiple calls.
    Uses validation_model (defaults to same as mapping model if not set).
    """
    model = cfg["claude"].get("validation_model", cfg["claude"]["model"])
    max_tokens = cfg["claude"]["max_tokens"]
    temperature = cfg["claude"]["temperature"]
    use_thinking = "opus" in model.lower()

    if property_name:
        pn_section = (
            f"\n## Property Name — CRITICAL TARGETING CHECK\n"
            f"The proforma data corresponds to **\"{property_name}\"** in the memo. "
            f"The proforma's own internal name may differ (old/rebranded name).\n\n"
            f"Verify that:\n"
            f"- All updates target the \"{property_name}\" column/row, NOT other "
            f"  properties' columns/rows.\n"
            f"- The name \"{property_name}\" is NOT renamed to the proforma's "
            f"  internal name. Flag any mapping that renames it as REJECT.\n"
            f"- If any mapping targets a different property's column/row, REJECT it.\n"
        )
    else:
        pn_section = ""

    prompt = VALIDATION_PROMPT.format(
        mappings_json=json.dumps(indexed_mappings, indent=2),
        memo_content=memo_content,
        proforma_data=proforma_data,
        property_name_section=pn_section,
    )

    log.info("Calling Claude API for validation (model=%s, thinking=%s, "
             "prompt=%d chars)...", model, use_thinking, len(prompt))

    api_kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if use_thinking:
        api_kwargs["thinking"] = {"type": "adaptive"}
    else:
        api_kwargs["temperature"] = temperature

    message = client.messages.create(**api_kwargs)

    # Extract text from response (skip thinking blocks)
    raw = ""
    for block in message.content:
        if block.type == "text":
            raw = block.text
            break
    log.info("Validation response received (%d chars, %s stop_reason)",
             len(raw), message.stop_reason)

    if message.stop_reason == "max_tokens":
        log.warning(
            "Claude's validation response was cut off (hit max_tokens=%d). "
            "This batch will pass through unvalidated.",
            max_tokens,
        )
        return {"rejected": [], "corrections": [], "missed": []}

    # Parse JSON using consolidated helper
    empty_result = {"rejected": [], "corrections": [], "missed": []}
    result = _parse_json_response(raw)
    if result is None:
        log.info("Validation returned empty/unparseable response — all entries pass")
        return empty_result

    return result


def validate_mappings(
    client: anthropic.Anthropic,
    mappings: dict,
    proforma_data: str,
    memo_content: str,
    cfg: dict,
    property_name: str = "",
) -> dict:
    """
    Second Claude API call — validates the proposed mappings by cross-checking
    old values against the memo and new values against the proforma.

    The prompt asks Claude to return ONLY rejections, corrections, and missed
    entries (not all valid entries), keeping the response compact and well
    within token limits. Valid entries are inferred by exclusion.

    For large decks, batches the validation by page groups (same threshold
    as the mapping step) so the prompt stays within model limits.

    Returns a validated/cleaned version of the mappings with rejected entries
    removed and any missed metrics flagged.
    """
    BATCH_THRESHOLD = 80_000  # chars; same as mapping step
    RATE_LIMIT_INTERVAL = 65  # seconds between API calls

    # Add idx to each entry so Claude can reference them by index
    table_updates = mappings.get("table_updates", [])
    text_updates = mappings.get("text_updates", [])
    row_inserts = mappings.get("row_inserts", [])
    indexed_mappings = {
        "table_updates": [
            {**entry, "idx": i} for i, entry in enumerate(table_updates)
        ],
        "text_updates": [
            {**entry, "idx": i} for i, entry in enumerate(text_updates)
        ],
        "row_inserts": [
            {**entry, "idx": i} for i, entry in enumerate(row_inserts)
        ],
    }

    prompt_size = (len(proforma_data) + len(memo_content)
                   + len(json.dumps(indexed_mappings)))

    if prompt_size > BATCH_THRESHOLD:
        # Batch by page groups — split memo into chunks and only send
        # the mappings relevant to each chunk's pages.
        log.info("Large validation prompt (%d chars) — batching by page groups",
                 prompt_size)
        memo_chunks = chunk_memo_by_pages(memo_content, pages_per_chunk=5)

        # Determine which pages each chunk covers
        merged_result = {"rejected": [], "corrections": [], "missed": []}
        last_api_call = 0
        for ci, chunk in enumerate(memo_chunks, 1):
            # Extract page numbers from this chunk
            chunk_pages = set(
                int(m) for m in re.findall(r"PAGE (\d+)", chunk)
            )

            # Filter mappings to only entries for pages in this chunk
            chunk_indexed = {
                "table_updates": [
                    e for e in indexed_mappings["table_updates"]
                    if e.get("page") in chunk_pages
                ],
                "text_updates": [
                    e for e in indexed_mappings["text_updates"]
                    if e.get("page") in chunk_pages
                ],
                "row_inserts": [
                    e for e in indexed_mappings["row_inserts"]
                    if e.get("page") in chunk_pages
                ],
            }

            n_entries = (len(chunk_indexed["table_updates"])
                         + len(chunk_indexed["text_updates"])
                         + len(chunk_indexed["row_inserts"]))
            if n_entries == 0:
                log.info("Validation batch %d/%d: no mappings for pages %s — skipping",
                         ci, len(memo_chunks), sorted(chunk_pages))
                continue

            if ci > 1 and last_api_call > 0:
                elapsed = time.time() - last_api_call
                wait = RATE_LIMIT_INTERVAL - elapsed
                if wait > 0:
                    log.info("Rate limit: waiting %.0f seconds...", wait)
                    time.sleep(wait)

            log.info("Validation batch %d/%d (%d entries, pages %s)...",
                     ci, len(memo_chunks), n_entries, sorted(chunk_pages))
            last_api_call = time.time()
            batch_result = _call_validation_api(
                client, chunk_indexed, proforma_data, chunk, cfg,
                property_name=property_name,
            )
            merged_result["rejected"].extend(batch_result.get("rejected", []))
            merged_result["corrections"].extend(batch_result.get("corrections", []))
            merged_result["missed"].extend(batch_result.get("missed", []))

        result = merged_result
    else:
        result = _call_validation_api(
            client, indexed_mappings, proforma_data, memo_content, cfg,
            property_name=property_name,
        )

    # Reconstruct validated mappings: start with originals, remove rejections,
    # apply corrections
    rejected_table_idxs = set()
    rejected_text_idxs = set()
    rejected_row_insert_idxs = set()
    correction_table = {}
    correction_text = {}
    correction_row_insert = {}

    for rej in result.get("rejected", []):
        idx = rej.get("idx")
        if idx is not None:
            if rej.get("type") == "text":
                rejected_text_idxs.add(idx)
            elif rej.get("type") == "row_insert":
                rejected_row_insert_idxs.add(idx)
            else:
                rejected_table_idxs.add(idx)

    for cor in result.get("corrections", []):
        idx = cor.get("idx")
        if idx is not None:
            if cor.get("type") == "text":
                correction_text[idx] = cor["corrected_entry"]
            elif cor.get("type") == "row_insert":
                correction_row_insert[idx] = cor["corrected_entry"]
            else:
                correction_table[idx] = cor["corrected_entry"]

    valid_table = []
    for i, entry in enumerate(table_updates):
        if i in rejected_table_idxs:
            continue
        if i in correction_table:
            valid_table.append(correction_table[i])
        else:
            valid_table.append(entry)

    valid_text = []
    for i, entry in enumerate(text_updates):
        if i in rejected_text_idxs:
            continue
        if i in correction_text:
            valid_text.append(correction_text[i])
        else:
            valid_text.append(entry)

    valid_row_inserts = []
    for i, entry in enumerate(row_inserts):
        if i in rejected_row_insert_idxs:
            continue
        if i in correction_row_insert:
            valid_row_inserts.append(correction_row_insert[i])
        else:
            valid_row_inserts.append(entry)

    n_rejected = len(result.get("rejected", []))
    n_corrections = len(result.get("corrections", []))
    n_missed = len(result.get("missed", []))
    log.info("Validation: %d passed, %d rejected, %d corrected, %d missed",
             len(valid_table) + len(valid_text) + len(valid_row_inserts),
             n_rejected, n_corrections, n_missed)

    if n_rejected > 0:
        for rej in result["rejected"]:
            log.warning("  REJECTED idx=%s: %s", rej.get("idx", "?"),
                        rej.get("reason", "unknown"))
    if n_corrections > 0:
        for cor in result["corrections"]:
            log.warning("  CORRECTED idx=%s: %s", cor.get("idx", "?"),
                        cor.get("reason", "unknown"))
    if n_missed > 0:
        for miss in result["missed"]:
            log.warning("  MISSED: page %s — %s", miss.get("page", "?"),
                        miss.get("description", ""))

    # Build rejected list for change log (include full original entries)
    rejected_entries = []
    for rej in result.get("rejected", []):
        idx = rej.get("idx")
        entry_type = rej.get("type", "table")
        original = {}
        if entry_type == "text" and idx is not None and idx < len(text_updates):
            original = text_updates[idx]
        elif entry_type == "row_insert" and idx is not None and idx < len(row_inserts):
            original = row_inserts[idx]
        elif idx is not None and idx < len(table_updates):
            original = table_updates[idx]
        rejected_entries.append({
            "original": original,
            "reason": rej.get("reason", "unknown"),
        })

    return {
        "table_updates": valid_table,
        "text_updates": valid_text,
        "row_inserts": valid_row_inserts,
        "rejected": rejected_entries,
        "missed": result.get("missed", []),
    }


# ============================================================================
# 8. PRE-VALIDATION (local Python check)
# ============================================================================
def pre_validate_mappings(mappings: dict, memo_content: str) -> dict:
    """
    Quick local check: verify each old_value / old_text actually exists in the
    memo content. Reject entries that don't match. This catches the most common
    Claude errors instantly without an API call.

    Returns a new mappings dict with non-matching entries moved to 'rejected'.
    """
    def _split_memo_by_page(content: str) -> dict:
        """
        Build a {page_number: page_text_block} map from extracted memo content.
        """
        blocks = {}
        page_headers = list(re.finditer(
            r"={60,}\nPAGE\s+(\d+)[^\n]*\n={60,}",
            content,
        ))
        if not page_headers:
            return blocks

        for i, m in enumerate(page_headers):
            page_num = int(m.group(1))
            start = m.start()
            end = page_headers[i + 1].start() if i + 1 < len(page_headers) else len(content)
            blocks[page_num] = content[start:end]
        return blocks

    page_blocks = _split_memo_by_page(memo_content)
    valid_table = []
    valid_text = []
    rejected = list(mappings.get("rejected", []))

    for upd in mappings.get("table_updates", []):
        page = upd.get("page")
        haystack = page_blocks.get(page, memo_content)
        old_value = upd.get("old_value", "")
        if old_value and old_value in haystack:
            valid_table.append(upd)
        else:
            rejected.append({
                "original": upd,
                "reason": (
                    f"old_value not found on page {page}: '{old_value}'"
                    if page in page_blocks
                    else f"old_value not found in memo: '{old_value}'"
                ),
            })

    for upd in mappings.get("text_updates", []):
        page = upd.get("page")
        haystack = page_blocks.get(page, memo_content)
        old_text = upd.get("old_text", "")
        if old_text and old_text in haystack:
            valid_text.append(upd)
        else:
            rejected.append({
                "original": upd,
                "reason": (
                    f"old_text not found on page {page}: '{old_text}'"
                    if page in page_blocks
                    else f"old_text not found in memo: '{old_text}'"
                ),
            })

    n_rejected_new = (len(rejected) - len(mappings.get("rejected", [])))
    if n_rejected_new > 0:
        log.warning("Pre-validation rejected %d entries (old value not in memo)",
                    n_rejected_new)

    return {
        "table_updates": valid_table,
        "text_updates": valid_text,
        "row_inserts": mappings.get("row_inserts", []),
        "rejected": rejected,
        "missed": mappings.get("missed", []),
    }


# ============================================================================
# 9. APPLY TEXT / TABLE UPDATES  (python-pptx)
# ============================================================================
def _replace_in_para(para, old_text: str, new_text: str) -> bool:
    """
    Replace old_text with new_text in a paragraph, handling the common
    case where a value is split across multiple XML runs.

    Strategy:
    1. Try a direct single-run replacement first (fastest, preserves all formatting).
    2. Fall back to a full-paragraph merge: concatenate all run texts, do the
       replacement, write the result into the first run, and clear the rest.
       This loses per-run character formatting within the cell but preserves
       paragraph-level formatting (alignment, spacing), which is acceptable
       for financial table values that are typically uniform within a cell.
    """
    # Pass 1: single-run replacement
    for run in para.runs:
        if old_text in run.text:
            run.text = run.text.replace(old_text, new_text)
            return True

    # Pass 2: cross-run replacement
    full_text = "".join(r.text for r in para.runs)
    if old_text in full_text:
        new_full = full_text.replace(old_text, new_text)
        if para.runs:
            para.runs[0].text = new_full
            for run in para.runs[1:]:
                run.text = ""
        return True

    return False


def _replace_in_cell(cell, old_text: str, new_text: str) -> bool:
    """Replace old_text with new_text inside a table cell, preserving runs."""
    for para in cell.text_frame.paragraphs:
        if _replace_in_para(para, old_text, new_text):
            return True
    return False


def _normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters that PowerPoint often substitutes:
    non-breaking spaces -> spaces, smart quotes -> ASCII quotes,
    en/em dashes -> hyphens, etc.
    """
    return (text
            .replace("\u00a0", " ")    # non-breaking space
            .replace("\u2013", "-")    # en dash
            .replace("\u2014", "-")    # em dash
            .replace("\u2018", "'")    # left single quote
            .replace("\u2019", "'")    # right single quote
            .replace("\u201c", '"')    # left double quote
            .replace("\u201d", '"')    # right double quote
            .replace("\u2502", "|")    # box drawing vertical
            .replace("\uff5c", "|"))   # fullwidth vertical line


def _replace_in_shape(shape, old_text: str, new_text: str) -> bool:
    """Replace old_text in any text-frame shape, preserving formatting.
    Falls back to Unicode-normalized matching if exact match fails."""
    if not shape.has_text_frame:
        return False
    # Pass 1: exact match
    for para in shape.text_frame.paragraphs:
        if old_text not in para.text:
            continue
        if _replace_in_para(para, old_text, new_text):
            return True
    # Pass 2: normalized match — find the actual text in the shape that
    # corresponds to the normalized old_text, then replace that
    norm_old = _normalize_unicode(old_text)
    for para in shape.text_frame.paragraphs:
        para_text = para.text
        norm_para = _normalize_unicode(para_text)
        if norm_old not in norm_para:
            continue
        # Find the actual substring in the original para text
        idx = norm_para.find(norm_old)
        actual_old = para_text[idx:idx + len(norm_old)]
        if _replace_in_para(para, actual_old, new_text):
            return True
    return False


def _normalize_for_match(text: str) -> str:
    """Normalize text for robust table/row matching."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _strip_to_core(text: str) -> str:
    """
    Strip a label down to its alphanumeric core for bedroom-type matching.
    '1BR/1BA' -> '1br1ba', '1 BR' -> '1br', '4BR/2BA' -> '4br2ba'
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _loose_match(expected: str, actual: str) -> bool:
    """
    True when expected roughly matches actual (exact or containment after
    whitespace/case normalization), or when both start with the same
    bedroom count (e.g. '1BR/1BA' matches '1 BR').
    """
    e = _normalize_for_match(expected)
    a = _normalize_for_match(actual)
    if not e:
        return True
    if not a:
        return False
    if e == a or e in a or a in e:
        return True
    # Bedroom-label fallback: match on leading digit + "br"
    ec = _strip_to_core(expected)
    ac = _strip_to_core(actual)
    br_pat = re.match(r"^(\d+)br", ec)
    br_pat2 = re.match(r"^(\d+)br", ac)
    if br_pat and br_pat2 and br_pat.group(1) == br_pat2.group(1):
        return True
    return False


def _find_table_target(slide, table_name: str, row_label: str,
                       col_idx: int, old_value: str):
    """
    Find the best target cell for a table update.
    Preference order:
    1) Tables matching table_name + rows matching row_label + cell has old_value
    2) Any table + rows matching row_label + cell has old_value
    3) Any table + ANY row where cell at col_idx has old_value (fallback)
    4) Any table + ANY row + ANY column has old_value (last resort)

    Returns:
      (shape, row_label_actual, cell) on success, or
      (None, row_label_actual, cell_text) for diagnostics.
    """
    tables = [s for s in slide.shapes if s.has_table]
    if not tables:
        return None, None, None

    if table_name:
        preferred = [s for s in tables if _loose_match(table_name, s.name)]
        fallback = [s for s in tables if s not in preferred]
        ordered_groups = [preferred, fallback] if preferred else [tables]
    else:
        ordered_groups = [tables]

    diagnostic_row = None
    diagnostic_cell = None

    # Pass 1: row_label match + old_value at col_idx
    for group in ordered_groups:
        for shape in group:
            for row in shape.table.rows:
                if col_idx >= len(row.cells):
                    continue
                row_head = row.cells[0].text.strip() if row.cells else ""
                if row_label and not _loose_match(row_label, row_head):
                    continue

                cell = row.cells[col_idx]
                cell_text = cell.text or ""
                if old_value in cell_text:
                    return shape, row_head, cell

                if diagnostic_row is None:
                    diagnostic_row = row_head
                    diagnostic_cell = cell_text

    # Pass 2: ignore row_label, find old_value at col_idx in any row
    for group in ordered_groups:
        for shape in group:
            for row in shape.table.rows:
                if col_idx >= len(row.cells):
                    continue
                cell = row.cells[col_idx]
                cell_text = cell.text or ""
                if old_value in cell_text:
                    row_head = row.cells[0].text.strip() if row.cells else ""
                    log.debug("Fallback match (ignoring row_label '%s'): "
                              "found '%s' at row '%s' col %d",
                              row_label, old_value, row_head, col_idx)
                    return shape, row_head, cell

    # Pass 3: ignore row_label AND col_idx, find old_value in any cell
    for group in ordered_groups:
        for shape in group:
            for row in shape.table.rows:
                for ci, cell in enumerate(row.cells):
                    cell_text = cell.text or ""
                    if old_value in cell_text:
                        row_head = row.cells[0].text.strip() if row.cells else ""
                        log.debug("Last-resort match: found '%s' at row '%s' "
                                  "col %d (requested col %d)",
                                  old_value, row_head, ci, col_idx)
                        return shape, row_head, cell

    return None, diagnostic_row, diagnostic_cell


def _find_row_by_label(table, row_label: str) -> int | None:
    """
    Find a row index in a python-pptx table by matching column-0 text
    using _loose_match. Returns the 0-based row index, or None.
    """
    for idx, row in enumerate(table.rows):
        cell_text = row.cells[0].text.strip() if row.cells else ""
        if _loose_match(row_label, cell_text):
            return idx
    return None


def _add_table_row(table, reference_row_idx: int, cell_values: list):
    """
    Clone the XML of the row at *reference_row_idx*, clear all cell text,
    populate with *cell_values*, and insert the new row immediately after
    the reference row.

    Uses lxml operations (deepcopy / addnext) on the underlying DrawingML
    XML, which is the only reliable way to add rows to a python-pptx table.
    """
    ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    tbl_xml = table._tbl  # lxml element <a:tbl>

    # Get all <a:tr> elements
    rows = tbl_xml.findall(f"{{{ns}}}tr")
    if reference_row_idx < 0 or reference_row_idx >= len(rows):
        log.warning("_add_table_row: reference_row_idx %d out of range (table has %d rows)",
                    reference_row_idx, len(rows))
        return

    ref_tr = rows[reference_row_idx]
    new_tr = deepcopy(ref_tr)

    # Clear text in every cell of the cloned row and set new values
    new_cells = new_tr.findall(f"{{{ns}}}tc")
    for ci, tc in enumerate(new_cells):
        # Clear all text runs inside the cell
        for p in tc.findall(f".//{{{ns}}}p"):
            for r in p.findall(f"{{{ns}}}r"):
                t = r.find(f"{{{ns}}}t")
                if t is not None:
                    t.text = ""
        # Set new value in the first run of the first paragraph
        first_p = tc.find(f".//{{{ns}}}p")
        if first_p is not None:
            first_r = first_p.find(f"{{{ns}}}r")
            if first_r is not None:
                t = first_r.find(f"{{{ns}}}t")
                if t is not None:
                    t.text = cell_values[ci] if ci < len(cell_values) else ""

    ref_tr.addnext(new_tr)


def apply_updates(memo_path: str, mappings: dict, dry_run: bool = False) -> list:
    """
    Open the memo, apply every table_update and text_update from the
    Claude-validated mappings, and save. Returns a list of change records.
    """
    prs = Presentation(memo_path)
    changes = []

    # --- Table updates ---
    for upd in mappings.get("table_updates", []):
        page = upd["page"]
        tbl_name = upd.get("table_name", "")
        col_idx = upd.get("column_index", 1)
        old_val = upd["old_value"]
        new_val = upd["new_value"]
        source = upd.get("source", "")
        row_label = upd.get("row_label", "")

        try:
            slide = prs.slides[page - 1]
        except IndexError:
            log.warning("Table update SKIPPED: page %d does not exist", page)
            continue
        shape, matched_row_label, cell_or_text = _find_table_target(
            slide, tbl_name, row_label, col_idx, old_val
        )
        if shape is not None:
            if not dry_run:
                _replace_in_cell(cell_or_text, old_val, new_val)
            location_table = shape.name or tbl_name or "<unnamed table>"
            location_row = matched_row_label or row_label or "<unknown row>"
            changes.append({
                "page": page, "type": "table",
                "location": f"{location_table} / {location_row} / col {col_idx}",
                "old": old_val, "new": new_val, "source": source,
            })
            continue

        if matched_row_label is not None:
            log.warning(
                "Table update NOT FOUND: page %d, '%s' -> '%s' "
                "(closest row '%s' col %d has '%s')",
                page, old_val, new_val, matched_row_label, col_idx, cell_or_text
            )
        else:
            log.warning(
                "Table update NOT FOUND: page %d, '%s' -> '%s' "
                "(no matching table/row for table_name='%s' row_label='%s')",
                page, old_val, new_val, tbl_name, row_label
            )

    # --- Text (narrative) updates ---
    for upd in mappings.get("text_updates", []):
        page = upd["page"]
        old_txt = upd["old_text"]
        new_txt = upd["new_text"]
        source = upd.get("source", "")

        try:
            slide = prs.slides[page - 1]
        except IndexError:
            log.warning("Text update SKIPPED: page %d does not exist", page)
            continue
        found = False
        for shape in slide.shapes:
            if not dry_run:
                if _replace_in_shape(shape, old_txt, new_txt):
                    changes.append({
                        "page": page, "type": "text",
                        "location": shape.name,
                        "old": old_txt, "new": new_txt, "source": source,
                    })
                    found = True
                    break
            else:
                # Dry-run: check if the text exists without modifying
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if old_txt in para.text:
                            changes.append({
                                "page": page, "type": "text",
                                "location": shape.name,
                                "old": old_txt, "new": new_txt, "source": source,
                            })
                            found = True
                            break
                if found:
                    break

        if not found:
            log.warning("Text update NOT FOUND: page %d, '%s' -> '%s'",
                        page, old_txt, new_txt)

    # --- Row inserts ---
    for ins in mappings.get("row_inserts", []):
        page = ins["page"]
        tbl_name = ins.get("table_name", "")
        ref_label = ins.get("insert_after_row_label", "")
        cell_values = ins.get("cells", [])
        source = ins.get("source", "")

        try:
            slide = prs.slides[page - 1]
        except IndexError:
            log.warning("Row insert SKIPPED: page %d does not exist", page)
            continue

        # Find the target table
        tables = [s for s in slide.shapes if s.has_table]
        target_shape = None
        if tbl_name:
            for s in tables:
                if _loose_match(tbl_name, s.name):
                    target_shape = s
                    break
        # Fallback: find any table containing the reference row label
        if target_shape is None:
            for s in tables:
                if _find_row_by_label(s.table, ref_label) is not None:
                    target_shape = s
                    break

        if target_shape is None:
            log.warning("Row insert NOT FOUND: page %d, table '%s', "
                        "ref_label '%s'", page, tbl_name, ref_label)
            continue

        table = target_shape.table
        ref_idx = _find_row_by_label(table, ref_label)
        if ref_idx is None:
            log.warning("Row insert ref row NOT FOUND: page %d, label '%s'",
                        page, ref_label)
            continue

        # Pad or truncate cells to match column count
        n_cols = len(table.rows[0].cells) if table.rows else 0
        if len(cell_values) < n_cols:
            cell_values = cell_values + [""] * (n_cols - len(cell_values))
        elif len(cell_values) > n_cols:
            cell_values = cell_values[:n_cols]

        if not dry_run:
            _add_table_row(table, ref_idx, cell_values)

        location_table = target_shape.name or tbl_name or "<unnamed table>"
        changes.append({
            "page": page, "type": "row_insert",
            "location": f"{location_table} / after '{ref_label}'",
            "old": "(new row)", "new": " | ".join(cell_values),
            "source": source,
        })

    if not dry_run:
        prs.save(memo_path)
        log.info("Memo saved with %d updates.", len(changes))
    else:
        log.info("Dry-run: %d updates identified (not saved).", len(changes))

    return changes


# ============================================================================
# 10. CHANGE LOG
# ============================================================================
def write_change_log(output_dir: str, all_changes: list, mappings: dict,
                     memo_path: str, proforma_path: str, backup_path: str):
    """Write a Markdown change-log summarizing every modification."""
    def _md_cell(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(output_dir, "CHANGE_LOG.md")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Memo Automator - Change Log\n\n")
        f.write(f"**Run date:** {ts}\n\n")
        f.write(f"**Memo:** `{os.path.basename(memo_path)}`\n\n")
        f.write(f"**Proforma:** `{os.path.basename(proforma_path)}`\n\n")
        f.write(f"**Backup:** `{os.path.basename(backup_path)}`\n\n")
        f.write(f"**Total changes applied:** {len(all_changes)}\n\n")

        f.write("## Applied Changes\n\n")
        f.write("| # | Page | Type | Location | Old | New | Source |\n")
        f.write("|---|------|------|----------|-----|-----|--------|\n")
        for i, c in enumerate(all_changes, 1):
            old_display = c['old'][:40] + "..." if len(c['old']) > 40 else c['old']
            new_display = c['new'][:40] + "..." if len(c['new']) > 40 else c['new']
            f.write(f"| {i} | {c['page']} | {c['type']} | "
                    f"{_md_cell(c['location'])} | {_md_cell(old_display)} | "
                    f"{_md_cell(new_display)} | {_md_cell(c['source'])} |\n")

        # Rejected updates
        rejected = mappings.get("rejected", [])
        if rejected:
            f.write("\n\n## Rejected Updates\n\n")
            f.write("These were proposed but failed validation:\n\n")
            for rej in rejected:
                f.write(f"- **Reason:** {rej.get('reason', 'unknown')}\n")
                orig = rej.get("original", {})
                f.write(f"  - Entry: `{json.dumps(orig)}`\n")

        # Missed metrics
        missed = mappings.get("missed", [])
        if missed:
            f.write("\n\n## Potentially Missed Metrics\n\n")
            f.write("These may need manual review:\n\n")
            for miss in missed:
                f.write(f"- **Page {miss.get('page', '?')}:** "
                        f"{miss.get('description', '')} "
                        f"(source: {miss.get('source', 'unknown')})\n")

        # Raw Claude mappings for auditability
        f.write("\n\n## Raw Claude API Mappings\n\n")
        f.write("```json\n")
        f.write(json.dumps(mappings, indent=2))
        f.write("\n```\n")

    log.info("Change log written: %s", log_path)
    return log_path


# ============================================================================
# 11. MAIN
# ============================================================================
def main():
    args = parse_args()

    # Load environment (.env file in script dir or working directory)
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    load_dotenv()  # also check cwd

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set. Copy .env.example to .env and "
                  "add your key.")
        sys.exit(1)

    # Validate inputs
    if not os.path.isfile(args.memo):
        log.error("Memo file not found: %s", args.memo)
        sys.exit(1)
    if not os.path.isfile(args.proforma):
        log.error("Proforma file not found: %s", args.proforma)
        sys.exit(1)

    # Load config
    cfg = load_config(args.config)
    log.info("Config loaded from %s", args.config)

    # Output directory
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.memo))
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Claude client (shared for mapping + validation)
    log.info("Mapping model: %s  |  Validation model: %s",
             cfg["claude"]["model"], cfg["claude"]["validation_model"])
    client = anthropic.Anthropic(
        api_key=api_key,
        max_retries=5,
        timeout=900.0,  # 15 min — needed for large batches and Opus thinking
    )

    # ---- Step 1: Backup ----
    log.info("=" * 60)
    log.info("STEP 1: Creating backup")
    log.info("=" * 60)
    backup_path = create_backup(args.memo, output_dir)

    # ---- Step 2: Extract proforma data ----
    log.info("=" * 60)
    log.info("STEP 2: Extracting proforma data")
    log.info("=" * 60)
    proforma_data = extract_proforma_data(args.proforma, cfg)

    # Save extraction for debugging / audit
    pf_dump = os.path.join(output_dir, "proforma_extract.txt")
    with open(pf_dump, "w", encoding="utf-8") as f:
        f.write(proforma_data)
    log.info("Proforma data saved to %s", pf_dump)

    # ---- Step 3: Extract memo content (full deck) ----
    log.info("=" * 60)
    log.info("STEP 3: Extracting memo content (all slides)")
    log.info("=" * 60)
    memo_content = extract_memo_content(args.memo, cfg)

    memo_dump = os.path.join(output_dir, "memo_extract.txt")
    with open(memo_dump, "w", encoding="utf-8") as f:
        f.write(memo_content)
    log.info("Memo content saved to %s", memo_dump)

    # ---- Step 4: Claude API — metric mapping (reasoning step) ----
    # This API call replaces human analysis: Claude reads the proforma and
    # memo data, identifies which metrics map to which cells, and determines
    # the exact old->new text replacements with proper formatting.
    # For large decks the memo is split into batches of 10 slides to stay
    # within the model's output token limit; results are merged afterward.
    log.info("=" * 60)
    log.info("STEP 4: Claude API — identifying metric mappings")
    log.info("=" * 60)
    BATCH_THRESHOLD = 80_000  # chars; above this, process slides in batches
    RATE_LIMIT_INTERVAL = 65  # seconds between API calls for rate limiting
    prompt_size = len(proforma_data) + len(memo_content)
    if prompt_size > BATCH_THRESHOLD:
        log.info("Large prompt (%d chars) — processing slides in batches of 3",
                 prompt_size)
        memo_chunks = chunk_memo_by_pages(memo_content, pages_per_chunk=3)
        mappings = {"table_updates": [], "text_updates": [], "row_inserts": []}
        last_api_call = 0
        for i, chunk in enumerate(memo_chunks, 1):
            if i > 1 and last_api_call > 0:
                elapsed = time.time() - last_api_call
                wait = RATE_LIMIT_INTERVAL - elapsed
                if wait > 0:
                    log.info("Rate limit: waiting %.0f seconds (%.0fs elapsed since last call)...",
                             wait, elapsed)
                    time.sleep(wait)
                else:
                    log.info("Rate limit: no wait needed (%.0fs elapsed since last call)",
                             elapsed)
            log.info("Mapping batch %d / %d ...", i, len(memo_chunks))
            last_api_call = time.time()
            try:
                batch = get_metric_mappings(client, proforma_data, chunk, cfg,
                                           property_name=args.property_name)
            except Exception as batch_err:
                log.warning("Batch %d failed (%s) — retrying as single-page sub-chunks",
                            i, batch_err)
                batch = {"table_updates": [], "text_updates": [],
                         "row_inserts": [], "_truncated": True}

            # --- Retry truncated/failed batches with single-page sub-chunks ---
            if batch.pop("_truncated", False):
                # Collect pages already covered by salvaged entries
                covered_pages = set()
                for e in batch.get("table_updates", []):
                    covered_pages.add(e.get("page"))
                for e in batch.get("text_updates", []):
                    covered_pages.add(e.get("page"))
                for e in batch.get("row_inserts", []):
                    covered_pages.add(e.get("page"))
                # Keep whatever was salvaged
                mappings["table_updates"].extend(batch.get("table_updates", []))
                mappings["text_updates"].extend(batch.get("text_updates", []))
                mappings["row_inserts"].extend(batch.get("row_inserts", []))

                log.info("Retrying truncated batch %d with single-page sub-chunks "
                         "(covered pages so far: %s)", i, sorted(covered_pages))
                sub_chunks = chunk_memo_by_pages(chunk, pages_per_chunk=1)
                for j, sub_chunk in enumerate(sub_chunks, 1):
                    sub_pages = set(
                        int(m) for m in re.findall(r"PAGE (\d+)", sub_chunk)
                    )
                    if sub_pages and sub_pages.issubset(covered_pages):
                        log.info("  Sub-chunk %d/%d (pages %s) already covered — skipping",
                                 j, len(sub_chunks), sorted(sub_pages))
                        continue

                    elapsed = time.time() - last_api_call
                    wait = RATE_LIMIT_INTERVAL - elapsed
                    if wait > 0:
                        log.info("Rate limit: waiting %.0f seconds...", wait)
                        time.sleep(wait)

                    log.info("  Sub-chunk %d/%d (pages %s)...",
                             j, len(sub_chunks), sorted(sub_pages))
                    last_api_call = time.time()
                    try:
                        sub_batch = get_metric_mappings(
                            client, proforma_data, sub_chunk, cfg,
                            property_name=args.property_name,
                        )
                    except Exception as sub_err:
                        log.warning("  Sub-chunk %d failed (%s) — skipping pages %s",
                                    j, sub_err, sorted(sub_pages))
                        continue
                    if sub_batch.pop("_truncated", False):
                        log.warning("  Sub-chunk %d still truncated after single-page "
                                    "retry — moving on", j)
                    mappings["table_updates"].extend(
                        sub_batch.get("table_updates", []))
                    mappings["text_updates"].extend(
                        sub_batch.get("text_updates", []))
                    mappings["row_inserts"].extend(
                        sub_batch.get("row_inserts", []))
            else:
                mappings["table_updates"].extend(batch.get("table_updates", []))
                mappings["text_updates"].extend(batch.get("text_updates", []))
                mappings["row_inserts"].extend(batch.get("row_inserts", []))
    else:
        mappings = get_metric_mappings(client, proforma_data, memo_content, cfg,
                                       property_name=args.property_name)
        mappings.pop("_truncated", None)

    # Save raw mappings for audit
    map_dump = os.path.join(output_dir, "mappings_raw.json")
    with open(map_dump, "w") as f:
        json.dump(mappings, f, indent=2)
    log.info("Raw mappings saved to %s", map_dump)

    # ---- Step 4a: Strip no-op entries (old == new) ----
    pre_table = len(mappings["table_updates"])
    pre_text = len(mappings["text_updates"])
    mappings["table_updates"] = [
        e for e in mappings["table_updates"]
        if e.get("old_value") != e.get("new_value")
    ]
    mappings["text_updates"] = [
        e for e in mappings["text_updates"]
        if e.get("old_text") != e.get("new_text")
    ]
    n_stripped = (pre_table - len(mappings["table_updates"])
                  + pre_text - len(mappings["text_updates"]))
    if n_stripped > 0:
        log.info("Stripped %d no-op entries (old == new)", n_stripped)

    # ---- Step 4b: Pre-validation (local Python check) ----
    mappings = pre_validate_mappings(mappings, memo_content)

    # ---- Step 5: Claude API — validation pass (QA reasoning step) ----
    # Second API call: Claude cross-checks the proposed updates to catch
    # errors — wrong old_value text, formatting mismatches, duplicates,
    # or missed metrics. This replaces human review of the mapping output.
    if args.skip_validation:
        log.info("=" * 60)
        log.info("STEP 5: SKIPPED (--skip-validation flag)")
        log.info("=" * 60)
        validated = mappings
        validated.setdefault("rejected", [])
        validated.setdefault("missed", [])
    else:
        log.info("=" * 60)
        log.info("STEP 5: Claude API — validating mappings")
        log.info("=" * 60)
        validated = validate_mappings(client, mappings, proforma_data, memo_content, cfg,
                                      property_name=args.property_name)

    # Save validated mappings for audit
    val_dump = os.path.join(output_dir, "mappings_validated.json")
    with open(val_dump, "w") as f:
        json.dump(validated, f, indent=2)
    log.info("Validated mappings saved to %s", val_dump)

    # ---- Step 6: Apply text / table updates ----
    log.info("=" * 60)
    log.info("STEP 6: Applying text / table updates")
    log.info("=" * 60)
    changes = apply_updates(args.memo, validated, dry_run=args.dry_run)

    # ---- Step 7: Change log ----
    log.info("=" * 60)
    log.info("STEP 7: Writing change log")
    log.info("=" * 60)
    log_path = write_change_log(
        output_dir, changes, validated,
        args.memo, args.proforma, backup_path,
    )

    # ---- Summary ----
    n_rejected = len(validated.get("rejected", []))
    n_missed = len(validated.get("missed", []))
    print("\n" + "=" * 60)
    print("MEMO AUTOMATOR COMPLETE")
    print("=" * 60)
    print(f"  Changes applied:     {len(changes)}")
    print(f"  Rejected by QA:      {n_rejected}")
    print(f"  Potentially missed:  {n_missed}")
    print(f"  Backup:              {backup_path}")
    print(f"  Change log:          {log_path}")
    if args.dry_run:
        print("  ** DRY RUN -- no files were modified **")
    print("=" * 60)


if __name__ == "__main__":
    main()
