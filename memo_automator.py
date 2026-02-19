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
        "Assumptions", "Comparison"
    ])
    cfg["proforma"].setdefault("max_rows_per_tab", 250)
    cfg["proforma"].setdefault("max_cols_per_tab", 30)
    cfg.setdefault("memo", {})
    cfg["memo"].setdefault("pages", "all")
    cfg.setdefault("claude", {})
    cfg["claude"].setdefault("model", "claude-sonnet-4-20250514")
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
                    row_data.append(f"{cell.coordinate}: {cell.value}")
            if row_data:
                lines.append(" | ".join(row_data))

    wb.close()
    proforma_text = "\n".join(lines)
    log.info("Proforma extraction complete (%d lines, %d chars)",
             len(lines), len(proforma_text))
    return proforma_text


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

Return ONLY valid JSON (no markdown fences, no commentary) matching this
schema exactly:

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
  ]
}}

Important:
- old_value / old_text must be the EXACT text currently in the memo.
- new_value / new_text must match the memo's formatting style.
- For dollar amounts use commas: $68,769,750 not $68769750.
- For percentages keep the same decimal places as the memo uses.
- Only include metrics that clearly map to the proforma. Do NOT guess.
- Scan ALL pages including large data tables (e.g. executive summary,
  cash flow, unit mix, development budget tables that may span full pages).
"""


def get_metric_mappings(
    client: anthropic.Anthropic,
    proforma_data: str,
    memo_content: str,
    cfg: dict,
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

    prompt = MAPPING_PROMPT.format(
        proforma_data=proforma_data,
        memo_content=memo_content,
    )

    log.info("Calling Claude API (model=%s, prompt=%d chars)...",
             model, len(prompt))

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    log.info("Claude response received (%d chars, %s stop_reason)",
             len(raw), message.stop_reason)

    if message.stop_reason == "max_tokens":
        log.error(
            "Claude's response was cut off (hit max_tokens=%d). "
            "Increase max_tokens in config.yaml and retry.",
            max_tokens,
        )
        sys.exit(1)

    # Parse JSON — handle optional markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    mappings = json.loads(text)
    n_table = len(mappings.get("table_updates", []))
    n_text = len(mappings.get("text_updates", []))
    log.info("Parsed mappings: %d table updates, %d text updates", n_table, n_text)
    return mappings


# ============================================================================
# 7. CLAUDE API — VALIDATION PASS
# ============================================================================
VALIDATION_PROMPT = """\
You are a QA reviewer for financial document updates. You have been given a
list of proposed changes to an IC memo. Your job is to validate them.

## Proposed Changes (JSON)
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

Return ONLY valid JSON (no markdown fences) matching this schema:

{{
  "validated_table_updates": [
    ... same schema as table_updates, only include entries that pass validation ...
  ],
  "validated_text_updates": [
    ... same schema as text_updates, only include entries that pass validation ...
  ],
  "rejected": [
    {{
      "original": {{ ... the rejected entry ... }},
      "reason": "<why it was rejected>"
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

Be strict: reject any update where old_value does not exactly match the memo,
or where the new_value formatting is inconsistent with the memo's style.
"""


def validate_mappings(
    client: anthropic.Anthropic,
    mappings: dict,
    proforma_data: str,
    memo_content: str,
    cfg: dict,
) -> dict:
    """
    Second Claude API call — validates the proposed mappings by cross-checking
    old values against the memo and new values against the proforma.

    Returns a validated/cleaned version of the mappings with rejected entries
    removed and any missed metrics flagged.
    """
    model = cfg["claude"]["model"]
    max_tokens = cfg["claude"]["max_tokens"]
    temperature = cfg["claude"]["temperature"]

    prompt = VALIDATION_PROMPT.format(
        mappings_json=json.dumps(mappings, indent=2),
        memo_content=memo_content,
        proforma_data=proforma_data,
    )

    log.info("Calling Claude API for validation pass (prompt=%d chars)...", len(prompt))

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    log.info("Validation response received (%d chars, %s stop_reason)",
             len(raw), message.stop_reason)

    if message.stop_reason == "max_tokens":
        log.error(
            "Claude's validation response was cut off (hit max_tokens=%d). "
            "Increase max_tokens in config.yaml and retry.",
            max_tokens,
        )
        sys.exit(1)

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    result = json.loads(text)

    n_valid_table = len(result.get("validated_table_updates", []))
    n_valid_text = len(result.get("validated_text_updates", []))
    n_rejected = len(result.get("rejected", []))
    n_missed = len(result.get("missed", []))
    log.info("Validation: %d table, %d text validated | %d rejected | %d missed",
             n_valid_table, n_valid_text, n_rejected, n_missed)

    if n_rejected > 0:
        for rej in result["rejected"]:
            log.warning("  REJECTED: %s", rej.get("reason", "unknown"))
    if n_missed > 0:
        for miss in result["missed"]:
            log.warning("  MISSED: page %s — %s", miss.get("page", "?"),
                        miss.get("description", ""))

    # Return the validated mappings in the standard format
    return {
        "table_updates": result.get("validated_table_updates", []),
        "text_updates": result.get("validated_text_updates", []),
        "rejected": result.get("rejected", []),
        "missed": result.get("missed", []),
    }


# ============================================================================
# 8. APPLY TEXT / TABLE UPDATES  (python-pptx)
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


def _replace_in_shape(shape, old_text: str, new_text: str) -> bool:
    """Replace old_text in any text-frame shape, preserving formatting."""
    if not shape.has_text_frame:
        return False
    for para in shape.text_frame.paragraphs:
        if old_text not in para.text:
            continue
        if _replace_in_para(para, old_text, new_text):
            return True
    return False


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

        slide = prs.slides[page - 1]
        found = False
        for shape in slide.shapes:
            if not shape.has_table:
                continue
            # Match by table name if provided, else try all tables
            if tbl_name and shape.name != tbl_name:
                continue
            for row in shape.table.rows:
                # Match by row label
                r0 = row.cells[0].text.strip()
                if row_label and row_label not in r0:
                    continue
                cell = row.cells[col_idx]
                if old_val in cell.text:
                    if not dry_run:
                        _replace_in_cell(cell, old_val, new_val)
                    changes.append({
                        "page": page, "type": "table",
                        "location": f"{tbl_name} / {row_label} / col {col_idx}",
                        "old": old_val, "new": new_val, "source": source,
                    })
                    found = True
                    break
            if found:
                break

        if not found:
            log.warning("Table update NOT FOUND: page %d, '%s' -> '%s'",
                        page, old_val, new_val)

    # --- Text (narrative) updates ---
    for upd in mappings.get("text_updates", []):
        page = upd["page"]
        old_txt = upd["old_text"]
        new_txt = upd["new_text"]
        source = upd.get("source", "")

        slide = prs.slides[page - 1]
        found = False
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if old_txt in para.text:
                        if not dry_run:
                            for run in para.runs:
                                if old_txt in run.text:
                                    run.text = run.text.replace(old_txt, new_txt)
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

    if not dry_run:
        prs.save(memo_path)
        log.info("Memo saved with %d updates.", len(changes))
    else:
        log.info("Dry-run: %d updates identified (not saved).", len(changes))

    return changes


# ============================================================================
# 9. CHANGE LOG
# ============================================================================
def write_change_log(output_dir: str, all_changes: list, mappings: dict,
                     memo_path: str, proforma_path: str, backup_path: str):
    """Write a Markdown change-log summarizing every modification."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(output_dir, "CHANGE_LOG.md")

    with open(log_path, "w") as f:
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
                    f"{c['location']} | {old_display} | {new_display} | "
                    f"{c['source']} |\n")

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
# 10. MAIN
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

    # Initialize Claude client (used for both mapping and validation)
    client = anthropic.Anthropic(api_key=api_key)

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
    BATCH_THRESHOLD = 120_000  # chars; above this, process slides in batches
    prompt_size = len(proforma_data) + len(memo_content)
    if prompt_size > BATCH_THRESHOLD:
        log.info("Large prompt (%d chars) — processing slides in batches of 10",
                 prompt_size)
        memo_chunks = chunk_memo_by_pages(memo_content, pages_per_chunk=10)
        mappings = {"table_updates": [], "text_updates": []}
        for i, chunk in enumerate(memo_chunks, 1):
            log.info("Mapping batch %d / %d ...", i, len(memo_chunks))
            batch = get_metric_mappings(client, proforma_data, chunk, cfg)
            mappings["table_updates"].extend(batch.get("table_updates", []))
            mappings["text_updates"].extend(batch.get("text_updates", []))
    else:
        mappings = get_metric_mappings(client, proforma_data, memo_content, cfg)

    # Save raw mappings for audit
    map_dump = os.path.join(output_dir, "mappings_raw.json")
    with open(map_dump, "w") as f:
        json.dump(mappings, f, indent=2)
    log.info("Raw mappings saved to %s", map_dump)

    # ---- Step 5: Claude API — validation pass (QA reasoning step) ----
    # Second API call: Claude cross-checks the proposed updates to catch
    # errors — wrong old_value text, formatting mismatches, duplicates,
    # or missed metrics. This replaces human review of the mapping output.
    log.info("=" * 60)
    log.info("STEP 5: Claude API — validating mappings")
    log.info("=" * 60)
    validated = validate_mappings(client, mappings, proforma_data, memo_content, cfg)

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
