"""
End-to-end test harness — mirrors the full app pipeline.

Takes the REAL base memo, applies proforma + schedule updates via Claude,
then brands and normalizes layout. The output is the edited base memo,
not a synthetic file.

Usage:
    python test_schedule.py
    python test_schedule.py --memo "path/to/memo.pptx"
    python test_schedule.py --schedule "path/to/schedule.mpp"
    python test_schedule.py --skip-validation
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memo_automator_app"))

import anthropic

from memo_automator import (
    apply_branding,
    apply_updates,
    chunk_memo_by_pages,
    create_backup,
    extract_memo_content,
    extract_proforma_data,
    extract_schedule_data,
    get_metric_mappings,
    load_config,
    normalize_layout,
    pre_validate_mappings,
    validate_mappings,
    write_change_log,
)

# ============================================================================
# DEFAULT PATHS (Limestone project)
# ============================================================================
DEFAULT_MEMO = r"C:\Users\BrandonZmuda\Desktop\Claude\g. Memo Automator\v2\a. Sandbox\Project Approval Update_UK_Limestone_20260218.pptx"
DEFAULT_PROFORMA = r"C:\Users\BrandonZmuda\Desktop\Claude\g. Memo Automator\v2\a. Sandbox\Proforma_Lexington-Limestone_20241021.xlsm"
DEFAULT_SCHEDULE = r"C:\Users\BrandonZmuda\Desktop\Lexington\c. Limestone\Internal Schedule_Lexington_Limestone_20251022 (working).mpp"


# ============================================================================
# MAIN TEST
# ============================================================================
def run_test(memo_path: str, proforma_path: str, schedule_path: str,
             skip_validation: bool = False):
    """Run the full pipeline against the real base memo."""
    config_path = os.path.join(os.path.dirname(__file__),
                               "memo_automator_app", "config.yaml")
    cfg = load_config(config_path)

    # Get API key
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), "memo_automator_app", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key, max_retries=5, timeout=900.0)

    # Output directory — same as memo
    output_dir = os.path.dirname(os.path.abspath(memo_path))

    print("=" * 70)
    print("MEMO AUTOMATOR — END-TO-END TEST")
    print("=" * 70)
    print(f"  Memo:      {memo_path}")
    print(f"  Proforma:  {proforma_path}")
    print(f"  Schedule:  {schedule_path}")
    print(f"  Validate:  {not skip_validation}")
    print()

    # ----- Step 1: Backup -----
    print("Step 1: Creating backup...")
    backup_path = create_backup(memo_path, output_dir)
    print(f"  Backup: {backup_path}")

    # ----- Step 2: Extract proforma data -----
    print("\nStep 2: Extracting proforma data...")
    proforma_data = extract_proforma_data(proforma_path, cfg)
    print(f"  {len(proforma_data)} chars")

    # ----- Step 3: Extract schedule data (append to proforma) -----
    if schedule_path and os.path.exists(schedule_path):
        print("\nStep 3: Extracting schedule data...")
        schedule_data = extract_schedule_data(schedule_path, cfg)
        proforma_data += "\n\n" + schedule_data
        print(f"  {len(schedule_data)} chars from schedule")
    else:
        print("\nStep 3: No schedule file — skipping")

    # ----- Step 4: Extract memo content -----
    print("\nStep 4: Extracting memo content...")
    memo_content = extract_memo_content(memo_path, cfg)
    print(f"  {len(memo_content)} chars")

    # ----- Step 5: Claude API — metric mapping -----
    print(f"\nStep 5: Running metric mappings...")
    BATCH_THRESHOLD = 80_000
    RATE_LIMIT_INTERVAL = 65
    prompt_size = len(proforma_data) + len(memo_content)
    print(f"  Prompt size: {prompt_size} chars")

    if prompt_size > BATCH_THRESHOLD:
        print(f"  Large prompt — processing in batches of 3 slides")
        memo_chunks = chunk_memo_by_pages(memo_content, pages_per_chunk=3)
        mappings = {"table_updates": [], "text_updates": [], "row_inserts": []}
        last_api_call = 0
        for i, chunk in enumerate(memo_chunks, 1):
            if i > 1 and last_api_call > 0:
                elapsed = time.time() - last_api_call
                wait = RATE_LIMIT_INTERVAL - elapsed
                if wait > 0:
                    print(f"  Rate limit: waiting {wait:.0f}s...")
                    time.sleep(wait)
            print(f"  Mapping batch {i}/{len(memo_chunks)}...")
            last_api_call = time.time()
            try:
                batch = get_metric_mappings(client, proforma_data, chunk, cfg)
            except Exception as e:
                print(f"  Batch {i} failed ({e}) — skipping")
                continue
            batch.pop("_truncated", None)
            mappings["table_updates"].extend(batch.get("table_updates", []))
            mappings["text_updates"].extend(batch.get("text_updates", []))
            mappings["row_inserts"].extend(batch.get("row_inserts", []))
    else:
        mappings = get_metric_mappings(client, proforma_data, memo_content, cfg)
        mappings.pop("_truncated", None)

    # Strip no-ops
    mappings["table_updates"] = [
        e for e in mappings["table_updates"]
        if e.get("old_value") != e.get("new_value")
    ]
    mappings["text_updates"] = [
        e for e in mappings["text_updates"]
        if e.get("old_text") != e.get("new_text")
    ]

    # Pre-validate
    mappings = pre_validate_mappings(mappings, memo_content)

    n_table = len(mappings.get("table_updates", []))
    n_text = len(mappings.get("text_updates", []))
    n_row = len(mappings.get("row_inserts", []))
    print(f"  Mappings: {n_table} table, {n_text} text, {n_row} row inserts")

    # ----- Step 6: Validation (optional) -----
    if skip_validation:
        print("\nStep 6: Validation SKIPPED")
        validated = mappings
        validated.setdefault("rejected", [])
        validated.setdefault("missed", [])
    else:
        print("\nStep 6: Validating mappings...")
        if last_api_call:
            elapsed = time.time() - last_api_call
            wait = RATE_LIMIT_INTERVAL - elapsed
            if wait > 0:
                print(f"  Rate limit: waiting {wait:.0f}s...")
                time.sleep(wait)
        validated = validate_mappings(client, mappings, proforma_data, memo_content, cfg)

    # Save mappings
    map_path = os.path.join(output_dir, "test_mappings.json")
    with open(map_path, "w") as f:
        json.dump(validated, f, indent=2)
    print(f"  Mappings saved: {map_path}")

    # ----- Step 7: Apply updates -----
    print(f"\nStep 7: Applying updates to memo...")
    all_changes = apply_updates(memo_path, validated, dry_run=False)
    print(f"  {len(all_changes)} changes applied")

    # ----- Step 8: Apply branding -----
    print(f"\nStep 8: Applying Subtext branding...")
    theme_path = os.path.join(os.path.dirname(__file__),
                               "memo_automator_app", "Subtext Brand Theme.thmx")
    if os.path.exists(theme_path):
        n_branded = apply_branding(memo_path, theme_path, cfg)
        print(f"  {n_branded} text runs reformatted")
    else:
        print(f"  WARNING: Theme not found at {theme_path}")

    # ----- Step 9: Normalize layout -----
    print(f"\nStep 9: Normalizing layout...")
    layout_summary = normalize_layout(memo_path, cfg)
    print(f"  {layout_summary}")

    # ----- Step 10: Change log -----
    print(f"\nStep 10: Writing change log...")
    log_path = write_change_log(output_dir, all_changes, validated,
                                memo_path, proforma_path, backup_path)
    print(f"  {log_path}")

    # ----- Summary -----
    n_rejected = len(validated.get("rejected", []))
    n_missed = len(validated.get("missed", []))

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Changes applied:     {len(all_changes)}")
    print(f"  Rejected by QA:      {n_rejected}")
    print(f"  Potentially missed:  {n_missed}")
    print(f"  Backup:              {backup_path}")
    print(f"  Output memo:         {memo_path}")
    print(f"  Change log:          {log_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end memo automator test")
    parser.add_argument("--memo", default=DEFAULT_MEMO,
                        help="Path to base memo (.pptx)")
    parser.add_argument("--proforma", default=DEFAULT_PROFORMA,
                        help="Path to proforma (.xlsx/.xlsm)")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE,
                        help="Path to schedule (.mpp)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the Claude validation pass")
    args = parser.parse_args()

    for p, label in [(args.memo, "Memo"), (args.proforma, "Proforma")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)

    if args.schedule and not os.path.exists(args.schedule):
        print(f"WARNING: Schedule not found: {args.schedule} — will skip schedule")
        args.schedule = None

    run_test(args.memo, args.proforma, args.schedule, args.skip_validation)
