"""
Test harness for schedule (.mpp) integration.

Extracts schedule data from a real .mpp file, randomizes dates,
creates a minimal test memo with timeline tables, and runs through
get_metric_mappings to verify schedule-to-memo mapping.

Usage:
    python test_schedule.py "path/to/schedule.mpp"
    python test_schedule.py "path/to/schedule.mpp" --iterations 3
"""
import argparse
import json
import os
import random
import re
import sys
import tempfile
from datetime import datetime, timedelta

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memo_automator_app"))

import anthropic
from pptx import Presentation
from pptx.util import Inches, Pt

from memo_automator import (
    extract_schedule_data,
    get_metric_mappings,
    load_config,
)


# ============================================================================
# SCHEDULE EXTRACTION & RANDOMIZATION
# ============================================================================
def extract_and_parse_schedule(schedule_path: str, cfg: dict) -> tuple[str, list[dict]]:
    """
    Extract schedule data and also parse it into structured task dicts
    for randomization purposes.

    Returns (schedule_text, parsed_tasks).
    """
    schedule_text = extract_schedule_data(schedule_path, cfg)

    # Parse the text output back into structured data
    tasks = []
    for line in schedule_text.split("\n"):
        match = re.match(
            r'\s*\[L(\d+)\]\s+(.+?)(\s+\[MILESTONE\])?\s+\|\s+'
            r'Start:\s+(\S+)\s+\|\s+Finish:\s+(\S+)\s+\|\s+Dur:\s+(.+)',
            line,
        )
        if match:
            tasks.append({
                "level": int(match.group(1)),
                "name": match.group(2).strip(),
                "milestone": bool(match.group(3)),
                "start": match.group(4),
                "finish": match.group(5),
                "duration": match.group(6).strip(),
            })
    return schedule_text, tasks


def randomize_schedule_text(schedule_text: str, offset_days: int) -> str:
    """
    Shift all dates in schedule text by offset_days and slightly randomize
    durations (+-20%).

    Returns modified schedule text string.
    """
    def shift_date(match):
        date_str = match.group(0)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            shifted = dt + timedelta(days=offset_days)
            return shifted.strftime("%Y-%m-%d")
        except ValueError:
            return date_str

    # Shift all YYYY-MM-DD dates
    shifted = re.sub(r'\d{4}-\d{2}-\d{2}', shift_date, schedule_text)

    # Randomize durations slightly
    def randomize_dur(match):
        prefix = match.group(1)
        num = int(match.group(2))
        unit = match.group(3)
        factor = random.uniform(0.8, 1.2)
        new_num = max(1, round(num * factor))
        return f"{prefix}{new_num}{unit}"

    shifted = re.sub(r'(Dur:\s+)(\d+)(d|mo)', randomize_dur, shifted)
    return shifted


# ============================================================================
# TEST MEMO CREATION
# ============================================================================
def create_test_memo(path: str, tasks: list[dict]) -> str:
    """
    Create a minimal .pptx memo with:
    - A 'Project Timeline' table with milestone dates
    - A 'Construction Schedule' table
    - Narrative text with embedded dates

    Uses placeholder dates that should get updated by the mapper.
    """
    prs = Presentation()

    # --- Slide 1: Project Timeline table ---
    slide1 = prs.slides.add_slide(prs.slide_layouts[5])
    # Add title text
    txBox = slide1.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(8), Inches(0.5))
    txBox.text_frame.text = "Project Timeline"

    # Find milestone-like tasks for the table
    timeline_tasks = []
    keywords = ["closing", "permit", "entitlement", "approval", "zoning",
                 "co ", "certificate", "delivery", "move-in", "completion"]
    for t in tasks:
        name_lower = t["name"].lower()
        if any(kw in name_lower for kw in keywords) or t["milestone"]:
            timeline_tasks.append(t)
    # Take up to 8 milestones
    timeline_tasks = timeline_tasks[:8]
    if not timeline_tasks:
        # Fallback: use first few tasks
        timeline_tasks = tasks[:5]

    n_rows = len(timeline_tasks) + 1  # +1 for header
    table_shape = slide1.shapes.add_table(n_rows, 3, Inches(0.5), Inches(1.0),
                                          Inches(8), Inches(0.4 * n_rows))
    table_shape.name = "ProjectTimeline"
    table = table_shape.table
    table.cell(0, 0).text = "Milestone"
    table.cell(0, 1).text = "Start"
    table.cell(0, 2).text = "Finish"

    for i, t in enumerate(timeline_tasks, 1):
        table.cell(i, 0).text = t["name"]
        # Use placeholder dates (old dates that should be updated)
        table.cell(i, 1).text = "TBD"
        table.cell(i, 2).text = "TBD"

    # --- Slide 2: Construction Schedule table ---
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    txBox2 = slide2.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(8), Inches(0.5))
    txBox2.text_frame.text = "Construction Schedule"

    construction_tasks = []
    constr_keywords = ["abatement", "demolition", "demo", "construction",
                       "foundation", "vertical", "framing"]
    for t in tasks:
        name_lower = t["name"].lower()
        if any(kw in name_lower for kw in constr_keywords):
            construction_tasks.append(t)
    construction_tasks = construction_tasks[:6]
    if not construction_tasks:
        construction_tasks = tasks[-3:]

    n_rows2 = len(construction_tasks) + 1
    table_shape2 = slide2.shapes.add_table(n_rows2, 4, Inches(0.5), Inches(1.0),
                                           Inches(8), Inches(0.4 * n_rows2))
    table_shape2.name = "ConstructionSchedule"
    table2 = table_shape2.table
    table2.cell(0, 0).text = "Phase"
    table2.cell(0, 1).text = "Start"
    table2.cell(0, 2).text = "Finish"
    table2.cell(0, 3).text = "Duration"

    for i, t in enumerate(construction_tasks, 1):
        table2.cell(i, 0).text = t["name"]
        table2.cell(i, 1).text = "TBD"
        table2.cell(i, 2).text = "TBD"
        table2.cell(i, 3).text = "TBD"

    # --- Slide 3: Narrative text with embedded dates ---
    slide3 = prs.slides.add_slide(prs.slide_layouts[5])
    txBox3 = slide3.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(8), Inches(5))
    tf = txBox3.text_frame
    tf.word_wrap = True

    # Build narrative with placeholder dates
    narrative = (
        "The project is anticipated to close in Q4 2025. "
        "Construction is expected to begin in Q1 2026 with abatement and demolition "
        "activities commencing shortly after closing. "
        "Vertical construction is expected to commence in Q2 2026 "
        "and substantial completion is targeted for Q4 2028. "
        "The anticipated delivery date is Q1 2029."
    )
    tf.text = narrative

    prs.save(path)
    print(f"  Test memo created at {path} ({len(prs.slides)} slides)")
    return path


# ============================================================================
# MAIN TEST HARNESS
# ============================================================================
def run_test(schedule_path: str, iterations: int = 1):
    """Run the schedule integration test."""
    # Load config
    config_path = os.path.join(os.path.dirname(__file__),
                               "memo_automator_app", "config.yaml")
    cfg = load_config(config_path)

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment or st.secrets")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=300.0)

    print("=" * 70)
    print("SCHEDULE INTEGRATION TEST")
    print("=" * 70)
    print(f"Schedule file: {schedule_path}")
    print(f"Iterations: {iterations}")
    print()

    # Step 1: Extract schedule
    print("Step 1: Extracting schedule data...")
    schedule_text, parsed_tasks = extract_and_parse_schedule(schedule_path, cfg)
    print(f"  Extracted {len(parsed_tasks)} tasks")
    print(f"  Schedule text: {len(schedule_text)} chars")
    print()

    # Print schedule structure for review
    print("Schedule structure:")
    for t in parsed_tasks[:20]:
        indent = "  " * t["level"]
        ms = " [MILESTONE]" if t["milestone"] else ""
        print(f"  {indent}L{t['level']} {t['name']}{ms} | {t['start']} - {t['finish']}")
    if len(parsed_tasks) > 20:
        print(f"  ... ({len(parsed_tasks) - 20} more tasks)")
    print()

    for iteration in range(1, iterations + 1):
        print(f"{'='*70}")
        print(f"ITERATION {iteration}/{iterations}")
        print(f"{'='*70}")

        # Step 2: Randomize dates
        offset = random.randint(-180, 180)
        print(f"Step 2: Randomizing dates (offset: {offset:+d} days)...")
        randomized_schedule = randomize_schedule_text(schedule_text, offset)

        # Step 3: Create test memo
        print("Step 3: Creating test memo...")
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
            memo_path = f.name
        create_test_memo(memo_path, parsed_tasks)

        # Step 4: Extract memo content for mapping
        print("Step 4: Extracting memo content...")
        from memo_automator import extract_memo_content
        memo_content = extract_memo_content(memo_path, cfg)

        # Step 5: Combine proforma (empty) + schedule data
        combined_data = "(No proforma data for this test)\n\n" + randomized_schedule

        # Step 6: Run mapping
        print("Step 5: Running metric mappings...")
        try:
            mappings = get_metric_mappings(client, combined_data, memo_content, cfg)
            mappings.pop("_truncated", None)
        except Exception as e:
            print(f"  ERROR: Mapping failed: {e}")
            os.unlink(memo_path)
            continue

        # Step 7: Report results
        n_table = len(mappings.get("table_updates", []))
        n_text = len(mappings.get("text_updates", []))
        n_row = len(mappings.get("row_inserts", []))

        print(f"\nResults:")
        print(f"  Table updates: {n_table}")
        print(f"  Text updates:  {n_text}")
        print(f"  Row inserts:   {n_row}")
        print()

        if n_table > 0:
            print("  Table updates:")
            for u in mappings["table_updates"]:
                print(f"    p{u.get('page')} | {u.get('table_name')} | "
                      f"{u.get('row_label')} | '{u.get('old_value')}' -> '{u.get('new_value')}' "
                      f"| src: {u.get('source')}")
            print()

        if n_text > 0:
            print("  Text updates:")
            for u in mappings["text_updates"]:
                old = u.get("old_text", "")[:60]
                new = u.get("new_text", "")[:60]
                print(f"    p{u.get('page')} | '{old}' -> '{new}' | src: {u.get('source')}")
            print()

        # Analysis questions for rule refinement
        if n_table == 0 and n_text == 0:
            print("  WARNING: No mappings generated! Questions to investigate:")
            print("    - Does the schedule text format match what the prompt expects?")
            print("    - Are the task names recognizable as milestones?")
            print("    - Is the test memo structured with detectable date fields?")
        else:
            # Check which milestones were and weren't mapped
            mapped_sources = set()
            for u in mappings.get("table_updates", []):
                mapped_sources.add(u.get("source", ""))
            for u in mappings.get("text_updates", []):
                mapped_sources.add(u.get("source", ""))

            print(f"  Mapped sources: {len(mapped_sources)}")
            for src in sorted(mapped_sources):
                print(f"    - {src}")

        # Cleanup
        os.unlink(memo_path)
        print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test schedule integration")
    parser.add_argument("schedule", help="Path to .mpp schedule file")
    parser.add_argument("--iterations", "-n", type=int, default=1,
                        help="Number of randomized iterations (default: 1)")
    args = parser.parse_args()

    if not os.path.exists(args.schedule):
        print(f"ERROR: Schedule file not found: {args.schedule}")
        sys.exit(1)

    run_test(args.schedule, args.iterations)
