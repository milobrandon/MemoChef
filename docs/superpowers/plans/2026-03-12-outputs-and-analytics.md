# Outputs & Analytics Feature Pack — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 features — proforma drift detection, slide image diff with redline overlay, run analytics dashboard, chart data updating from market workbooks, and market comp slide builder.

**Architecture:** Each feature is a self-contained module in `memo_chef/` with DB service functions in `app_services.py` and UI integration in `app.py`. Features are independent and can be shipped as separate PRs. Implementation follows spec priority order: F10 → F6 → F9 → F8 → F7.

**Tech Stack:** Python 3.11+, Streamlit, python-pptx, openpyxl, psycopg2 (PostgreSQL), Anthropic Claude API, pywin32 (COM), Pillow, rapidfuzz

**Spec:** `docs/superpowers/specs/2026-03-12-outputs-and-analytics-design.md`

---

## File Map

### New Files
| File | Responsibility | Feature |
|------|---------------|---------|
| `memo_chef/drift.py` | Proforma text parsing, cell diff algorithm | F10 |
| `tests/test_drift.py` | Unit tests for drift detection | F10 |
| `memo_chef/redline.py` | PowerPoint COM slide export, Pillow diff overlay | F6 |
| `tests/test_redline.py` | Unit tests for redline (mocked COM + real Pillow) | F6 |
| `tests/test_run_analytics.py` | Unit tests for analytics aggregation | F9 |
| `memo_chef/chart_extraction.py` | Market workbook tabular extraction | F8 |
| `prompts/chart_mapping_v1.txt` | Claude prompt for chart-to-memo mapping | F8 |
| `tests/test_chart_extraction.py` | Unit tests for chart extraction + mapping | F8 |
| `memo_chef/comp_builder.py` | Comp normalization, dedup, slide building | F7 |
| `tests/test_comp_builder.py` | Unit tests for comp builder | F7 |

### Modified Files
| File | Changes | Features |
|------|---------|----------|
| `app_services.py:53-193` | Add `proforma_snapshots` table, `store/get_snapshot()`, `get_run_analytics()` | F10, F9 |
| `app.py:568-806` | Drift panel in new run tab, redline expander in results, analytics in admin | F10, F6, F9 |
| `app.py:984-1006` | Analytics section in admin tab | F9 |
| `memo_chef/models.py` | Add fields to `RunRequest` for F7/F8 | F7, F8 |
| `memo_chef/pipeline.py` | Store proforma snapshot after run, integrate chart updates, comp builder | F10, F7, F8 |
| `memo_automator.py:2784-2879` | Add "Proforma Drift" section to `write_change_log()` | F10 |
| `memo_automator.py:2184-2309` | Extend `_apply_chart_updates()` for categories + data labels | F8 |
| `requirements.txt` | Add `pywin32`, `Pillow`, `rapidfuzz` | F6, F7 |

---

## Chunk 1: F10 — Proforma Drift Detection

### Task 0: Create feature branch

- [ ] **Step 1: Create branch from main**

```bash
git checkout main && git pull origin main
git checkout -b feat/proforma-drift-detection
```

---

### Task 1: Create `proforma_snapshots` DB table

**Files:**
- Modify: `app_services.py:53-193` (inside `get_db_conn()`)

- [ ] **Step 1: Add table creation SQL to `get_db_conn()`**

After the existing `ALTER TABLE` block for `change_log_html` (line 191), add:

```python
        cur.execute(
            "CREATE TABLE IF NOT EXISTS proforma_snapshots ("
            "  id TEXT PRIMARY KEY,"
            "  property_name TEXT NOT NULL,"
            "  run_id TEXT NOT NULL,"
            "  extracted_text TEXT NOT NULL,"
            "  tab_hashes JSONB,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_proforma_snapshots_property "
            "ON proforma_snapshots(property_name, created_at DESC)"
        )
```

- [ ] **Step 2: Verify table creation**

Run: `python -c "import streamlit as st; from app_services import get_db_conn; get_db_conn(); print('OK')"`

This will fail in a headless context due to `st.secrets`. Verify by running the Streamlit app and checking logs. If the app boots without errors, the migration is good.

- [ ] **Step 3: Commit**

```bash
git add app_services.py
git commit -m "feat(F10): add proforma_snapshots table migration"
```

---

### Task 2: Add snapshot storage and retrieval functions

**Files:**
- Modify: `app_services.py` (after `get_run_details()`, around line 581)

- [ ] **Step 1: Write `normalize_property_name()` helper**

```python
def normalize_property_name(name: str) -> str:
    """Normalize property name for consistent snapshot lookup."""
    n = name.strip().lower()
    for prefix in ("the ", "at "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n.strip()
```

- [ ] **Step 2: Write `store_proforma_snapshot()`**

```python
def store_proforma_snapshot(
    property_name: str,
    run_id: str,
    extracted_text: str,
    tab_hashes: dict[str, str] | None = None,
    max_snapshots: int = 3,
) -> None:
    """Store a proforma snapshot. Auto-prunes to keep last max_snapshots per property."""
    import json
    import uuid
    conn = get_db_conn()
    normalized = normalize_property_name(property_name)
    snapshot_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proforma_snapshots (id, property_name, run_id, extracted_text, tab_hashes) "
            "VALUES (%s, %s, %s, %s, %s)",
            (snapshot_id, normalized, run_id, extracted_text, json.dumps(tab_hashes) if tab_hashes else None),
        )
        # Prune old snapshots beyond max_snapshots
        cur.execute(
            "DELETE FROM proforma_snapshots WHERE id IN ("
            "  SELECT id FROM proforma_snapshots "
            "  WHERE property_name = %s "
            "  ORDER BY created_at DESC "
            "  OFFSET %s"
            ")",
            (normalized, max_snapshots),
        )
```

- [ ] **Step 3: Write `get_previous_proforma_snapshot()`**

```python
def get_previous_proforma_snapshot(property_name: str) -> dict | None:
    """Retrieve the most recent snapshot for a property.

    Returns: {"run_id": str, "extracted_text": str, "created_at": str} or None.
    """
    conn = get_db_conn()
    normalized = normalize_property_name(property_name)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_id, extracted_text, created_at "
            "FROM proforma_snapshots "
            "WHERE property_name = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (normalized,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"run_id": row[0], "extracted_text": row[1], "created_at": str(row[2])}
```

- [ ] **Step 4: Commit**

```bash
git add app_services.py
git commit -m "feat(F10): add snapshot storage and retrieval functions"
```

---

### Task 3: Build proforma text parser and diff engine

**Files:**
- Create: `memo_chef/drift.py`
- Create: `tests/test_drift.py`

- [ ] **Step 1: Write failing tests for `parse_proforma_to_cells()`**

Create `tests/test_drift.py`:

```python
"""Tests for proforma drift detection."""
from memo_chef.drift import parse_proforma_to_cells, compute_proforma_diff


SAMPLE_PROFORMA_TEXT = """
======================================================================
TAB: Assumptions
======================================================================
Row 1:\tProperty Name\tThe Reserve
Row 2:\tTotal Units\t250
Row 3:\tTotal Beds\t510
Row 5:\tAvg Rent/Bed\t1325

======================================================================
TAB: Development Summary
======================================================================
Row 1:\tCategory\tAmount\t% of Total
Row 2:\tLand\t12500000\t0.2403846
Row 3:\tHard Costs\t28000000\t0.5384615
"""


def test_parse_proforma_basic():
    result = parse_proforma_to_cells(SAMPLE_PROFORMA_TEXT)
    assert "Assumptions" in result
    assert "Development Summary" in result
    assert result["Assumptions"][1] == ["Property Name", "The Reserve"]
    assert result["Assumptions"][5] == ["Avg Rent/Bed", "1325"]
    assert result["Development Summary"][2] == ["Land", "12500000", "0.2403846"]


def test_parse_proforma_empty_string():
    result = parse_proforma_to_cells("")
    assert result == {}


def test_parse_proforma_skips_empty_rows():
    """Row 4 is missing from sample — should not appear in output."""
    result = parse_proforma_to_cells(SAMPLE_PROFORMA_TEXT)
    assert 4 not in result["Assumptions"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_drift.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `parse_proforma_to_cells()`**

Create `memo_chef/drift.py`:

```python
"""Proforma drift detection — parse, diff, and summarize proforma changes."""
from __future__ import annotations

import re


def parse_proforma_to_cells(text: str) -> dict[str, dict[int, list[str]]]:
    """Parse extract_proforma_data() text into structured cells.

    Returns: {"TabName": {row_num: [val1, val2, ...], ...}, ...}
    """
    if not text.strip():
        return {}

    result: dict[str, dict[int, list[str]]] = {}
    current_tab: str | None = None
    separator = "=" * 70

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == separator:
            continue
        tab_match = re.match(r"^TAB:\s*(.+)$", stripped)
        if tab_match:
            current_tab = tab_match.group(1).strip()
            result[current_tab] = {}
            continue
        if current_tab is None:
            continue
        row_match = re.match(r"^Row (\d+):\t(.+)$", line)
        if row_match:
            row_num = int(row_match.group(1))
            cells = row_match.group(2).split("\t")
            result[current_tab][row_num] = cells

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_drift.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Write failing tests for `compute_proforma_diff()`**

Append to `tests/test_drift.py`:

```python
SAMPLE_PROFORMA_V2 = """
======================================================================
TAB: Assumptions
======================================================================
Row 1:\tProperty Name\tThe Reserve
Row 2:\tTotal Units\t260
Row 3:\tTotal Beds\t530
Row 5:\tAvg Rent/Bed\t1350
Row 7:\tNew Metric\t42

======================================================================
TAB: Development Summary
======================================================================
Row 1:\tCategory\tAmount\t% of Total
Row 2:\tLand\t13000000\t0.25
Row 3:\tHard Costs\t28000000\t0.5384615
"""


def test_diff_detects_changed_values():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assumptions = diff["by_tab"]["Assumptions"]
    changed_rows = [c["row"] for c in assumptions["changed"]]
    assert 2 in changed_rows  # Units 250 -> 260
    assert 3 in changed_rows  # Beds 510 -> 530
    assert 5 in changed_rows  # Rent 1325 -> 1350


def test_diff_detects_added_rows():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assumptions = diff["by_tab"]["Assumptions"]
    added_rows = [a["row"] for a in assumptions["added"]]
    assert 7 in added_rows


def test_diff_detects_removed_rows():
    # Remove a row from v2 that exists in v1 — use reversed order
    diff = compute_proforma_diff(SAMPLE_PROFORMA_V2, SAMPLE_PROFORMA_TEXT)
    assumptions = diff["by_tab"]["Assumptions"]
    removed_rows = [r["row"] for r in assumptions["removed"]]
    assert 7 in removed_rows


def test_diff_unchanged_rows_not_in_changed():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assumptions = diff["by_tab"]["Assumptions"]
    changed_rows = [c["row"] for c in assumptions["changed"]]
    assert 1 not in changed_rows  # Property Name unchanged


def test_diff_total_changes():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    # Assumptions: 3 changed (rows 2,3,5) + 1 added (row 7)
    # Dev Summary: 1 changed (row 2)
    assert diff["total_changes"] == 5


def test_diff_summary_string():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_V2)
    assert "5 values changed" in diff["summary"]
    assert "2 tabs" in diff["summary"]


def test_diff_identical_proformas():
    diff = compute_proforma_diff(SAMPLE_PROFORMA_TEXT, SAMPLE_PROFORMA_TEXT)
    assert diff["total_changes"] == 0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_drift.py -v`
Expected: FAIL (new tests fail, old tests pass)

- [ ] **Step 7: Implement `compute_proforma_diff()`**

Add to `memo_chef/drift.py`:

```python
def compute_proforma_diff(previous_text: str, current_text: str) -> dict:
    """Compare two proforma text extractions and return structured diff."""
    prev = parse_proforma_to_cells(previous_text)
    curr = parse_proforma_to_cells(current_text)

    all_tabs = sorted(set(prev.keys()) | set(curr.keys()))
    by_tab: dict[str, dict] = {}
    total_changes = 0
    tabs_with_changes = 0

    for tab in all_tabs:
        prev_rows = prev.get(tab, {})
        curr_rows = curr.get(tab, {})
        all_row_nums = sorted(set(prev_rows.keys()) | set(curr_rows.keys()))

        added: list[dict] = []
        changed: list[dict] = []
        removed: list[dict] = []

        for row_num in all_row_nums:
            if row_num not in prev_rows:
                added.append({"row": row_num, "values": curr_rows[row_num]})
            elif row_num not in curr_rows:
                removed.append({"row": row_num, "values": prev_rows[row_num]})
            elif prev_rows[row_num] != curr_rows[row_num]:
                # Find which cells changed
                old_cells = prev_rows[row_num]
                new_cells = curr_rows[row_num]
                max_len = max(len(old_cells), len(new_cells))
                for col_idx in range(max_len):
                    old_val = old_cells[col_idx] if col_idx < len(old_cells) else ""
                    new_val = new_cells[col_idx] if col_idx < len(new_cells) else ""
                    if old_val != new_val:
                        changed.append({
                            "row": row_num,
                            "col_idx": col_idx,
                            "old": old_val,
                            "new": new_val,
                        })

        tab_total = len(added) + len(changed) + len(removed)
        if tab_total > 0:
            tabs_with_changes += 1
        total_changes += tab_total
        by_tab[tab] = {"added": added, "changed": changed, "removed": removed}

    return {
        "total_changes": total_changes,
        "by_tab": by_tab,
        "summary": f"{total_changes} values changed across {tabs_with_changes} tab{'s' if tabs_with_changes != 1 else ''}",
    }
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/test_drift.py -v`
Expected: All PASSED

- [ ] **Step 9: Commit**

```bash
git add memo_chef/drift.py tests/test_drift.py
git commit -m "feat(F10): implement proforma text parser and diff engine"
```

---

### Task 4: Integrate drift detection into Streamlit UI

**Files:**
- Modify: `app.py:568-635` (inside `render_new_run_tab()`)
- Modify: `app.py:16-46` (imports)

- [ ] **Step 1: Add imports to `app.py`**

Add to the import block near the top of `app.py` (after the `app_services` imports, around line 46):

```python
from memo_chef.drift import compute_proforma_diff
```

Add `get_previous_proforma_snapshot` and `normalize_property_name` to the `app_services` import block (line 16-46):

```python
from app_services import (
    ...
    get_previous_proforma_snapshot,
    normalize_property_name,
    ...
)
```

- [ ] **Step 2: Add drift detection UI after property name input**

In `render_new_run_tab()`, after the `property_name` text input (line 625-631) and before the file processing section, add a drift detection panel. Find the section after `property_rename_to` is defined and before the run config continues:

```python
    # --- Proforma drift detection ---
    if property_name and proforma_file is not None:
        prev_snapshot = get_previous_proforma_snapshot(property_name)
        if prev_snapshot:
            try:
                from memo_automator import load_config, extract_proforma_data
                import tempfile, os
                cfg = load_config(config_path)
                # Write uploaded proforma to temp file for extraction
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(proforma_file.name)[1],
                    delete=False,
                ) as tmp:
                    tmp.write(proforma_file.getvalue())
                    tmp_path = tmp.name
                try:
                    current_text = extract_proforma_data(tmp_path, cfg)
                    st.session_state["cached_proforma_text"] = current_text
                    diff = compute_proforma_diff(prev_snapshot["extracted_text"], current_text)
                    if diff["total_changes"] > 0:
                        st.info(
                            f"**Proforma drift detected:** {diff['summary']} "
                            f"(vs. run on {prev_snapshot['created_at'][:10]})"
                        )
                        with st.expander("View proforma changes"):
                            for tab_name, changes in diff["by_tab"].items():
                                n = len(changes["changed"]) + len(changes["added"]) + len(changes["removed"])
                                if n == 0:
                                    continue
                                st.markdown(f"**{tab_name}**: {n} changes")
                                if changes["changed"]:
                                    rows = []
                                    for c in changes["changed"]:
                                        rows.append({"Row": c["row"], "Column": c["col_idx"], "Previous": c["old"], "Current": c["new"]})
                                    st.dataframe(rows, use_container_width=True, hide_index=True)
                                if changes["added"]:
                                    st.caption(f"{len(changes['added'])} new rows added")
                                if changes["removed"]:
                                    st.caption(f"{len(changes['removed'])} rows removed")
                finally:
                    os.unlink(tmp_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("Drift detection skipped: %s", e)
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(F10): add drift detection panel in new run tab"
```

---

### Task 5: Store snapshot after pipeline run + change log integration

**Files:**
- Modify: `memo_chef/pipeline.py` (after run completes)
- Modify: `memo_automator.py:2784-2879` (`write_change_log()`)
- Modify: `app.py:16-46` (add `store_proforma_snapshot` import)

- [ ] **Step 1: Add snapshot storage to pipeline**

In `memo_chef/pipeline.py`, inside the `with checkpoint.stage("artifacts", ...)` block (around line 645), after `checkpoint.set_count("estimated_cost_microdollars", ...)` (line 664) and before `checkpoint.manifest.status = "completed"` (line 666), add:

```python
            # Store proforma snapshot for drift detection
            if request.property_name:
                try:
                    from app_services import store_proforma_snapshot
                    store_proforma_snapshot(
                        property_name=request.property_name,
                        run_id=request.run_id,
                        extracted_text=proforma_text,  # already extracted earlier in pipeline
                    )
                except Exception as e:
                    log.warning("Failed to store proforma snapshot: %s", e)
```

- [ ] **Step 2: Add drift section to `write_change_log()`**

In `memo_automator.py`, modify `write_change_log()` to accept an optional `proforma_drift` parameter and render it before "Applied Changes".

Add parameter to function signature at line 2784 (note: actual params are `memo_path`, `proforma_path`, `backup_path`):

```python
def write_change_log(output_dir: str, all_changes: list, mappings: dict,
                     memo_path: str, proforma_path: str, backup_path: str,
                     run_metadata: dict | None = None,
                     proforma_drift: dict | None = None):
```

After the confidence score section (after `f.write(f"- Miss rate: ...")` around line 2827) and before the applied changes table, add inside the `with open(log_path, "w") as f:` block. Note: `_md_cell()` is a nested function inside `write_change_log()`, so this code must go inside the same `with` block:

```python
        # Proforma drift section
        if proforma_drift and proforma_drift.get("total_changes", 0) > 0:
            f.write("\n## Proforma Drift\n\n")
            f.write(f"**{proforma_drift['summary']}**\n\n")
            for tab_name, changes in proforma_drift["by_tab"].items():
                n = len(changes["changed"]) + len(changes["added"]) + len(changes["removed"])
                if n == 0:
                    continue
                f.write(f"\n### {tab_name} ({n} changes)\n\n")
                if changes["changed"]:
                    f.write("| Row | Col | Previous | Current |\n")
                    f.write("|-----|-----|----------|---------|\\n")
                    for c in changes["changed"]:
                        old = _md_cell(str(c["old"]))
                        new = _md_cell(str(c["new"]))
                        f.write(f"| {c['row']} | {c['col_idx']} | {old} | {new} |\n")
                if changes["added"]:
                    f.write(f"\n*{len(changes['added'])} new rows added*\n")
                if changes["removed"]:
                    f.write(f"\n*{len(changes['removed'])} rows removed*\n")
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -x -v`
Expected: All tests pass (new parameter is optional, no existing callers break)

- [ ] **Step 4: Commit**

```bash
git add memo_chef/pipeline.py memo_automator.py app.py
git commit -m "feat(F10): store snapshot after run, add drift to change log"
```

- [ ] **Step 5: Create PR**

```bash
git checkout -b feat/proforma-drift-detection
git push -u origin feat/proforma-drift-detection
gh pr create --title "feat: proforma drift detection (F10)" --body "$(cat <<'EOF'
## Summary
- New `memo_chef/drift.py`: proforma text parser + cell-level diff engine
- New `proforma_snapshots` DB table for storing extraction history per property
- Drift detection panel in Streamlit UI (shows before vs. after proforma changes)
- Drift summary section in CHANGE_LOG.md
- Property name normalization for consistent lookup

## Test plan
- [ ] Run `pytest tests/test_drift.py -v` — all drift tests pass
- [ ] Run `pytest -x` — full suite passes, no regressions
- [ ] Manual: run a memo update, then change the proforma and run again — drift panel should appear
- [ ] Manual: verify CHANGE_LOG.md includes drift section on second run

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Chunk 2: F6 — Slide Image Diff with Redline Overlay

### Task 5.5: Create feature branch

- [ ] **Step 1: Create branch from main**

```bash
git checkout main && git pull origin main
git checkout -b feat/redline-diff-overlay
```

---

### Task 6: Install dependencies and create redline module scaffold

**Files:**
- Modify: `requirements.txt`
- Create: `memo_chef/redline.py`

- [ ] **Step 1: Add dependencies**

```bash
pip install pywin32 Pillow
```

Add to `requirements.txt`:

```
pywin32>=306
Pillow>=10.0
```

- [ ] **Step 2: Create `memo_chef/redline.py` scaffold with availability check**

```python
"""Slide image diff with redline overlay using PowerPoint COM + Pillow."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_COM_LOCK = threading.Lock()

_POWERPOINT_AVAILABLE: bool | None = None


def is_powerpoint_available() -> bool:
    """Check if PowerPoint COM automation is available on this machine."""
    global _POWERPOINT_AVAILABLE
    if _POWERPOINT_AVAILABLE is None:
        try:
            import win32com.client  # noqa: F401
            _POWERPOINT_AVAILABLE = True
        except ImportError:
            _POWERPOINT_AVAILABLE = False
            log.info("pywin32 not available — redline feature disabled")
    return _POWERPOINT_AVAILABLE
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt memo_chef/redline.py
git commit -m "feat(F6): add redline module scaffold and dependencies"
```

---

### Task 7: Implement slide-to-image export via PowerPoint COM

**Files:**
- Modify: `memo_chef/redline.py`
- Create: `tests/test_redline.py`

- [ ] **Step 1: Write failing test for `export_slides_as_images()`**

Create `tests/test_redline.py`:

```python
"""Tests for slide image diff and redline overlay."""
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest


def test_export_not_available_returns_empty():
    """When PowerPoint is not available, export returns empty dict."""
    from memo_chef.redline import export_slides_as_images
    with patch("memo_chef.redline.is_powerpoint_available", return_value=False):
        result = export_slides_as_images("fake.pptx", [1, 2], "/tmp/out")
        assert result == {}


def test_export_calls_com_correctly():
    """Verify COM calls: open presentation, export slides, quit."""
    from memo_chef.redline import export_slides_as_images

    mock_app = MagicMock()
    mock_prs = MagicMock()
    mock_slide = MagicMock()
    mock_app.Presentations.Open.return_value = mock_prs
    mock_prs.Slides.return_value = mock_slide

    with patch("memo_chef.redline.is_powerpoint_available", return_value=True), \
         patch("memo_chef.redline._create_com_app", return_value=mock_app), \
         patch("memo_chef.redline._export_slide") as mock_export:
        mock_export.return_value = Path("/tmp/out/slide_3.png")
        result = export_slides_as_images("test.pptx", [3], "/tmp/out")

    mock_app.Presentations.Open.assert_called_once()
    mock_app.Quit.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_redline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `export_slides_as_images()`**

Add to `memo_chef/redline.py`:

```python
def _create_com_app():
    """Create a PowerPoint COM application instance."""
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = False
    return app


def _cleanup_com_app(app):
    """Safely close COM application."""
    try:
        app.Quit()
    except Exception:
        pass
    try:
        import pythoncom
        pythoncom.CoUninitialize()
    except Exception:
        pass


def _export_slide(slide, output_path: Path, width: int = 1920) -> Path:
    """Export a single slide as PNG."""
    slide.Export(str(output_path), "PNG", width)
    return output_path


def export_slides_as_images(
    pptx_path: str,
    slide_numbers: list[int],
    output_dir: str,
    dpi: int = 150,
    max_slides: int = 20,
) -> dict[int, Path]:
    """Export specific slides from a PPTX as PNG using PowerPoint COM.

    Returns dict mapping slide number to PNG file path.
    Returns empty dict if PowerPoint is not available.
    """
    if not is_powerpoint_available():
        return {}

    slide_numbers = slide_numbers[:max_slides]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}

    with _COM_LOCK:
        app = None
        try:
            app = _create_com_app()
            prs = app.Presentations.Open(
                os.path.abspath(pptx_path),
                ReadOnly=True,
                Untitled=False,
                WithWindow=False,
            )
            try:
                for slide_num in slide_numbers:
                    if 1 <= slide_num <= prs.Slides.Count:
                        slide = prs.Slides(slide_num)
                        png_path = output_path / f"slide_{slide_num}.png"
                        _export_slide(slide, png_path)
                        if png_path.exists():
                            result[slide_num] = png_path
            finally:
                prs.Close()
        except Exception as e:
            log.warning("PowerPoint COM export failed: %s", e)
        finally:
            if app is not None:
                _cleanup_com_app(app)

    return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_redline.py -v`
Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add memo_chef/redline.py tests/test_redline.py
git commit -m "feat(F6): implement COM-based slide export"
```

---

### Task 8: Implement Pillow-based redline diff overlay

**Files:**
- Modify: `memo_chef/redline.py`
- Modify: `tests/test_redline.py`

- [ ] **Step 1: Write failing tests for `generate_redline_image()` and `generate_side_by_side()`**

Append to `tests/test_redline.py`:

```python
from PIL import Image
import io


def _make_solid_png(color: tuple, size: tuple = (200, 150)) -> bytes:
    """Create a solid-color PNG as bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_redline_identical_images_no_highlights():
    from memo_chef.redline import generate_redline_image
    img = _make_solid_png((255, 255, 255))
    result = generate_redline_image(img, img)
    assert isinstance(result, bytes)
    # Should produce a valid PNG
    out = Image.open(io.BytesIO(result))
    assert out.size == (200, 150)


def test_redline_different_images_has_red_overlay():
    from memo_chef.redline import generate_redline_image
    before = _make_solid_png((255, 255, 255))
    after = _make_solid_png((200, 200, 200))  # slightly different
    result = generate_redline_image(before, after, threshold=10)
    out = Image.open(io.BytesIO(result))
    # Check that there are red-ish pixels in the output (overlay was applied)
    pixels = list(out.getdata())
    red_pixels = [p for p in pixels if p[0] > 200 and p[1] < 100]
    assert len(red_pixels) > 0


def test_side_by_side_doubles_width():
    from memo_chef.redline import generate_side_by_side
    img = _make_solid_png((255, 255, 255), size=(200, 150))
    result = generate_side_by_side(img, img)
    out = Image.open(io.BytesIO(result))
    # Side-by-side should be roughly 2x width + gap
    assert out.width > 350
    assert out.height >= 150
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_redline.py::test_redline_identical_images_no_highlights -v`
Expected: FAIL

- [ ] **Step 3: Implement `generate_redline_image()` and `generate_side_by_side()`**

Add to `memo_chef/redline.py`:

```python
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
import io


def generate_redline_image(
    before_img: bytes,
    after_img: bytes,
    threshold: int = 30,
) -> bytes:
    """Generate a redline overlay highlighting pixel differences.

    Takes the 'after' image and overlays red-tinted bounding boxes
    around regions that differ from the 'before' image.
    """
    before = Image.open(io.BytesIO(before_img)).convert("RGB")
    after = Image.open(io.BytesIO(after_img)).convert("RGB")

    # Resize to match if needed
    if before.size != after.size:
        after = after.resize(before.size, Image.LANCZOS)

    # Compute pixel diff
    diff = ImageChops.difference(before, after).convert("L")
    # Threshold to binary mask
    mask = diff.point(lambda p: 255 if p > threshold else 0)
    # Dilate to make regions more visible
    mask = mask.filter(ImageFilter.MaxFilter(size=7))

    # Create red overlay
    overlay = Image.new("RGBA", after.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Find bounding boxes of changed regions
    bbox = mask.getbbox()
    if bbox:
        # Paint semi-transparent red over changed area
        red_mask = Image.new("L", after.size, 0)
        red_mask.paste(mask, (0, 0))
        red_overlay = Image.new("RGBA", after.size, (255, 50, 50, 80))
        overlay = Image.composite(red_overlay, overlay, red_mask)
        # Draw border around changed region
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(bbox, outline=(255, 0, 0, 200), width=3)

    # Composite
    result = Image.alpha_composite(after.convert("RGBA"), overlay)
    buf = io.BytesIO()
    result.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def generate_side_by_side(
    before_img: bytes,
    after_img: bytes,
    label_before: str = "Before",
    label_after: str = "After",
    gap: int = 20,
) -> bytes:
    """Generate a side-by-side comparison image."""
    before = Image.open(io.BytesIO(before_img)).convert("RGB")
    after = Image.open(io.BytesIO(after_img)).convert("RGB")

    if before.size != after.size:
        after = after.resize(before.size, Image.LANCZOS)

    label_height = 30
    total_width = before.width * 2 + gap
    total_height = before.height + label_height

    canvas = Image.new("RGB", (total_width, total_height), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    # Labels
    draw.text((before.width // 2 - 20, 5), label_before, fill=(0, 0, 0))
    draw.text((before.width + gap + after.width // 2 - 20, 5), label_after, fill=(0, 0, 0))

    # Images
    canvas.paste(before, (0, label_height))
    canvas.paste(after, (before.width + gap, label_height))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 4: Run all redline tests**

Run: `pytest tests/test_redline.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add memo_chef/redline.py tests/test_redline.py
git commit -m "feat(F6): implement Pillow-based redline diff overlay"
```

---

### Task 9: Integrate redline into Streamlit results and pipeline

**Files:**
- Modify: `app.py:272-283` (`_persist_result()`)
- Modify: `app.py:756-806` (results display)
- Modify: `app.py:16-46` (imports)

- [ ] **Step 1: Add redline generation after pipeline result**

In `app.py`, import the redline module:

```python
from memo_chef.redline import (
    export_slides_as_images,
    generate_redline_image,
    generate_side_by_side,
    is_powerpoint_available,
)
```

In `_persist_result()` (line 272), after persisting the existing result data, add redline generation:

```python
    # Generate redline images if PowerPoint is available
    if is_powerpoint_available() and result.changes:
        try:
            changed_pages = sorted({c["page"] for c in result.changes if "page" in c})
            run_dir = str(Path(result.memo_path).parent)
            # Export before slides (from backup)
            backup_path = None
            for f in Path(run_dir).glob("*_BACKUP_*"):
                backup_path = str(f)
                break
            if backup_path:
                before_imgs = export_slides_as_images(backup_path, changed_pages, run_dir + "/redline_before")
                after_imgs = export_slides_as_images(result.memo_path, changed_pages, run_dir + "/redline_after")

                redline_data = {}
                for page in changed_pages:
                    if page in before_imgs and page in after_imgs:
                        before_bytes = before_imgs[page].read_bytes()
                        after_bytes = after_imgs[page].read_bytes()
                        redline_data[page] = {
                            "redline": str(Path(run_dir) / f"redline_{page}.png"),
                            "side_by_side": str(Path(run_dir) / f"sbs_{page}.png"),
                            "before": str(before_imgs[page]),
                            "after": str(after_imgs[page]),
                        }
                        # Write redline and side-by-side to disk
                        Path(redline_data[page]["redline"]).write_bytes(
                            generate_redline_image(before_bytes, after_bytes)
                        )
                        Path(redline_data[page]["side_by_side"]).write_bytes(
                            generate_side_by_side(before_bytes, after_bytes)
                        )
                st.session_state["redline_images"] = redline_data
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Redline generation failed: %s", e)
```

- [ ] **Step 2: Add redline display in results section**

In the results section (after line 806, after the execution log expander), add:

```python
        redline_data = st.session_state.get("redline_images", {})
        if redline_data:
            with st.expander("Redline View", expanded=False):
                changed_slides = sorted(redline_data.keys())
                change_counts = {}
                for c in result_changes:
                    p = c.get("page", 0)
                    change_counts[p] = change_counts.get(p, 0) + 1

                selected_slide = st.selectbox(
                    "Slide",
                    changed_slides,
                    format_func=lambda x: f"Slide {x} ({change_counts.get(x, 0)} changes)",
                )
                view_mode = st.radio(
                    "View", ["Redline", "Side by side", "Before", "After"],
                    horizontal=True,
                )
                paths = redline_data[selected_slide]
                view_map = {
                    "Redline": "redline",
                    "Side by side": "side_by_side",
                    "Before": "before",
                    "After": "after",
                }
                img_path = paths[view_map[view_mode]]
                if Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning("Image file not found.")
        elif is_powerpoint_available():
            pass  # No changes to show
        else:
            st.caption("Redline view unavailable (requires PowerPoint).")
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -x -v`
Expected: All tests pass

- [ ] **Step 4: Commit and create PR**

```bash
git add app.py memo_chef/redline.py tests/test_redline.py requirements.txt
git checkout -b feat/redline-diff-overlay
git push -u origin feat/redline-diff-overlay
gh pr create --title "feat: slide image diff with redline overlay (F6)" --body "$(cat <<'EOF'
## Summary
- New `memo_chef/redline.py`: PowerPoint COM slide export + Pillow diff overlay
- Redline view in Streamlit results: slide picker, 4 view modes (redline/side-by-side/before/after)
- Graceful degradation when PowerPoint unavailable
- Thread-safe COM with cleanup guards
- New deps: `pywin32`, `Pillow`

## Test plan
- [ ] Run `pytest tests/test_redline.py -v` — all tests pass
- [ ] Run full pipeline with PowerPoint installed — redline images generated
- [ ] Check redline view in Streamlit — slider, view modes work
- [ ] Test without pywin32 installed — graceful fallback, no crash

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Chunk 3: F9 — Run Analytics Dashboard

### Task 9.5: Create feature branch

- [ ] **Step 1: Create branch from main**

```bash
git checkout main && git pull origin main
git checkout -b feat/run-analytics-dashboard
```

---

### Task 10: Implement `get_run_analytics()` service function

**Files:**
- Modify: `app_services.py` (after `get_run_details()`, around line 581)
- Create: `tests/test_run_analytics.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_run_analytics.py`:

```python
"""Tests for run analytics aggregation."""
import json
from unittest.mock import patch, MagicMock

import pytest


def _mock_rows():
    """Simulate query results from memo_chef_runs."""
    return [
        {
            "run_id": "r1", "username": "alice", "status": "completed",
            "change_count": 42, "rejected_count": 2, "missed_count": 1,
            "duration_seconds": 120.0, "estimated_cost_microdollars": 50000,
            "confidence_score": 92.5, "coverage_pct": 95.0,
            "warnings_json": json.dumps([{"stage": "validation", "message": "truncation"}]),
            "created_at": "2026-03-10T10:00:00",
        },
        {
            "run_id": "r2", "username": "bob", "status": "completed",
            "change_count": 30, "rejected_count": 0, "missed_count": 3,
            "duration_seconds": 90.0, "estimated_cost_microdollars": 40000,
            "confidence_score": 88.0, "coverage_pct": 90.0,
            "warnings_json": json.dumps([]),
            "created_at": "2026-03-11T14:00:00",
        },
    ]


def test_analytics_total_runs():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert result["total_runs"] == 2


def test_analytics_total_cost():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert result["total_cost_usd"] == pytest.approx(0.09, abs=0.01)


def test_analytics_avg_confidence():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert result["avg_confidence"] == pytest.approx(90.25, abs=0.1)


def test_analytics_by_user():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    users = {u["username"]: u for u in result["by_user"]}
    assert "alice" in users
    assert "bob" in users
    assert users["alice"]["runs"] == 1


def test_analytics_warning_counts():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert any(w["warning"] == "truncation" for w in result["warning_counts"])


def test_analytics_empty_input():
    from app_services import _compute_analytics
    result = _compute_analytics([])
    assert result["total_runs"] == 0
    assert result["avg_confidence"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_analytics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `_compute_analytics()` and `get_run_analytics()`**

Add to `app_services.py`:

```python
def _compute_analytics(rows: list[dict]) -> dict:
    """Compute analytics from a list of run row dicts (pure function, no DB)."""
    import json
    from collections import Counter

    if not rows:
        return {
            "total_runs": 0, "total_cost_usd": 0.0, "avg_confidence": 0.0,
            "avg_duration_sec": 0.0, "total_changes": 0,
            "cost_by_date": [], "accuracy_by_date": [], "by_user": [],
            "warning_counts": [],
        }

    total_cost_micro = sum(r.get("estimated_cost_microdollars", 0) or 0 for r in rows)
    confidences = [r["confidence_score"] for r in rows if r.get("confidence_score") is not None]
    durations = [r["duration_seconds"] for r in rows if r.get("duration_seconds") is not None]
    total_changes = sum(r.get("change_count", 0) or 0 for r in rows)

    # Cost by date
    cost_by_date: dict[str, float] = {}
    for r in rows:
        d = str(r["created_at"])[:10]
        cost_by_date[d] = cost_by_date.get(d, 0) + (r.get("estimated_cost_microdollars", 0) or 0) / 1_000_000

    # Accuracy by date (average when multiple runs per day)
    accuracy_by_date_accum: dict[str, dict] = {}
    for r in rows:
        d = str(r["created_at"])[:10]
        cc = (r.get("change_count", 0) or 0)
        rc = (r.get("rejected_count", 0) or 0)
        mc = (r.get("missed_count", 0) or 0)
        total = cc + rc
        rej_rate = (rc / total * 100) if total > 0 else 0.0
        miss_rate = (mc / (cc + mc) * 100) if (cc + mc) > 0 else 0.0
        if d not in accuracy_by_date_accum:
            accuracy_by_date_accum[d] = {"confs": [], "rejs": [], "misses": []}
        accuracy_by_date_accum[d]["confs"].append(r.get("confidence_score") or 0)
        accuracy_by_date_accum[d]["rejs"].append(rej_rate)
        accuracy_by_date_accum[d]["misses"].append(miss_rate)

    accuracy_by_date: dict[str, dict] = {}
    for d, acc in accuracy_by_date_accum.items():
        accuracy_by_date[d] = {
            "confidence": round(sum(acc["confs"]) / len(acc["confs"]), 1),
            "rejection_rate": round(sum(acc["rejs"]) / len(acc["rejs"]), 1),
            "miss_rate": round(sum(acc["misses"]) / len(acc["misses"]), 1),
        }

    # By user
    user_data: dict[str, dict] = {}
    for r in rows:
        u = r["username"]
        if u not in user_data:
            user_data[u] = {"runs": 0, "confidences": [], "cost_micro": 0, "last_run": ""}
        user_data[u]["runs"] += 1
        if r.get("confidence_score") is not None:
            user_data[u]["confidences"].append(r["confidence_score"])
        user_data[u]["cost_micro"] += r.get("estimated_cost_microdollars", 0) or 0
        user_data[u]["last_run"] = str(r["created_at"])[:10]

    by_user = []
    for username, data in sorted(user_data.items()):
        by_user.append({
            "username": username,
            "runs": data["runs"],
            "avg_confidence": round(sum(data["confidences"]) / len(data["confidences"]), 1) if data["confidences"] else 0.0,
            "total_cost_usd": round(data["cost_micro"] / 1_000_000, 4),
            "last_run": data["last_run"],
        })

    # Warning frequency
    warning_counter: Counter = Counter()
    for r in rows:
        try:
            warnings = json.loads(r.get("warnings_json") or "[]")
            for w in warnings:
                warning_counter[w.get("message", "unknown")] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "total_runs": len(rows),
        "total_cost_usd": round(total_cost_micro / 1_000_000, 4),
        "avg_confidence": round(sum(confidences) / len(confidences), 1) if confidences else 0.0,
        "avg_duration_sec": round(sum(durations) / len(durations), 1) if durations else 0.0,
        "total_changes": total_changes,
        "cost_by_date": [{"date": d, "cost_usd": v} for d, v in sorted(cost_by_date.items())],
        "accuracy_by_date": [{"date": d, **v} for d, v in sorted(accuracy_by_date.items())],
        "by_user": by_user,
        "warning_counts": [{"warning": w, "count": c} for w, c in warning_counter.most_common()],
    }


def get_run_analytics(days: int | None = None) -> dict:
    """Aggregate run statistics for the analytics dashboard."""
    conn = get_db_conn()
    with conn.cursor() as cur:
        if days:
            cur.execute(
                "SELECT run_id, username, status, change_count, rejected_count, missed_count, "
                "duration_seconds, estimated_cost_microdollars, confidence_score, coverage_pct, "
                "warnings_json, created_at "
                "FROM memo_chef_runs WHERE created_at >= NOW() - INTERVAL '%s days' "
                "ORDER BY created_at",
                (days,),
            )
        else:
            cur.execute(
                "SELECT run_id, username, status, change_count, rejected_count, missed_count, "
                "duration_seconds, estimated_cost_microdollars, confidence_score, coverage_pct, "
                "warnings_json, created_at "
                "FROM memo_chef_runs ORDER BY created_at"
            )
        columns = [desc[0] for desc in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return _compute_analytics(rows)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_run_analytics.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add app_services.py tests/test_run_analytics.py
git commit -m "feat(F9): implement run analytics aggregation service"
```

---

### Task 11: Add analytics dashboard to admin panel

**Files:**
- Modify: `app.py:984-1006` (inside `render_admin_tab()`)
- Modify: `app.py:16-46` (imports)

- [ ] **Step 1: Add import**

Add `get_run_analytics` to the `app_services` import block.

- [ ] **Step 2: Add analytics section to admin tab**

In `render_admin_tab()`, after the existing admin sub-tabs block (after line 1006), add a new section before the admin_tabs:

```python
def render_admin_tab() -> None:
    st.subheader("Admin")
    st.caption("Manage users, credits, and review system activity.")

    # --- Analytics Dashboard ---
    st.markdown("---")
    st.markdown("### Run Analytics")
    period_map = {"Last 7 days": 7, "Last 30 days": 30, "All time": None}
    period_label = st.selectbox("Period", list(period_map.keys()), index=1)
    analytics = get_run_analytics(days=period_map[period_label])

    summary_cols = st.columns(5)
    summary_cols[0].metric("Total Runs", analytics["total_runs"])
    summary_cols[1].metric("Avg Confidence", f"{analytics['avg_confidence']:.0f}/100" if analytics["avg_confidence"] else "—")
    summary_cols[2].metric("Total API Spend", f"${analytics['total_cost_usd']:.2f}")
    avg_dur_min = analytics["avg_duration_sec"] / 60 if analytics["avg_duration_sec"] else 0
    summary_cols[3].metric("Avg Duration", f"{avg_dur_min:.1f} min" if avg_dur_min else "—")
    summary_cols[4].metric("Total Changes", analytics["total_changes"])

    if analytics["cost_by_date"]:
        import pandas as pd
        cost_df = pd.DataFrame(analytics["cost_by_date"])
        st.markdown("**API Cost Trend**")
        st.line_chart(cost_df, x="date", y="cost_usd")

    if analytics["accuracy_by_date"]:
        acc_df = pd.DataFrame(analytics["accuracy_by_date"])
        st.markdown("**Accuracy Trend**")
        st.line_chart(acc_df, x="date", y=["confidence", "rejection_rate", "miss_rate"])

    if analytics["by_user"]:
        st.markdown("**Per-User Breakdown**")
        st.dataframe(analytics["by_user"], use_container_width=True, hide_index=True)

    if analytics["warning_counts"]:
        st.markdown("**Warning Frequency**")
        warn_df = pd.DataFrame(analytics["warning_counts"])
        st.bar_chart(warn_df, x="warning", y="count")

    # Time savings estimate
    manual_hours = 4.0  # configurable baseline
    total_hours_saved = (analytics["total_runs"] * manual_hours) - (analytics["avg_duration_sec"] * analytics["total_runs"] / 3600) if analytics["total_runs"] else 0
    if total_hours_saved > 0:
        st.metric("Estimated Hours Saved", f"{total_hours_saved:.0f} hrs")

    st.markdown("---")
    # ... existing user management code continues below
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -x -v`
Expected: All pass

- [ ] **Step 4: Commit and create PR**

```bash
git checkout -b feat/run-analytics-dashboard
git add app.py app_services.py tests/test_run_analytics.py
git push -u origin feat/run-analytics-dashboard
gh pr create --title "feat: run analytics dashboard in admin panel (F9)" --body "$(cat <<'EOF'
## Summary
- New `get_run_analytics()` and `_compute_analytics()` in app_services.py
- Analytics dashboard in admin tab: summary cards, cost/accuracy trends, per-user breakdown, warning frequency, time savings estimate
- Pure aggregation on existing `memo_chef_runs` data — no new tables

## Test plan
- [ ] Run `pytest tests/test_run_analytics.py -v` — all tests pass
- [ ] Manual: check admin panel shows analytics section
- [ ] Verify period filter (7d/30d/all) works
- [ ] Verify charts render with real run data

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Chunk 4: F8 — Chart Data Updating from Market Workbooks

### Task 11.5: Create feature branch

- [ ] **Step 1: Create branch from main**

```bash
git checkout main && git pull origin main
git checkout -b feat/chart-data-updating
```

---

### Task 12: Create chart extraction module and prompt

**Files:**
- Create: `memo_chef/chart_extraction.py`
- Create: `prompts/chart_mapping_v1.txt`
- Create: `tests/test_chart_extraction.py`

- [ ] **Step 1: Write failing test for `extract_workbook_tables()`**

Create `tests/test_chart_extraction.py`:

```python
"""Tests for market workbook chart extraction."""
import pytest
from pathlib import Path


def test_extract_workbook_tables_format(tmp_path):
    """Verify output matches proforma text format."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rent Growth"
    ws["A1"] = "Year"
    ws["B1"] = "Submarket"
    ws["A2"] = 2024
    ws["B2"] = 3.5
    wb.save(tmp_path / "test.xlsx")

    from memo_chef.chart_extraction import extract_workbook_tables
    result = extract_workbook_tables(str(tmp_path / "test.xlsx"))
    assert "TAB: Rent Growth" in result
    assert "Row 1:" in result
    assert "Year" in result


def test_extract_workbook_tables_specific_tabs(tmp_path):
    """Only extract specified tabs."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Tab A"
    ws1["A1"] = "data"
    ws2 = wb.create_sheet("Tab B")
    ws2["A1"] = "other"
    wb.save(tmp_path / "test.xlsx")

    from memo_chef.chart_extraction import extract_workbook_tables
    result = extract_workbook_tables(str(tmp_path / "test.xlsx"), tab_names=["Tab A"])
    assert "TAB: Tab A" in result
    assert "TAB: Tab B" not in result


def test_extract_workbook_tables_empty_tabs(tmp_path):
    """Empty tabs are skipped."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Empty"
    wb.save(tmp_path / "test.xlsx")

    from memo_chef.chart_extraction import extract_workbook_tables
    result = extract_workbook_tables(str(tmp_path / "test.xlsx"))
    assert "TAB:" not in result or result.strip() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chart_extraction.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `extract_workbook_tables()`**

Create `memo_chef/chart_extraction.py`:

```python
"""Market workbook extraction and chart mapping."""
from __future__ import annotations

import logging
from typing import Any

import openpyxl

log = logging.getLogger(__name__)


def extract_workbook_tables(
    workbook_path: str,
    tab_names: list[str] | None = None,
) -> str:
    """Extract tabular data from all (or specified) tabs as text.

    Uses the same format as extract_proforma_data() for consistency.
    """
    wb = openpyxl.load_workbook(workbook_path, data_only=True)
    lines: list[str] = []

    sheets = tab_names if tab_names else wb.sheetnames
    for tab_name in sheets:
        if tab_name not in wb.sheetnames:
            log.warning("Tab '%s' not found in workbook — skipping", tab_name)
            continue
        ws = wb[tab_name]
        tab_lines: list[str] = []
        for row in ws.iter_rows(values_only=False):
            row_data = [str(cell.value) for cell in row if cell.value is not None]
            if row_data:
                tab_lines.append(f"Row {row[0].row}:\t" + "\t".join(row_data))
        if tab_lines:
            lines.append(f"\n{'=' * 70}")
            lines.append(f"TAB: {tab_name}")
            lines.append(f"{'=' * 70}")
            lines.extend(tab_lines)

    wb.close()
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_chart_extraction.py -v`
Expected: All PASSED

- [ ] **Step 5: Create prompt template**

Create `prompts/chart_mapping_v1.txt`:

```
You are a financial data analyst. The user has provided a market data workbook
and instructions for updating charts in an IC memo.

Your task: match the workbook data to existing memo charts and return structured
updates so the charts can be programmatically updated.

## User Instructions
{user_instructions}

## Workbook Data (extracted from Excel tabs)
{workbook_data}

## Existing Memo Charts
{memo_charts_json}

Return a JSON array of chart updates. Each update must have:
- "page": integer (1-based slide number)
- "chart_name": string (the shape name or chart title in the memo)
- "series_name": string (which data series to update)
- "new_values": array of numbers
- "new_categories": array of strings OR null (if categories don't change)
- "source": string (which workbook tab and rows the data came from)

IMPORTANT: Only include updates where you can confidently match workbook data
to a specific memo chart. If unsure, omit rather than guess.

Return ONLY the JSON array. No commentary.
```

- [ ] **Step 6: Commit**

```bash
git add memo_chef/chart_extraction.py prompts/chart_mapping_v1.txt tests/test_chart_extraction.py
git commit -m "feat(F8): add workbook extraction and chart mapping prompt"
```

---

### Task 13: Implement Claude-based chart mapping and extend `_apply_chart_updates()`

**Files:**
- Modify: `memo_chef/chart_extraction.py`
- Modify: `memo_automator.py:2184-2309` (`_apply_chart_updates()`)
- Modify: `memo_chef/models.py`

- [ ] **Step 1: Add `map_market_charts()` to chart_extraction.py**

```python
def map_market_charts(
    workbook_text: str,
    memo_charts: list[dict],
    user_instructions: str,
    client: Any,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 4096,
) -> list[dict]:
    """Use Claude to map workbook data to memo charts."""
    import json
    from pathlib import Path

    prompt_path = Path(__file__).parent.parent / "prompts" / "chart_mapping_v1.txt"
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.replace("{user_instructions}", user_instructions)
    prompt = prompt.replace("{workbook_data}", workbook_text[:50_000])
    prompt = prompt.replace("{memo_charts_json}", json.dumps(memo_charts, indent=2))

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()

    import re
    json_match = re.search(r"\[[\s\S]*\]", text)
    if not json_match:
        log.warning("No JSON array in chart mapping response")
        return []

    return json.loads(json_match.group())
```

- [ ] **Step 2: Add `RunRequest` fields for chart updating**

In `memo_chef/models.py`, add to `RunRequest`:

```python
    market_workbook_path: str | None = None
    chart_instructions: str | None = None
```

- [ ] **Step 3: Extend `_apply_chart_updates()` for category labels**

In `memo_automator.py:2204`, replace the `new_categories` line:

```python
        new_categories = upd.get("categories", None)  # noqa: F841 – reserved for future use
```

with:

```python
        new_categories = upd.get("new_categories") or upd.get("categories")
```

Then after the series value update block (after `break` at line 2297, still inside the `for upd in chart_updates:` loop), add category label update logic. Note: `_nsmap` is defined at line 2271, `target_chart` holds the chart object:

```python
        # Update category labels if provided
        if new_categories and not dry_run:
            try:
                # Access the chart's plot XML to find category cache
                plot_el = target_chart.plots[0]._element
                cat_el = plot_el.find(".//c:cat/c:strRef/c:strCache", _nsmap)
                if cat_el is None:
                    cat_el = plot_el.find(".//c:cat/c:strLit", _nsmap)
                if cat_el is not None:
                    pts = cat_el.findall("c:pt", _nsmap)
                    for i, pt in enumerate(pts):
                        if i < len(new_categories):
                            v_elem = pt.find("c:v", _nsmap)
                            if v_elem is not None:
                                v_elem.text = str(new_categories[i])
                    log.info("Updated %d category labels on page %d", min(len(pts), len(new_categories)), page)
            except Exception as e:
                log.warning("Failed to update category labels on page %d: %s", page, e)
```

- [ ] **Step 4: Run full test suite**

Run: `pytest -x -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add memo_chef/chart_extraction.py memo_chef/models.py memo_automator.py
git commit -m "feat(F8): implement chart mapping and category label updates"
```

---

### Task 14: Add chart workbook UI inputs and pipeline integration

**Files:**
- Modify: `app.py` (run config section)
- Modify: `memo_chef/pipeline.py`

- [ ] **Step 1: Add UI inputs in run config**

In `render_new_run_tab()`, after the existing supplemental data inputs, add:

```python
    st.markdown("**Market Data (Charts)**")
    market_wb_file = st.file_uploader("Market workbook", type=["xlsx", "xlsm"], key="market_wb")
    chart_instructions = st.text_area(
        "Chart instructions",
        placeholder="e.g., Update the rent trend chart on slide 12 with submarket rents from the 'Rent Growth' tab.",
        key="chart_instructions",
    )
```

Pass through to `RunRequest` / job payload when creating the run.

- [ ] **Step 2: Add chart update step to pipeline**

In `memo_chef/pipeline.py`, after the main mapping/validation/apply cycle, add a chart update step that triggers when `req.market_workbook_path` and `req.chart_instructions` are set:

```python
    if req.market_workbook_path and req.chart_instructions:
        from memo_chef.chart_extraction import extract_workbook_tables, map_market_charts
        wb_text = extract_workbook_tables(req.market_workbook_path)
        # memo_charts extracted during memo extraction step
        chart_updates = map_market_charts(
            workbook_text=wb_text,
            memo_charts=memo_charts,
            user_instructions=req.chart_instructions,
            client=client,
        )
        if chart_updates:
            from memo_automator import _apply_chart_updates
            chart_changes = _apply_chart_updates(output_memo_path, chart_updates, dry_run=req.dry_run)
            all_changes.extend(chart_changes)
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -x -v`
Expected: All pass

- [ ] **Step 4: Commit and create PR**

```bash
git checkout -b feat/chart-data-updating
git add memo_chef/chart_extraction.py memo_chef/models.py memo_chef/pipeline.py memo_automator.py app.py prompts/chart_mapping_v1.txt tests/test_chart_extraction.py
git push -u origin feat/chart-data-updating
gh pr create --title "feat: chart data updating from market workbooks (F8)" --body "$(cat <<'EOF'
## Summary
- New `memo_chef/chart_extraction.py`: workbook table extraction + Claude chart mapping
- New `prompts/chart_mapping_v1.txt`: chart-to-memo mapping prompt
- Extended `_apply_chart_updates()` to handle category label updates
- Market workbook upload + chart instructions text box in Streamlit UI
- Pipeline integration: auto-update memo charts from market data

## Test plan
- [ ] Run `pytest tests/test_chart_extraction.py -v` — all tests pass
- [ ] Manual: upload a market workbook with chart instructions, verify memo charts are updated
- [ ] Verify category labels update correctly

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Chunk 5: F7 — Market Comp Slide Builder

### Task 14.5: Create feature branch

- [ ] **Step 1: Create branch from main**

```bash
git checkout main && git pull origin main
git checkout -b feat/comp-slide-builder
```

---

### Task 15: Add rapidfuzz dependency and comp data models

**Files:**
- Modify: `requirements.txt`
- Modify: `memo_chef/models.py`

- [ ] **Step 1: Install rapidfuzz**

```bash
pip install rapidfuzz
```

Add to `requirements.txt`:

```
rapidfuzz>=3.0
```

- [ ] **Step 2: Add comp models to `memo_chef/models.py`**

```python
class UnitMixEntry(BaseModel):
    unit_type: str
    beds: int | None = None
    baths: int | None = None
    sf: int | None = None
    rent: float | None = None
    rent_per_sf: float | None = None


class CompProperty(BaseModel):
    name: str
    address: str | None = None
    distance_mi: float | None = None
    unit_mix: list[UnitMixEntry] = Field(default_factory=list)
    total_units: int | None = None
    occupancy_pct: float | None = None
    year_built: int | None = None
    concessions: str | None = None
    source: str  # "url", "realpage", "csv", "manual"
    source_detail: str = ""


class CompSlideRequest(BaseModel):
    subject_property: CompProperty
    comps: list[CompProperty]
    sort_by: str = "distance"
    max_comps: int = 6
    include_narrative: bool = True
```

Add to `RunRequest`:

```python
    comp_csv_path: str | None = None
    comp_manual_entries: list[dict] | None = None
    auto_generate_comp_slide: bool = False
    comp_max_comps: int = 6
    comp_sort_by: str = "distance"
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt memo_chef/models.py
git commit -m "feat(F7): add comp data models and rapidfuzz dependency"
```

---

### Task 16: Implement comp normalization and deduplication

**Files:**
- Create: `memo_chef/comp_builder.py`
- Create: `tests/test_comp_builder.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_comp_builder.py`:

```python
"""Tests for comp slide builder."""
import pytest
from memo_chef.models import CompProperty, UnitMixEntry


def test_normalize_from_csv(tmp_path):
    """CSV with standard columns parses correctly."""
    import csv
    csv_path = tmp_path / "comps.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Address", "Units", "Occupancy", "Year Built"])
        writer.writerow(["The Reserve", "123 Main St", "250", "95.0", "2020"])
        writer.writerow(["Oak Park", "456 Oak Ave", "180", "92.5", "2018"])

    from memo_chef.comp_builder import normalize_comps_from_csv
    comps = normalize_comps_from_csv(str(csv_path))
    assert len(comps) == 2
    assert comps[0].name == "The Reserve"
    assert comps[0].total_units == 250
    assert comps[0].occupancy_pct == 95.0


def test_dedup_merges_duplicates():
    from memo_chef.comp_builder import deduplicate_comps
    c1 = CompProperty(name="The Reserve", total_units=250, source="csv")
    c2 = CompProperty(name="The Reserve at Oak Creek", occupancy_pct=95.0, source="realpage")
    result = deduplicate_comps([c1, c2])
    assert len(result) == 1
    # realpage wins on conflict, csv fills missing fields
    assert result[0].occupancy_pct == 95.0
    assert result[0].total_units == 250


def test_dedup_no_false_merges():
    from memo_chef.comp_builder import deduplicate_comps
    c1 = CompProperty(name="Oak Park Apartments", source="csv")
    c2 = CompProperty(name="The Reserve", source="csv")
    result = deduplicate_comps([c1, c2])
    assert len(result) == 2


def test_normalize_from_csv_flexible_columns(tmp_path):
    """Column names are matched case-insensitively."""
    import csv
    csv_path = tmp_path / "comps.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["property name", "total units", "year_built"])
        writer.writerow(["Test Prop", "100", "2022"])

    from memo_chef.comp_builder import normalize_comps_from_csv
    comps = normalize_comps_from_csv(str(csv_path))
    assert len(comps) == 1
    assert comps[0].name == "Test Prop"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_comp_builder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement normalization and dedup**

Create `memo_chef/comp_builder.py`:

```python
"""Comp slide builder — normalize, deduplicate, and generate comp slides."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from pptx import Presentation

from memo_chef.models import CompProperty, UnitMixEntry

log = logging.getLogger(__name__)

_SOURCE_PRIORITY = {"realpage": 0, "csv": 1, "url": 2, "manual": 3}

# Column name mapping: normalized key -> CompProperty field
_COLUMN_MAP = {
    "name": "name", "property name": "name", "property": "name",
    "address": "address",
    "units": "total_units", "total units": "total_units", "unit count": "total_units",
    "occupancy": "occupancy_pct", "occ": "occupancy_pct", "occupancy %": "occupancy_pct",
    "year built": "year_built", "year_built": "year_built", "vintage": "year_built",
    "distance": "distance_mi", "distance (mi)": "distance_mi",
    "concessions": "concessions",
}


def normalize_comps_from_csv(csv_path: str) -> list[CompProperty]:
    """Parse a CSV file into CompProperty objects."""
    path = Path(csv_path)
    comps: list[CompProperty] = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data: dict[str, Any] = {"source": "csv", "source_detail": path.name}
            for col, val in row.items():
                key = col.strip().lower()
                field = _COLUMN_MAP.get(key)
                if field and val.strip():
                    v = val.strip()
                    if field in ("total_units", "year_built"):
                        try:
                            data[field] = int(float(v))
                        except ValueError:
                            pass
                    elif field in ("occupancy_pct", "distance_mi"):
                        try:
                            data[field] = float(v)
                        except ValueError:
                            pass
                    else:
                        data[field] = v
            if "name" in data:
                comps.append(CompProperty(**data))

    return comps


def normalize_comps_from_urls(
    comp_urls: list,
    extracted_texts: dict,
) -> list[CompProperty]:
    """Convert scraped comp URL data into CompProperty objects.

    Uses existing extraction.py output. Extracts property name and any
    numeric fields found in the scraped text.
    """
    comps: list[CompProperty] = []
    for comp_url in comp_urls:
        url = comp_url.url if hasattr(comp_url, "url") else str(comp_url)
        label = comp_url.label if hasattr(comp_url, "label") else ""
        text = extracted_texts.get(url, "")
        if label or text:
            comps.append(CompProperty(
                name=label or url,
                source="url",
                source_detail=url,
            ))
    return comps


def deduplicate_comps(comps: list[CompProperty]) -> list[CompProperty]:
    """Fuzzy-match by name, merge fields with source priority."""
    from rapidfuzz import fuzz

    if not comps:
        return []

    # Sort by source priority (highest priority first)
    sorted_comps = sorted(comps, key=lambda c: _SOURCE_PRIORITY.get(c.source, 99))
    merged: list[CompProperty] = []

    for comp in sorted_comps:
        matched = False
        for i, existing in enumerate(merged):
            score = fuzz.token_sort_ratio(comp.name, existing.name)
            if score > 85:
                # Merge: existing has priority, comp fills gaps
                merged_data = existing.model_dump()
                new_data = comp.model_dump()
                for field, val in new_data.items():
                    if field in ("source", "source_detail"):
                        continue
                    if val is not None and merged_data.get(field) is None:
                        merged_data[field] = val
                merged[i] = CompProperty(**merged_data)
                matched = True
                break
        if not matched:
            merged.append(comp)

    return merged
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_comp_builder.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add memo_chef/comp_builder.py tests/test_comp_builder.py
git commit -m "feat(F7): implement comp normalization and deduplication"
```

---

### Task 17: Implement comp slide generation

**Files:**
- Modify: `memo_chef/comp_builder.py`
- Modify: `tests/test_comp_builder.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_comp_builder.py`:

```python
def test_build_comp_slide_from_scratch():
    """Build a comp slide without a template."""
    from pptx import Presentation
    from memo_chef.comp_builder import build_comp_slide

    prs = Presentation()
    # Add a blank slide so we have something
    prs.slides.add_slide(prs.slide_layouts[0])

    subject = CompProperty(
        name="The Reserve", total_units=250, occupancy_pct=95.0,
        year_built=2020, source="manual",
    )
    comps = [
        CompProperty(name="Oak Park", total_units=180, occupancy_pct=92.5,
                     year_built=2018, source="csv"),
        CompProperty(name="Maple Grove", total_units=200, occupancy_pct=93.0,
                     year_built=2019, source="csv"),
    ]
    sections = [{"name": "Competitive Landscape", "start_page": 1, "end_page": 1}]

    build_comp_slide(prs, subject, comps, sections)
    # Should have added a new slide
    assert len(prs.slides) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_comp_builder.py::test_build_comp_slide_from_scratch -v`
Expected: FAIL

- [ ] **Step 3: Implement `build_comp_slide()`**

Add to `memo_chef/comp_builder.py`:

```python
from memo_chef.slide_insertion import (
    clone_slide,
    find_template_slide,
    insert_slide_at_position,
    detect_memo_sections,
)
from pptx.util import Inches, Pt


def build_comp_slide(
    prs: Presentation,
    subject: CompProperty,
    comps: list[CompProperty],
    memo_sections: list[dict],
    narrative: str | None = None,
) -> None:
    """Clone existing comp slide or build from scratch, populate with comp data."""
    # Try to find and clone existing comp slide
    template_idx = find_template_slide(prs, "Comp", "table", memo_sections)

    if template_idx is not None:
        new_slide = clone_slide(prs, template_idx)
        _populate_comp_table(new_slide, subject, comps)
        target_section = None
        for s in memo_sections:
            if "comp" in s["name"].lower() or "competitive" in s["name"].lower():
                target_section = s
                break
        if target_section:
            insert_slide_at_position(prs, new_slide, target_section["end_page"] - 1)
    else:
        _build_comp_slide_from_scratch(prs, subject, comps, memo_sections, narrative)


def _populate_comp_table(slide, subject: CompProperty, comps: list[CompProperty]) -> None:
    """Repopulate an existing cloned comp slide's table with new data."""
    for shape in slide.shapes:
        if shape.has_table:
            table = shape.table
            all_props = [subject] + comps
            # Populate columns: each property gets a column
            for col_idx, prop in enumerate(all_props):
                if col_idx + 1 >= len(table.columns):
                    break
                _set_cell_safe(table, 0, col_idx + 1, prop.name)
                row_data = [
                    str(prop.total_units or ""),
                    f"{prop.occupancy_pct:.1f}%" if prop.occupancy_pct else "",
                    str(prop.year_built or ""),
                    f"{prop.distance_mi:.1f} mi" if prop.distance_mi else "",
                    prop.concessions or "",
                ]
                for row_idx, val in enumerate(row_data):
                    if row_idx + 1 < len(table.rows):
                        _set_cell_safe(table, row_idx + 1, col_idx + 1, val)
            break


def _set_cell_safe(table, row: int, col: int, text: str) -> None:
    """Set table cell text, preserving formatting."""
    try:
        cell = table.cell(row, col)
        if cell.text_frame.paragraphs:
            cell.text_frame.paragraphs[0].text = text
        else:
            cell.text = text
    except (IndexError, AttributeError):
        pass


def _build_comp_slide_from_scratch(
    prs: Presentation,
    subject: CompProperty,
    comps: list[CompProperty],
    memo_sections: list[dict],
    narrative: str | None = None,
) -> None:
    """Build a comp slide from scratch when no template is available."""
    layout = prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[0]
    slide = prs.slides.add_slide(layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(9), Inches(0.6))
    tf = title_box.text_frame
    tf.text = "Rent Comparison"
    for para in tf.paragraphs:
        para.font.size = Pt(24)
        para.font.bold = True

    all_props = [subject] + comps[:6]
    row_labels = ["Property", "Units", "Occupancy", "Year Built", "Distance", "Concessions"]
    rows = len(row_labels)
    cols = len(all_props) + 1  # +1 for row labels

    table_shape = slide.shapes.add_table(
        rows, cols, Inches(0.3), Inches(1.0), Inches(9.4), Inches(3.5)
    )
    table = table_shape.table

    for i, label in enumerate(row_labels):
        table.cell(i, 0).text = label

    for col_idx, prop in enumerate(all_props):
        table.cell(0, col_idx + 1).text = prop.name
        table.cell(1, col_idx + 1).text = str(prop.total_units or "")
        table.cell(2, col_idx + 1).text = f"{prop.occupancy_pct:.1f}%" if prop.occupancy_pct else ""
        table.cell(3, col_idx + 1).text = str(prop.year_built or "")
        table.cell(4, col_idx + 1).text = f"{prop.distance_mi:.1f} mi" if prop.distance_mi else ""
        table.cell(5, col_idx + 1).text = prop.concessions or ""

    if narrative:
        text_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.0), Inches(9), Inches(1.5))
        tf = text_box.text_frame
        tf.word_wrap = True
        tf.text = narrative
        for para in tf.paragraphs:
            para.font.size = Pt(11)

    # Position after comp section
    for s in memo_sections:
        if "comp" in s["name"].lower() or "competitive" in s["name"].lower():
            insert_slide_at_position(prs, slide, s["end_page"] - 1)
            break
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_comp_builder.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add memo_chef/comp_builder.py tests/test_comp_builder.py
git commit -m "feat(F7): implement comp slide generation (clone + from-scratch)"
```

---

### Task 18: Add comp builder UI and pipeline integration

**Files:**
- Modify: `app.py` (run config section)
- Modify: `memo_chef/pipeline.py`

- [ ] **Step 1: Add comp builder UI in run config**

In `render_new_run_tab()`, after the market data workbook section, add:

```python
    st.markdown("**Comp Slide Builder**")
    auto_comp = st.checkbox("Auto-generate comp slide", value=False, key="auto_comp")
    comp_csv = None
    if auto_comp:
        comp_csv = st.file_uploader("Comp data (CSV)", type=["csv"], key="comp_csv")
```

Pass `auto_generate_comp_slide`, `comp_csv_path` through to `RunRequest`.

- [ ] **Step 2: Add comp builder step to pipeline**

In `memo_chef/pipeline.py`, after slide insertion and before final save:

```python
    if req.auto_generate_comp_slide:
        from memo_chef.comp_builder import (
            normalize_comps_from_csv,
            deduplicate_comps,
            build_comp_slide,
        )
        from memo_chef.slide_insertion import detect_memo_sections

        all_comps = []
        if req.comp_csv_path:
            all_comps.extend(normalize_comps_from_csv(req.comp_csv_path))
        # Add comp URL sources if available
        if req.comp_urls:
            from memo_chef.comp_builder import normalize_comps_from_urls
            all_comps.extend(normalize_comps_from_urls(req.comp_urls, comp_texts))

        if all_comps:
            deduped = deduplicate_comps(all_comps)
            sections = detect_memo_sections(memo_text)
            subject = deduped[0]  # First comp treated as subject if no explicit subject
            build_comp_slide(prs, subject, deduped[1:], sections)
```

- [ ] **Step 3: Run full test suite**

Run: `pytest -x -v`
Expected: All pass

- [ ] **Step 4: Commit and create PR**

```bash
git checkout -b feat/comp-slide-builder
git add memo_chef/comp_builder.py memo_chef/models.py memo_chef/pipeline.py app.py tests/test_comp_builder.py requirements.txt
git push -u origin feat/comp-slide-builder
gh pr create --title "feat: market comp slide builder (F7)" --body "$(cat <<'EOF'
## Summary
- New `memo_chef/comp_builder.py`: comp normalization (CSV/URL/RealPage/manual), fuzzy dedup (rapidfuzz), slide generation
- Comp data models: `CompProperty`, `UnitMixEntry`, `CompSlideRequest` in models.py
- Template cloning: reuses existing comp slide format when available
- From-scratch fallback: generates branded comp table slide
- Streamlit UI: auto-generate toggle + CSV upload
- New dep: `rapidfuzz>=3.0`

## Test plan
- [ ] Run `pytest tests/test_comp_builder.py -v` — all tests pass
- [ ] Manual: upload CSV with comp data, check generated slide matches existing format
- [ ] Verify dedup merges duplicate properties correctly
- [ ] Run `pytest -x` — full suite passes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Implementation Summary

| PR | Feature | Branch | Key Files |
|----|---------|--------|-----------|
| #16 | F10: Proforma Drift Detection | `feat/proforma-drift-detection` | `memo_chef/drift.py`, `app_services.py`, `app.py`, `memo_automator.py` |
| #17 | F6: Redline Diff Overlay | `feat/redline-diff-overlay` | `memo_chef/redline.py`, `app.py` |
| #18 | F9: Run Analytics Dashboard | `feat/run-analytics-dashboard` | `app_services.py`, `app.py` |
| #19 | F8: Chart Data Updating | `feat/chart-data-updating` | `memo_chef/chart_extraction.py`, `memo_automator.py`, `memo_chef/pipeline.py` |
| #20 | F7: Comp Slide Builder | `feat/comp-slide-builder` | `memo_chef/comp_builder.py`, `memo_chef/models.py`, `memo_chef/pipeline.py` |

Each PR is independent and can be merged in any order, though the recommended order (F10 → F6 → F9 → F8 → F7) minimizes risk and maximizes early value.
