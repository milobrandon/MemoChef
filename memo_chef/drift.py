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

        changed_rows = len({c["row"] for c in changed})
        tab_total = len(added) + changed_rows + len(removed)
        if tab_total > 0:
            tabs_with_changes += 1
        total_changes += tab_total
        by_tab[tab] = {"added": added, "changed": changed, "removed": removed}

    return {
        "total_changes": total_changes,
        "by_tab": by_tab,
        "summary": (
            f"{total_changes} values changed across "
            f"{tabs_with_changes} tab{'s' if tabs_with_changes != 1 else ''}"
        ),
    }
