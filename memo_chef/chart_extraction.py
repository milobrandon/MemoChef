"""Market workbook extraction and chart mapping."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
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


def map_market_charts(
    workbook_text: str,
    memo_charts: list[dict],
    user_instructions: str,
    client: Any,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 4096,
) -> list[dict]:
    """Use Claude to map workbook data to memo charts."""
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

    json_match = re.search(r"\[[\s\S]*\]", text)
    if not json_match:
        log.warning("No JSON array in chart mapping response")
        return []

    return json.loads(json_match.group())
