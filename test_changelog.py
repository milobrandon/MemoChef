"""Tests for change log output formatting."""

from pathlib import Path

from memo_automator import write_change_log


def test_write_change_log_with_telemetry(tmp_dir):
    memo_path = str(Path(tmp_dir) / "memo.pptx")
    proforma_path = str(Path(tmp_dir) / "proforma.xlsx")
    backup_path = str(Path(tmp_dir) / "memo_BACKUP.pptx")

    # Create placeholder files for basename rendering.
    Path(memo_path).write_text("x", encoding="utf-8")
    Path(proforma_path).write_text("x", encoding="utf-8")
    Path(backup_path).write_text("x", encoding="utf-8")

    changes = [
        {
            "page": 1,
            "type": "table",
            "location": "Table A",
            "old": "100",
            "new": "110",
            "source": "Executive Summary B2",
        }
    ]
    mappings = {"table_updates": [], "text_updates": [], "row_inserts": []}
    metadata = {
        "run_duration_sec": 12.34,
        "mapping_api_calls": 2,
        "validation_api_calls": 1,
        "steps": {"backup": 0.1, "mapping": 7.2},
    }

    log_path = write_change_log(
        tmp_dir, changes, mappings, memo_path, proforma_path, backup_path, run_metadata=metadata
    )
    body = Path(log_path).read_text(encoding="utf-8")
    assert "## Run Telemetry" in body
    assert "Mapping API calls: 2" in body
    assert "Validation API calls: 1" in body
