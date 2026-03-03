"""Integration test for full pipeline with mocked Claude responses."""
import json
import os

from memo_automator import (
    apply_updates,
    create_backup,
    extract_memo_content,
    extract_proforma_data,
    get_metric_mappings,
    pre_validate_mappings,
    validate_mappings,
    write_change_log,
)


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = "end_turn"


class _FakeMessagesAPI:
    def __init__(self, mapping_payload: dict, validation_payload: dict):
        self.mapping_payload = mapping_payload
        self.validation_payload = validation_payload

    def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "## Proposed Changes" in prompt:
            return _FakeMessage(json.dumps(self.validation_payload))
        return _FakeMessage(json.dumps(self.mapping_payload))


class _FakeClient:
    def __init__(self, mapping_payload: dict, validation_payload: dict):
        self.messages = _FakeMessagesAPI(mapping_payload, validation_payload)


def _cfg():
    return {
        "proforma": {
            "tabs": ["Executive Summary", "Cash Flow"],
            "max_rows_per_tab": 250,
            "max_cols_per_tab": 30,
        },
        "memo": {"pages": "all"},
        "claude": {
            "model": "claude-sonnet-4-6",
            "validation_model": "claude-sonnet-4-6",
            "max_tokens": 16000,
            "temperature": 0,
        },
    }


def test_full_pipeline_with_mocked_claude(sample_pptx, sample_proforma_xlsx, tmp_dir):
    mapping_payload = {
        "table_updates": [
            {
                "page": 1,
                "table_name": "UnitMixTable",
                "row_label": "1BR",
                "column_index": 1,
                "old_value": "120",
                "new_value": "130",
                "source": "Executive Summary B3",
            }
        ],
        "text_updates": [
            {
                "page": 1,
                "old_text": "IRR is 5.0%",
                "new_text": "IRR is 6.5%",
                "source": "Executive Summary B2",
            }
        ],
        "row_inserts": [],
    }
    validation_payload = {"rejected": [], "corrections": [], "missed": []}
    client = _FakeClient(mapping_payload, validation_payload)
    cfg = _cfg()

    backup_path = create_backup(sample_pptx, tmp_dir)
    assert os.path.exists(backup_path)

    proforma_data = extract_proforma_data(sample_proforma_xlsx, cfg)
    memo_content = extract_memo_content(sample_pptx, cfg)

    mappings = get_metric_mappings(client, proforma_data, memo_content, cfg)
    mappings = pre_validate_mappings(mappings, memo_content)
    validated = validate_mappings(client, mappings, proforma_data, memo_content, cfg)

    changes = apply_updates(sample_pptx, validated, dry_run=False)
    assert len(changes) == 2

    log_path = write_change_log(
        tmp_dir, changes, validated, sample_pptx, sample_proforma_xlsx, backup_path
    )
    assert os.path.exists(log_path)
