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


class _FakeUsage:
    input_tokens = 100
    output_tokens = 50


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = "end_turn"
        self.usage = _FakeUsage()


class _FakeStream:
    """Context manager that mimics client.messages.stream()."""

    def __init__(self, message: "_FakeMessage"):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_final_message(self):
        return self._message


class _FakeMessagesAPI:
    def __init__(self, mapping_payload: dict, validation_payload: dict):
        self.mapping_payload = mapping_payload
        self.validation_payload = validation_payload

    def _resolve(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "## Proposed Changes" in prompt:
            return _FakeMessage(json.dumps(self.validation_payload))
        return _FakeMessage(json.dumps(self.mapping_payload))

    def create(self, **kwargs):
        return self._resolve(**kwargs)

    def stream(self, **kwargs):
        return _FakeStream(self._resolve(**kwargs))


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


class TestValidateMappingsCorrections:
    """Tests for validate_mappings correction-application logic (lines ~1711-1746).

    These tests mock _call_validation_api to avoid real API calls and focus
    on the correction/rejection reconstruction logic.
    """

    @staticmethod
    def _make_mappings():
        return {
            "table_updates": [
                {"page": 1, "table_name": "T1", "row_label": "A",
                 "column_index": 1, "old_value": "100", "new_value": "200",
                 "source": "S1"},
                {"page": 1, "table_name": "T1", "row_label": "B",
                 "column_index": 1, "old_value": "300", "new_value": "400",
                 "source": "S2"},
            ],
            "text_updates": [
                {"page": 1, "old_text": "old narrative", "new_text": "new narrative",
                 "source": "S3"},
            ],
            "row_inserts": [],
        }

    def test_correction_replaces_entry_at_correct_index(self):
        """A correction at idx=0 replaces the first table_update."""
        from unittest.mock import patch

        mappings = self._make_mappings()
        corrected_entry = {
            "page": 1, "table_name": "T1", "row_label": "A",
            "column_index": 1, "old_value": "100", "new_value": "250",
            "source": "S1",
        }
        validation_result = {
            "rejected": [],
            "corrections": [
                {"idx": 0, "type": "table", "corrected_entry": corrected_entry,
                 "reason": "wrong value"},
            ],
            "missed": [],
        }

        with patch("memo_automator._call_validation_api", return_value=validation_result):
            result = validate_mappings(
                client=None, mappings=mappings, proforma_data="data",
                memo_content="content", cfg=_cfg(),
            )

        assert result["table_updates"][0]["new_value"] == "250"
        assert result["table_updates"][1]["new_value"] == "400"  # unchanged

    def test_rejection_removes_entry(self):
        """A rejection at idx=1 removes the second table_update."""
        from unittest.mock import patch

        mappings = self._make_mappings()
        validation_result = {
            "rejected": [
                {"idx": 1, "type": "table", "reason": "old_value not found"},
            ],
            "corrections": [],
            "missed": [],
        }

        with patch("memo_automator._call_validation_api", return_value=validation_result):
            result = validate_mappings(
                client=None, mappings=mappings, proforma_data="data",
                memo_content="content", cfg=_cfg(),
            )

        assert len(result["table_updates"]) == 1
        assert result["table_updates"][0]["new_value"] == "200"  # first kept
        assert len(result["text_updates"]) == 1  # text untouched
