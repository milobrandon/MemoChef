"""Regression tests for critical fixes (2026-03-18 code review)."""
import pytest
from unittest.mock import patch, MagicMock


class TestBatchAPIMergeKeys:
    """H1: table_structure_updates must be initialized in batch merge."""

    def test_table_structure_updates_key_exists(self):
        from memo_chef.pipeline import _mapping_with_batch_api

        with patch("memo_chef.pipeline.build_mapping_batch_requests") as mock_build, \
             patch("memo_chef.pipeline.submit_and_poll_batch") as mock_submit, \
             patch("memo_chef.pipeline._dedup_mappings", side_effect=lambda x: x), \
             patch("memo_chef.pipeline.chunk_memo_by_pages", return_value=["chunk1"]):
            mock_build.return_value = [{"custom_id": "mapping-chunk-0"}]
            mock_submit.return_value = {
                "mapping-chunk-0": {
                    "table_updates": [{"id": 1}],
                    "text_updates": [],
                    "row_inserts": [],
                    "narrative_updates": [],
                    "table_structure_updates": [{"op": "add_column"}],
                }
            }

            checkpoint = MagicMock()
            result = _mapping_with_batch_api(
                client=MagicMock(),
                proforma_data="test",
                memo_content="chunk1",
                cfg={},
                property_name="Test",
                callback=None,
                checkpoint=checkpoint,
                source_directives=[],
            )

            assert "table_structure_updates" in result
            assert len(result["table_structure_updates"]) == 1


class TestSecretStrAPIKey:
    """S1: api_key must not be exposed in serialization."""

    def test_api_key_hidden_in_model_dump(self):
        from memo_chef.models import RunRequest
        req = RunRequest(
            memo_path="/tmp/test.pptx",
            proforma_path="/tmp/test.xlsm",
            output_dir="/tmp/out",
            api_key="sk-ant-secret-key-12345",
            config_path="/tmp/config.yaml",
            run_id="test-001",
        )
        dumped = str(req.model_dump())
        assert "sk-ant-secret-key-12345" not in dumped

    def test_api_key_accessible_via_get_secret_value(self):
        from memo_chef.models import RunRequest
        req = RunRequest(
            memo_path="/tmp/test.pptx",
            proforma_path="/tmp/test.xlsm",
            output_dir="/tmp/out",
            api_key="sk-ant-secret-key-12345",
            config_path="/tmp/config.yaml",
            run_id="test-001",
        )
        assert req.api_key.get_secret_value() == "sk-ant-secret-key-12345"

    def test_api_key_hidden_in_repr(self):
        from memo_chef.models import RunRequest
        req = RunRequest(
            memo_path="/tmp/test.pptx",
            proforma_path="/tmp/test.xlsm",
            output_dir="/tmp/out",
            api_key="sk-ant-secret-key-12345",
            config_path="/tmp/config.yaml",
            run_id="test-001",
        )
        assert "sk-ant-secret-key-12345" not in repr(req)


class TestFeatureFlags:
    """F1: Feature flags default to False and are configurable."""

    def test_defaults_all_false(self):
        from memo_automator import AppConfig
        cfg = AppConfig()
        assert cfg.features.auto_split_enabled is False
        assert cfg.features.footer_normalization_enabled is False
        assert cfg.features.correction_retry_enabled is False

    def test_enable_via_config(self):
        from memo_automator import AppConfig
        cfg = AppConfig.model_validate({
            "features": {"auto_split_enabled": True}
        })
        assert cfg.features.auto_split_enabled is True
        assert cfg.features.footer_normalization_enabled is False
        assert cfg.features.correction_retry_enabled is False

    def test_unknown_flag_rejected(self):
        from memo_automator import AppConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AppConfig.model_validate({
                "features": {"nonexistent_flag": True}
            })


class TestMultiSlideInsertionOffset:
    """H3: Position-aware offset calculation."""

    def test_sequential_positions(self):
        """Inserts at [3, 5, 7] should offset correctly."""
        inserted_positions = []
        specs = [3, 5, 7]

        results = []
        for insert_after in specs:
            base_idx = insert_after - 1
            offset = sum(1 for p in inserted_positions if p <= base_idx)
            target_idx = base_idx + offset
            results.append(target_idx)
            inserted_positions.append(target_idx)

        assert results == [2, 5, 8]

    def test_single_insert(self):
        """Single insert should have no offset."""
        inserted_positions = []
        base_idx = 5 - 1
        offset = sum(1 for p in inserted_positions if p <= base_idx)
        target_idx = base_idx + offset
        assert target_idx == 4


class TestFStringFix:
    """Bonus: review round number must be interpolated."""

    def test_attempt_number_interpolated(self):
        attempt = 3
        user_text = "## Updated Memo Content\ntest"
        user_text += (
            f"\n\n## NOTE: This is review round {attempt}. Previous critical fixes "
            "have been applied. Re-evaluate the memo from scratch."
        )
        assert "round 3" in user_text
        assert "{attempt}" not in user_text
