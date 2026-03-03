"""Unit tests for config loading and validation."""
import os

import pytest

from memo_automator import load_config, _validate_config


class TestLoadConfig:
    def test_loads_valid_config(self, sample_config):
        cfg = load_config(sample_config)
        assert cfg["proforma"]["tabs"] == ["Executive Summary", "Cash Flow"]
        assert cfg["claude"]["model"] == "claude-sonnet-4-6"
        assert cfg["claude"]["temperature"] == 0

    def test_applies_defaults(self, tmp_dir):
        """An empty YAML file should get all defaults applied."""
        path = os.path.join(tmp_dir, "empty.yaml")
        with open(path, "w") as f:
            f.write("# empty config\n")
        cfg = load_config(path)
        assert isinstance(cfg["proforma"]["tabs"], list)
        assert len(cfg["proforma"]["tabs"]) > 0
        assert cfg["proforma"]["max_rows_per_tab"] == 250
        assert cfg["memo"]["pages"] == "all"
        assert cfg["claude"]["model"] == "claude-sonnet-4-6"

    def test_validation_model_defaults_to_model(self, tmp_dir):
        path = os.path.join(tmp_dir, "model.yaml")
        with open(path, "w") as f:
            f.write("claude:\n  model: claude-opus-4-6\n")
        cfg = load_config(path)
        assert cfg["claude"]["validation_model"] == "claude-opus-4-6"


class TestValidateConfig:
    def test_valid_config_returns_no_errors(self):
        cfg = {
            "proforma": {"tabs": ["Sheet1"], "max_rows_per_tab": 100, "max_cols_per_tab": 20},
            "schedule": {"max_tasks": 500},
            "claude": {"model": "claude-sonnet-4-6", "temperature": 0},
        }
        assert _validate_config(cfg) == []

    def test_empty_tabs(self):
        cfg = {"proforma": {"tabs": []}, "claude": {"model": "x", "temperature": 0}, "schedule": {}}
        errors = _validate_config(cfg)
        assert any("tabs" in e for e in errors)

    def test_non_string_tabs(self):
        cfg = {"proforma": {"tabs": [1, 2]}, "claude": {"model": "x", "temperature": 0}, "schedule": {}}
        errors = _validate_config(cfg)
        assert any("strings" in e for e in errors)

    def test_negative_max_rows(self):
        cfg = {
            "proforma": {"tabs": ["A"], "max_rows_per_tab": -1, "max_cols_per_tab": 10},
            "claude": {"model": "x", "temperature": 0},
            "schedule": {},
        }
        errors = _validate_config(cfg)
        assert any("max_rows_per_tab" in e for e in errors)

    def test_empty_model(self):
        cfg = {"proforma": {"tabs": ["A"]}, "claude": {"model": "", "temperature": 0}, "schedule": {}}
        errors = _validate_config(cfg)
        assert any("model" in e for e in errors)

    def test_temperature_out_of_range(self):
        cfg = {"proforma": {"tabs": ["A"]}, "claude": {"model": "x", "temperature": 2.0}, "schedule": {}}
        errors = _validate_config(cfg)
        assert any("temperature" in e for e in errors)

    def test_load_config_raises_on_invalid(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.yaml")
        with open(path, "w") as f:
            f.write("proforma:\n  tabs: []\n")
        with pytest.raises(ValueError, match="Invalid config"):
            load_config(path)
