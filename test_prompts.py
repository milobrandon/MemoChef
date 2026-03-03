"""Tests for prompt template externalization and loading behavior."""

from memo_automator import (
    MAPPING_PROMPT,
    PROMPTS_DIR,
    VALIDATION_PROMPT,
    _load_prompt_template,
)


def test_prompt_files_exist():
    assert (PROMPTS_DIR / "mapping_v1.txt").exists()
    assert (PROMPTS_DIR / "validation_v1.txt").exists()


def test_prompt_files_include_placeholders():
    mapping = (PROMPTS_DIR / "mapping_v1.txt").read_text(encoding="utf-8")
    validation = (PROMPTS_DIR / "validation_v1.txt").read_text(encoding="utf-8")
    assert "{proforma_data}" in mapping
    assert "{memo_content}" in mapping
    assert "{property_name_section}" in mapping
    assert "{mappings_json}" in validation
    assert "{property_name_section}" in validation


def test_runtime_prompts_loaded_from_templates():
    mapping_file = (PROMPTS_DIR / "mapping_v1.txt").read_text(encoding="utf-8").strip()
    validation_file = (PROMPTS_DIR / "validation_v1.txt").read_text(encoding="utf-8").strip()
    assert MAPPING_PROMPT == mapping_file
    assert VALIDATION_PROMPT == validation_file


def test_prompt_loader_fallback():
    fallback = "fallback prompt body"
    loaded = _load_prompt_template("does_not_exist.txt", fallback)
    assert loaded == fallback
