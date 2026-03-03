"""Unit tests for JSON response parsing and truncation salvage."""
from memo_automator import _parse_json_response, _salvage_truncated_json


def test_parse_json_response_direct():
    raw = '{"table_updates": [], "text_updates": [], "row_inserts": []}'
    parsed = _parse_json_response(raw)
    assert parsed is not None
    assert parsed["table_updates"] == []


def test_parse_json_response_markdown_fence():
    raw = "```json\n{\"table_updates\": [], \"text_updates\": [], \"row_inserts\": []}\n```"
    parsed = _parse_json_response(raw)
    assert parsed is not None
    assert "text_updates" in parsed


def test_parse_json_response_with_trailing_text():
    raw = '{"table_updates": [], "text_updates": [], "row_inserts": []}\nExtra commentary'
    parsed = _parse_json_response(raw)
    assert parsed is not None
    assert parsed["row_inserts"] == []


def test_parse_json_response_invalid_returns_none():
    assert _parse_json_response("not json at all") is None


def test_salvage_truncated_json_recovers_complete_entry():
    raw = '{"table_updates":[{"page":1,"old_value":"1","new_value":"2"}'
    salvaged = _salvage_truncated_json(raw)
    assert salvaged is not None
    assert len(salvaged["table_updates"]) == 1


def test_salvage_truncated_json_none_for_unsalvageable():
    raw = "plain text with no json object"
    assert _salvage_truncated_json(raw) is None
