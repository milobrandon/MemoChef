"""Unit tests for Streamlit auth/credit helper logic."""

from app_helpers import (
    count_changes_from_log,
    fire_button_disabled_reason,
    hash_password,
    should_disable_fire_button,
    verify_password,
)


def test_verify_password_round_trip():
    stored = hash_password("s3cr3t!")
    assert verify_password("s3cr3t!", stored)
    assert not verify_password("wrong", stored)


def test_verify_password_rejects_malformed_hash():
    assert not verify_password("pw", "")
    assert not verify_password("pw", "not-a-hash")
    assert not verify_password("pw", "zzzz:nothex")
    assert not verify_password("pw", "abcd")


def test_fire_button_disabled_on_credits_failure():
    assert should_disable_fire_button(object(), object(), 5, "db unavailable")


def test_fire_button_enabled_with_valid_inputs():
    assert not should_disable_fire_button(object(), object(), 3, None)


def test_fire_button_not_disabled_for_missing_inputs():
    assert not should_disable_fire_button(None, object(), 3, None)
    assert not should_disable_fire_button(object(), None, 3, None)


def test_fire_button_disabled_for_credits_only():
    assert should_disable_fire_button(object(), object(), 0, None)


def test_fire_button_reason_explains_missing_inputs():
    assert fire_button_disabled_reason(None, None, 3, None) == (
        "Upload a memo deck before starting a run."
    )
    assert fire_button_disabled_reason(object(), None, 3, None) == (
        "Upload a proforma, or set meeting lookback > 0 for a narrative-only run."
    )


def test_fire_button_allows_narrative_only_mode():
    assert fire_button_disabled_reason(
        object(), None, 3, None, meeting_lookback_days=30
    ) is None


def test_fire_button_still_requires_memo_in_narrative_only_mode():
    assert fire_button_disabled_reason(
        None, None, 3, None, meeting_lookback_days=30
    ) == "Upload a memo deck before starting a run."


def test_count_changes_full_run_format():
    log = "# Changelog\n\n**Total changes applied:** 310\n\n## Summary..."
    assert count_changes_from_log(log) == 310


def test_count_changes_narrative_format():
    log = "# Changelog\n\n## Changes Applied\n\n**Total changes: 3**\n\n### Change 1\n..."
    assert count_changes_from_log(log) == 3


def test_count_changes_falls_back_to_change_headings():
    log = "### Change 1\nfoo\n### Change 2\nbar\n### Change 3\nbaz"
    assert count_changes_from_log(log) == 3


def test_count_changes_empty_log():
    assert count_changes_from_log("") == 0
    assert count_changes_from_log(None) == 0  # type: ignore[arg-type]
