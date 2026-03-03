"""Unit tests for Streamlit auth/credit helper logic."""

from app_helpers import hash_password, should_disable_fire_button, verify_password


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


def test_fire_button_disabled_for_missing_inputs_or_credits():
    assert should_disable_fire_button(None, object(), 3, None)
    assert should_disable_fire_button(object(), None, 3, None)
    assert should_disable_fire_button(object(), object(), 0, None)
