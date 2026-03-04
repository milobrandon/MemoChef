"""Tests for user_management.py — password hashing, CRUD, invites, audit."""

import pytest

from user_management import (
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_returns_salt_colon_hash(self):
        result = hash_password("test123")
        assert ":" in result
        salt_hex, hash_hex = result.split(":", 1)
        assert len(salt_hex) == 32  # 16 bytes hex-encoded
        assert len(hash_hex) == 64  # SHA-256 = 32 bytes hex-encoded

    def test_verify_correct_password(self):
        stored = hash_password("mypassword")
        assert verify_password("mypassword", stored) is True

    def test_verify_wrong_password(self):
        stored = hash_password("mypassword")
        assert verify_password("wrongpassword", stored) is False

    def test_different_salts_produce_different_hashes(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        # Different salts → different outputs
        assert h1 != h2
        # But both verify correctly
        assert verify_password("same", h1) is True
        assert verify_password("same", h2) is True

    def test_explicit_salt(self):
        salt = b"\x00" * 16
        h1 = hash_password("test", salt=salt)
        h2 = hash_password("test", salt=salt)
        assert h1 == h2

    def test_empty_password(self):
        stored = hash_password("")
        assert verify_password("", stored) is True
        assert verify_password("notempty", stored) is False
