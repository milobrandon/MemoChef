"""Testable helper logic shared by the Streamlit app."""

import hashlib
import secrets


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return ``salt_hex:hash_hex`` using PBKDF2-SHA256."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Validate password against stored ``salt_hex:hash_hex``."""
    try:
        salt_hex, hash_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        bytes.fromhex(hash_hex)
    except (AttributeError, ValueError):
        return False

    candidate = hash_password(password, salt)
    return secrets.compare_digest(candidate, stored_hash)


def should_disable_fire_button(
    memo_file: object | None,
    proforma_file: object | None,
    remaining_credits: int,
    credits_error: str | None,
) -> bool:
    """Return True when the run button should be disabled."""
    return (
        not (memo_file and proforma_file)
        or remaining_credits <= 0
        or credits_error is not None
    )
