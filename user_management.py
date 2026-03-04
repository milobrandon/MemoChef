"""User management, invite system, and audit logging for Memo Chef.

All functions accept a psycopg2 connection as the first argument for
testability and decoupling from Streamlit.
"""

import hashlib
import logging
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing (mirrors app.py helpers — single source of truth now)
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return ``"salt_hex:hash_hex"`` using PBKDF2-SHA256 (260 000 iterations)."""
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return f"{salt.hex()}:{dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Check *password* against a stored ``salt_hex:hash_hex`` string."""
    salt_hex, _ = stored_hash.split(":", 1)
    salt = bytes.fromhex(salt_hex)
    return hash_password(password, salt) == stored_hash


# ---------------------------------------------------------------------------
# Schema creation (idempotent)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        TEXT UNIQUE NOT NULL,
    display_name    TEXT NOT NULL DEFAULT '',
    email           TEXT,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user',
    credits_per_week INTEGER NOT NULL DEFAULT 5,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS users_email_unique
    ON users (email) WHERE email IS NOT NULL;

CREATE TABLE IF NOT EXISTS invites (
    id              SERIAL PRIMARY KEY,
    email           TEXT NOT NULL,
    token           TEXT UNIQUE NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user',
    credits_per_week INTEGER NOT NULL DEFAULT 5,
    invited_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL,
    accepted_at     TIMESTAMPTZ,
    is_used         BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS user_audit_log (
    id              SERIAL PRIMARY KEY,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    target_user     TEXT,
    details         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def create_tables(conn) -> None:
    """Create ``users``, ``invites``, and ``user_audit_log`` tables if absent."""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    logger.info("User management tables ensured.")


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def _row_to_dict(cur, row) -> dict[str, Any] | None:
    if row is None:
        return None
    cols = [desc[0] for desc in cur.description]
    return dict(zip(cols, row))


def get_user(conn, username: str) -> dict[str, Any] | None:
    """Fetch a single user by username.  Returns ``None`` if not found."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        return _row_to_dict(cur, cur.fetchone())


def list_users(conn) -> list[dict[str, Any]]:
    """Return all users ordered by username."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users ORDER BY username")
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def create_user(
    conn,
    username: str,
    password: str,
    *,
    email: str | None = None,
    display_name: str = "",
    role: str = "user",
    credits_per_week: int = 5,
) -> dict[str, Any]:
    """Insert a new user.  Returns the created user dict."""
    pw_hash = hash_password(password)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (username, display_name, email, password_hash, role, credits_per_week)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (username, display_name, email, pw_hash, role, credits_per_week),
        )
        return _row_to_dict(cur, cur.fetchone())


def update_user(conn, username: str, **fields) -> dict[str, Any] | None:
    """Update one or more fields on an existing user.

    Allowed fields: ``display_name``, ``email``, ``role``,
    ``credits_per_week``, ``is_active``.

    If *password* is included it will be hashed automatically.
    """
    allowed = {"display_name", "email", "role", "credits_per_week", "is_active", "password"}
    invalid = set(fields) - allowed
    if invalid:
        raise ValueError(f"Cannot update fields: {invalid}")

    if "password" in fields:
        fields["password_hash"] = hash_password(fields.pop("password"))

    if not fields:
        return get_user(conn, username)

    set_clause = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [username]
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE users SET {set_clause}, updated_at = now() WHERE username = %s RETURNING *",
            values,
        )
        return _row_to_dict(cur, cur.fetchone())


def deactivate_user(conn, username: str) -> bool:
    """Soft-delete a user by setting ``is_active = FALSE``."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET is_active = FALSE, updated_at = now() WHERE username = %s",
            (username,),
        )
        return cur.rowcount > 0


def activate_user(conn, username: str) -> bool:
    """Re-activate a previously deactivated user."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET is_active = TRUE, updated_at = now() WHERE username = %s",
            (username,),
        )
        return cur.rowcount > 0


def verify_user(conn, username: str, password: str) -> dict[str, Any] | None:
    """Authenticate a user.  Returns user dict on success, ``None`` otherwise."""
    user = get_user(conn, username)
    if user is None:
        return None
    if not user["is_active"]:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


# ---------------------------------------------------------------------------
# Invite system
# ---------------------------------------------------------------------------

_INVITE_EXPIRY_DAYS = 7


def create_invite(
    conn,
    email: str,
    invited_by: str,
    *,
    role: str = "user",
    credits_per_week: int = 5,
) -> str:
    """Create an invite and return the token string."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=_INVITE_EXPIRY_DAYS)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO invites (email, token, role, credits_per_week, invited_by, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (email, token, role, credits_per_week, invited_by, expires),
        )
    log_audit(conn, invited_by, "invited", None, {"email": email, "role": role})
    return token


def get_invite(conn, token: str) -> dict[str, Any] | None:
    """Fetch an invite by token."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM invites WHERE token = %s", (token,))
        return _row_to_dict(cur, cur.fetchone())


def accept_invite(
    conn,
    token: str,
    username: str,
    password: str,
) -> dict[str, Any] | None:
    """Accept an invite: create the user and mark the invite as used.

    Returns the created user dict, or ``None`` if the token is invalid/expired.
    """
    invite = get_invite(conn, token)
    if invite is None:
        return None
    if invite["is_used"]:
        return None
    if invite["expires_at"].replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        return None

    user = create_user(
        conn,
        username,
        password,
        email=invite["email"],
        role=invite["role"],
        credits_per_week=invite["credits_per_week"],
    )

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE invites SET is_used = TRUE, accepted_at = now() WHERE token = %s",
            (token,),
        )

    log_audit(conn, username, "accepted_invite", username, {"email": invite["email"]})
    return user


def list_invites(conn) -> list[dict[str, Any]]:
    """Return all invites, most recent first."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM invites ORDER BY created_at DESC")
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def log_audit(
    conn,
    actor: str,
    action: str,
    target_user: str | None = None,
    details: dict | None = None,
) -> None:
    """Write an entry to the audit log."""
    import json

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_audit_log (actor, action, target_user, details)
            VALUES (%s, %s, %s, %s)
            """,
            (actor, action, target_user, json.dumps(details) if details else None),
        )


def get_audit_log(conn, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent audit log entries."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM user_audit_log ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------

def send_invite_email(
    email: str,
    token: str,
    inviter: str,
    app_url: str,
    smtp_cfg: dict[str, Any],
) -> bool:
    """Send an invite email.  Returns ``True`` on success.

    *smtp_cfg* keys: ``host``, ``port``, ``user``, ``password``, ``from_name``.
    """
    invite_url = f"{app_url.rstrip('/')}/?invite={token}"

    body_text = (
        f"Hi there,\n\n"
        f"{inviter} has invited you to join Memo Chef \u2014 the IC memo automation tool.\n\n"
        f"Click the link below to set up your account:\n"
        f"{invite_url}\n\n"
        f"This invitation expires in {_INVITE_EXPIRY_DAYS} days.\n\n"
        f"\u2014 Memo Chef\n"
    )

    body_html = (
        f"<p>Hi there,</p>"
        f"<p><strong>{inviter}</strong> has invited you to join "
        f"<strong>Memo Chef</strong> \u2014 the IC memo automation tool.</p>"
        f'<p><a href="{invite_url}" style="display:inline-block;padding:10px 24px;'
        f"background:#16352E;color:#fff;border-radius:6px;text-decoration:none;"
        f'font-weight:bold;">Set Up Your Account</a></p>'
        f"<p style=\"color:#888;font-size:13px;\">This invitation expires in "
        f"{_INVITE_EXPIRY_DAYS} days.</p>"
        f"<p>\u2014 Memo Chef</p>"
    )

    from_name = smtp_cfg.get("from_name", "Memo Chef")
    from_addr = smtp_cfg["user"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "You're invited to Memo Chef"
    msg["From"] = f"{from_name} <{from_addr}>"
    msg["To"] = email
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(smtp_cfg["host"], int(smtp_cfg["port"])) as server:
            server.starttls()
            server.login(smtp_cfg["user"], smtp_cfg["password"])
            server.sendmail(from_addr, [email], msg.as_string())
        logger.info("Invite email sent to %s", email)
        return True
    except Exception:
        logger.exception("Failed to send invite email to %s", email)
        return False
