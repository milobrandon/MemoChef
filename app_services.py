from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import psycopg2
import streamlit as st

from app_helpers import hash_password

log = logging.getLogger(__name__)


_BINARY_PAYLOAD_KEY = "__memo_chef_binary__"
_BINARY_DATA_KEY = "data"


def _json_safe_payload(value: Any) -> Any:
    """Convert bytes in queue payloads into JSON-safe base64 wrappers."""
    if isinstance(value, bytes):
        return {
            _BINARY_PAYLOAD_KEY: True,
            _BINARY_DATA_KEY: base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, dict):
        return {key: _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_payload(item) for item in value]
    return value


def _restore_json_payload(value: Any) -> Any:
    """Restore base64-wrapped queue payload bytes after JSON decoding."""
    if isinstance(value, dict):
        if value.get(_BINARY_PAYLOAD_KEY) is True and isinstance(value.get(_BINARY_DATA_KEY), str):
            return base64.b64decode(value[_BINARY_DATA_KEY])
        return {key: _restore_json_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_json_payload(item) for item in value]
    return value


@st.cache_resource
def get_db_conn():
    """Return a psycopg2 connection to the app database."""
    conn = psycopg2.connect(st.secrets["CREDITS_DATABASE_URL"])
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS credit_usage ("
            "  username TEXT PRIMARY KEY,"
            "  week TEXT NOT NULL,"
            "  used INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS credit_charge_events ("
            "  username TEXT NOT NULL,"
            "  week TEXT NOT NULL,"
            "  run_id TEXT NOT NULL,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  PRIMARY KEY (username, week, run_id)"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS memo_chef_users ("
            "  username TEXT PRIMARY KEY,"
            "  password_hash TEXT NOT NULL,"
            "  role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user')),"
            "  credits_per_week INTEGER NOT NULL DEFAULT 5,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS memo_chef_runs ("
            "  run_id TEXT PRIMARY KEY,"
            "  username TEXT NOT NULL,"
            "  status TEXT NOT NULL,"
            "  memo_name TEXT NOT NULL,"
            "  proforma_name TEXT NOT NULL,"
            "  property_name TEXT,"
            "  dry_run BOOLEAN NOT NULL DEFAULT false,"
            "  skip_validation BOOLEAN NOT NULL DEFAULT false,"
            "  change_count INTEGER NOT NULL DEFAULT 0,"
            "  rejected_count INTEGER NOT NULL DEFAULT 0,"
            "  missed_count INTEGER NOT NULL DEFAULT 0,"
            "  duration_seconds DOUBLE PRECISION,"
            "  warnings_json TEXT,"
            "  approval_status TEXT NOT NULL DEFAULT 'pending',"
            "  approval_notes TEXT,"
            "  approved_by TEXT,"
            "  approved_at TIMESTAMPTZ,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS approval_status TEXT NOT NULL DEFAULT 'pending'"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS input_tokens INTEGER NOT NULL DEFAULT 0"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS output_tokens INTEGER NOT NULL DEFAULT 0"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS estimated_cost_microdollars INTEGER NOT NULL DEFAULT 0"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS approval_notes TEXT"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS approved_by TEXT"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS memo_chef_profiles ("
            "  profile_name TEXT PRIMARY KEY,"
            "  owner_username TEXT NOT NULL,"
            "  property_name TEXT,"
            "  dry_run BOOLEAN NOT NULL DEFAULT false,"
            "  skip_validation BOOLEAN NOT NULL DEFAULT false,"
            "  notes TEXT,"
            "  config_profile TEXT,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        cur.execute(
            "ALTER TABLE memo_chef_profiles ADD COLUMN IF NOT EXISTS config_profile TEXT"
        )
        cur.execute(
            "ALTER TABLE memo_chef_profiles ADD COLUMN IF NOT EXISTS property_rename_to TEXT"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS memo_chef_jobs ("
            "  job_id TEXT PRIMARY KEY,"
            "  username TEXT NOT NULL,"
            "  status TEXT NOT NULL,"
            "  payload_json TEXT NOT NULL,"
            "  run_id TEXT,"
            "  error_message TEXT,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS memo_chef_invitations ("
            "  id TEXT PRIMARY KEY,"
            "  email TEXT NOT NULL,"
            "  role TEXT NOT NULL DEFAULT 'user',"
            "  credits_per_week INTEGER NOT NULL DEFAULT 5,"
            "  status TEXT NOT NULL DEFAULT 'pending',"
            "  invited_by TEXT NOT NULL,"
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  accepted_at TIMESTAMPTZ,"
            "  expires_at TIMESTAMPTZ NOT NULL"
            ")"
        )
        cur.execute(
            "ALTER TABLE memo_chef_users ADD COLUMN IF NOT EXISTS email TEXT"
        )
        # Phase 3: accuracy metrics + slide insertion + manifest storage
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS slides_inserted INTEGER DEFAULT 0"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS confidence_score REAL"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS coverage_pct REAL"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS correction_rate_pct REAL"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS run_manifest_json TEXT"
        )
        cur.execute(
            "ALTER TABLE memo_chef_runs ADD COLUMN IF NOT EXISTS change_log_html TEXT"
        )
    return conn


def get_storage_root() -> Path:
    root = Path(__file__).resolve().parent / "run_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_job_staging_dir(job_id: str) -> Path:
    """Return (and create) a directory for staging job input files at enqueue time."""
    path = get_storage_root() / f"job_{job_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_storage_dir(run_id: str) -> Path:
    path = get_storage_root() / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_users_seeded() -> None:
    try:
        secrets_users = dict(st.secrets["users"])
    except (KeyError, FileNotFoundError):
        return
    with db_cursor() as cur:
        cur.execute("SELECT count(*) FROM memo_chef_users")
        if cur.fetchone()[0] > 0:
            return
        for uname, user_cfg in secrets_users.items():
            cur.execute(
                "INSERT INTO memo_chef_users (username, password_hash, role, credits_per_week) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    uname,
                    user_cfg["password_hash"],
                    user_cfg.get("role", "user"),
                    int(user_cfg.get("credits_per_week", 5)),
                ),
            )


@contextmanager
def db_cursor():
    conn = get_db_conn()
    # Detect stale connections before yielding (a context manager must yield exactly once)
    try:
        conn.cursor().close()
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        get_db_conn.clear()
        conn = get_db_conn()
    with conn.cursor() as cur:
        yield cur


def current_week_start() -> str:
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    return monday.strftime("%Y-%m-%d")


def get_users() -> dict:
    try:
        with db_cursor() as cur:
            cur.execute(
                "SELECT username, password_hash, role, credits_per_week FROM memo_chef_users"
            )
            rows = cur.fetchall()
        if rows:
            return {
                row[0]: {
                    "password_hash": row[1],
                    "role": row[2],
                    "credits_per_week": row[3],
                }
                for row in rows
            }
    except Exception as e:
        log.warning("Failed to load users from database, falling back to secrets: %s", e)
    try:
        return dict(st.secrets["users"])
    except (KeyError, FileNotFoundError):
        return {}


def get_all_usernames() -> list[str]:
    with db_cursor() as cur:
        cur.execute("SELECT username FROM memo_chef_users ORDER BY username")
        return [row[0] for row in cur.fetchall()]


def get_user_credits(username: str, credits_per_week: int) -> tuple[int, int]:
    week = current_week_start()
    with db_cursor() as cur:
        cur.execute("SELECT week, used FROM credit_usage WHERE username = %s", (username,))
        row = cur.fetchone()
        if row is None or row[0] != week:
            cur.execute(
                "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 0) "
                "ON CONFLICT (username) DO UPDATE SET week = %s, used = 0",
                (username, week, week),
            )
            return 0, credits_per_week
        used = row[1]
    return used, max(0, credits_per_week - used)


def consume_credit(username: str, credits_per_week: int, run_id: str | None = None) -> bool:
    week = current_week_start()
    with db_cursor() as cur:
        if run_id:
            cur.execute(
                "INSERT INTO credit_charge_events (username, week, run_id) VALUES (%s, %s, %s) "
                "ON CONFLICT DO NOTHING RETURNING run_id",
                (username, week, run_id),
            )
            inserted = cur.fetchone()
            if inserted is None:
                return True
        cur.execute(
            "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 0) "
            "ON CONFLICT (username) DO UPDATE SET "
            "  used = CASE WHEN credit_usage.week = %s THEN credit_usage.used ELSE 0 END, "
            "  week = %s",
            (username, week, week, week),
        )
        cur.execute(
            "UPDATE credit_usage SET used = used + 1 "
            "WHERE username = %s AND week = %s AND used < %s "
            "RETURNING used",
            (username, week, credits_per_week),
        )
        charged = cur.fetchone() is not None
        if not charged and run_id:
            cur.execute(
                "DELETE FROM credit_charge_events WHERE username = %s AND week = %s AND run_id = %s",
                (username, week, run_id),
            )
        return charged


def reset_user_credits(username: str) -> None:
    week = current_week_start()
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO credit_usage (username, week, used) VALUES (%s, %s, 0) "
            "ON CONFLICT (username) DO UPDATE SET week = %s, used = 0",
            (username, week, week),
        )


def add_user(username: str, password: str, role: str, credits_per_week: int) -> bool:
    password_hash = hash_password(password)
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_users (username, password_hash, role, credits_per_week) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING username",
            (username, password_hash, role, credits_per_week),
        )
        return cur.fetchone() is not None


def update_user(
    username: str,
    role: str | None = None,
    credits_per_week: int | None = None,
    new_password: str | None = None,
) -> None:
    with db_cursor() as cur:
        if role is not None:
            cur.execute(
                "UPDATE memo_chef_users SET role = %s, updated_at = now() WHERE username = %s",
                (role, username),
            )
        if credits_per_week is not None:
            cur.execute(
                "UPDATE memo_chef_users SET credits_per_week = %s, updated_at = now() WHERE username = %s",
                (credits_per_week, username),
            )
        if new_password is not None:
            password_hash = hash_password(new_password)
            cur.execute(
                "UPDATE memo_chef_users SET password_hash = %s, updated_at = now() WHERE username = %s",
                (password_hash, username),
            )


def delete_user(username: str) -> None:
    with db_cursor() as cur:
        cur.execute("DELETE FROM credit_usage WHERE username = %s", (username,))
        cur.execute("DELETE FROM credit_charge_events WHERE username = %s", (username,))
        cur.execute("DELETE FROM memo_chef_users WHERE username = %s", (username,))


def record_run(
    *,
    run_id: str,
    username: str,
    status: str,
    memo_name: str,
    proforma_name: str,
    property_name: str | None,
    dry_run: bool,
    skip_validation: bool,
    change_count: int,
    rejected_count: int,
    missed_count: int,
    duration_seconds: float | None,
    warnings: list[dict] | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_microdollars: int = 0,
    slides_inserted: int = 0,
    confidence_score: float | None = None,
    coverage_pct: float | None = None,
    correction_rate_pct: float | None = None,
    run_manifest_json: str | None = None,
    change_log_html: str | None = None,
    _conn: Any | None = None,
) -> None:
    """Record a run. Pass _conn for thread-safe usage from background worker."""
    def _do_insert(cur):
        cur.execute(
            "INSERT INTO memo_chef_runs ("
            " run_id, username, status, memo_name, proforma_name, property_name,"
            " dry_run, skip_validation, change_count, rejected_count, missed_count,"
            " duration_seconds, warnings_json, input_tokens, output_tokens,"
            " estimated_cost_microdollars, slides_inserted, confidence_score,"
            " coverage_pct, correction_rate_pct, run_manifest_json, change_log_html"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id) DO UPDATE SET "
            " status = EXCLUDED.status,"
            " change_count = EXCLUDED.change_count,"
            " rejected_count = EXCLUDED.rejected_count,"
            " missed_count = EXCLUDED.missed_count,"
            " duration_seconds = EXCLUDED.duration_seconds,"
            " warnings_json = EXCLUDED.warnings_json,"
            " input_tokens = EXCLUDED.input_tokens,"
            " output_tokens = EXCLUDED.output_tokens,"
            " estimated_cost_microdollars = EXCLUDED.estimated_cost_microdollars,"
            " slides_inserted = EXCLUDED.slides_inserted,"
            " confidence_score = EXCLUDED.confidence_score,"
            " coverage_pct = EXCLUDED.coverage_pct,"
            " correction_rate_pct = EXCLUDED.correction_rate_pct,"
            " run_manifest_json = EXCLUDED.run_manifest_json,"
            " change_log_html = EXCLUDED.change_log_html",
            (
                run_id,
                username,
                status,
                memo_name,
                proforma_name,
                property_name,
                dry_run,
                skip_validation,
                change_count,
                rejected_count,
                missed_count,
                duration_seconds,
                json.dumps(warnings or []),
                input_tokens,
                output_tokens,
                estimated_cost_microdollars,
                slides_inserted,
                confidence_score,
                coverage_pct,
                correction_rate_pct,
                run_manifest_json,
                change_log_html,
            ),
        )

    if _conn is not None:
        with _conn.cursor() as cur:
            _do_insert(cur)
    else:
        with db_cursor() as cur:
            _do_insert(cur)


def update_run_approval(
    run_id: str,
    approval_status: str,
    approved_by: str,
    approval_notes: str | None = None,
) -> None:
    with db_cursor() as cur:
        cur.execute(
            "UPDATE memo_chef_runs SET approval_status = %s, approval_notes = %s, "
            "approved_by = %s, approved_at = now() WHERE run_id = %s",
            (approval_status, approval_notes, approved_by, run_id),
        )


def get_recent_runs(username: str | None = None, limit: int = 20) -> list[dict]:
    with db_cursor() as cur:
        if username:
            cur.execute(
                "SELECT run_id, username, status, memo_name, proforma_name, property_name, "
                "dry_run, skip_validation, change_count, rejected_count, missed_count, "
                "duration_seconds, created_at, warnings_json, approval_status, approved_by, "
                "input_tokens, output_tokens, estimated_cost_microdollars, confidence_score, slides_inserted "
                "FROM memo_chef_runs WHERE username = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (username, limit),
            )
        else:
            cur.execute(
                "SELECT run_id, username, status, memo_name, proforma_name, property_name, "
                "dry_run, skip_validation, change_count, rejected_count, missed_count, "
                "duration_seconds, created_at, warnings_json, approval_status, approved_by, "
                "input_tokens, output_tokens, estimated_cost_microdollars, confidence_score, slides_inserted "
                "FROM memo_chef_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        rows = cur.fetchall()
    results = []
    for row in rows:
        cost_usd = (row[18] or 0) / 1_000_000
        results.append(
            {
                "Run ID": row[0],
                "User": row[1],
                "Status": row[2],
                "Memo": row[3],
                "Proforma": row[4],
                "Property": row[5] or "",
                "Dry Run": "Yes" if row[6] else "No",
                "Skip QA": "Yes" if row[7] else "No",
                "Changes": row[8],
                "Rejected": row[9],
                "Missed": row[10],
                "Duration (s)": round(row[11], 1) if row[11] is not None else None,
                "Created": row[12].strftime("%Y-%m-%d %H:%M") if row[12] else "",
                "Warnings": len(json.loads(row[13] or "[]")),
                "Approval": row[14],
                "Reviewer": row[15] or "",
                "Tokens In": row[16] or 0,
                "Tokens Out": row[17] or 0,
                "Est. Cost ($)": round(cost_usd, 4) if cost_usd else None,
                "Confidence": row[19],
                "Slides Inserted": row[20] or 0,
            }
        )
    return results


def get_run_details(run_id: str) -> dict | None:
    with db_cursor() as cur:
        cur.execute(
            "SELECT run_id, username, status, memo_name, proforma_name, property_name, "
            "dry_run, skip_validation, change_count, rejected_count, missed_count, "
            "duration_seconds, created_at, warnings_json, approval_status, approval_notes, "
            "approved_by, approved_at, slides_inserted, confidence_score, coverage_pct, "
            "correction_rate_pct, run_manifest_json, change_log_html "
            "FROM memo_chef_runs WHERE run_id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "username": row[1],
        "status": row[2],
        "memo_name": row[3],
        "proforma_name": row[4],
        "property_name": row[5] or "",
        "dry_run": row[6],
        "skip_validation": row[7],
        "change_count": row[8],
        "rejected_count": row[9],
        "missed_count": row[10],
        "duration_seconds": row[11],
        "created_at": row[12].strftime("%Y-%m-%d %H:%M") if row[12] else "",
        "warnings": json.loads(row[13] or "[]"),
        "approval_status": row[14],
        "approval_notes": row[15] or "",
        "approved_by": row[16] or "",
        "approved_at": row[17].strftime("%Y-%m-%d %H:%M") if row[17] else "",
        "slides_inserted": row[18] or 0,
        "confidence_score": row[19],
        "coverage_pct": row[20],
        "correction_rate_pct": row[21],
        "run_manifest_json": row[22],
        "change_log_html": row[23],
    }


def save_profile(
    profile_name: str,
    owner_username: str,
    property_name: str | None,
    dry_run: bool,
    skip_validation: bool,
    notes: str | None = None,
    config_profile: str | None = None,
    property_rename_to: str | None = None,
) -> None:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_profiles "
            "(profile_name, owner_username, property_name, property_rename_to, dry_run, skip_validation, notes, config_profile) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (profile_name) DO UPDATE SET "
            "owner_username = EXCLUDED.owner_username, "
            "property_name = EXCLUDED.property_name, "
            "property_rename_to = EXCLUDED.property_rename_to, "
            "dry_run = EXCLUDED.dry_run, "
            "skip_validation = EXCLUDED.skip_validation, "
            "notes = EXCLUDED.notes, "
            "config_profile = EXCLUDED.config_profile, "
            "updated_at = now()",
            (profile_name, owner_username, property_name, property_rename_to, dry_run, skip_validation, notes, config_profile or None),
        )


def get_profiles(owner_username: str | None = None) -> list[dict]:
    with db_cursor() as cur:
        if owner_username:
            cur.execute(
                "SELECT profile_name, owner_username, property_name, dry_run, skip_validation, notes, updated_at, config_profile, property_rename_to "
                "FROM memo_chef_profiles WHERE owner_username = %s "
                "ORDER BY profile_name",
                (owner_username,),
            )
        else:
            cur.execute(
                "SELECT profile_name, owner_username, property_name, dry_run, skip_validation, notes, updated_at, config_profile, property_rename_to "
                "FROM memo_chef_profiles ORDER BY profile_name"
            )
        rows = cur.fetchall()
    return [
        {
            "Profile": row[0],
            "Owner": row[1],
            "Property": row[2] or "",
            "Preview Only": row[3],
            "Skip QA": row[4],
            "Notes": row[5] or "",
            "Updated": row[6].strftime("%Y-%m-%d %H:%M") if row[6] else "",
            "Config Profile": row[7] or "",
            "Rename To": row[8] or "",
        }
        for row in rows
    ]


def get_platform_health() -> list[dict]:
    checks: list[dict] = []
    try:
        with db_cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        checks.append({"Component": "Database", "Status": "healthy", "Detail": "Connected"})
    except Exception as err:
        checks.append({"Component": "Database", "Status": "error", "Detail": str(err)})

    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        detail = "Configured" if api_key else "Missing"
        status = "healthy" if api_key else "warning"
    except (KeyError, FileNotFoundError):
        status, detail = "warning", "Missing"
    checks.append({"Component": "Anthropic API key", "Status": status, "Detail": detail})

    for label, path in [
        ("Config", "config.yaml"),
        ("Theme", "Subtext Brand Theme.thmx"),
        ("Mapping prompt", "prompts/mapping_v1.txt"),
        ("Validation prompt", "prompts/validation_v1.txt"),
    ]:
        status = "healthy" if os.path.exists(path) else "warning"
        detail = "Present" if os.path.exists(path) else "Missing"
        checks.append({"Component": label, "Status": status, "Detail": detail})
    storage_root = get_storage_root()
    checks.append(
        {
            "Component": "Artifact storage",
            "Status": "healthy" if storage_root.exists() else "warning",
            "Detail": str(storage_root),
        }
    )
    return checks


def enqueue_job(username: str, payload: dict) -> str:
    job_id = payload["job_id"]
    payload_json = json.dumps(_json_safe_payload(payload))
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_jobs (job_id, username, status, payload_json) "
            "VALUES (%s, %s, 'queued', %s) "
            "ON CONFLICT (job_id) DO UPDATE SET payload_json = EXCLUDED.payload_json, "
            "status = 'queued', updated_at = now()",
            (job_id, username, payload_json),
        )
    return job_id


def update_job_status(
    job_id: str,
    status: str,
    *,
    run_id: str | None = None,
    error_message: str | None = None,
) -> None:
    with db_cursor() as cur:
        cur.execute(
            "UPDATE memo_chef_jobs SET status = %s, run_id = COALESCE(%s, run_id), "
            "error_message = %s, updated_at = now() WHERE job_id = %s",
            (status, run_id, error_message, job_id),
        )


def get_job_queue(username: str | None = None) -> list[dict]:
    with db_cursor() as cur:
        if username:
            cur.execute(
                "SELECT job_id, username, status, payload_json, run_id, error_message, created_at, updated_at "
                "FROM memo_chef_jobs WHERE username = %s ORDER BY created_at ASC",
                (username,),
            )
        else:
            cur.execute(
                "SELECT job_id, username, status, payload_json, run_id, error_message, created_at, updated_at "
                "FROM memo_chef_jobs ORDER BY created_at ASC"
            )
        rows = cur.fetchall()
    results = []
    for row in rows:
        payload = _restore_json_payload(json.loads(row[3]))
        results.append(
            {
                "job_id": row[0],
                "username": row[1],
                "status": row[2],
                "payload": payload,
                "run_id": row[4] or "",
                "error_message": row[5] or "",
                "created_at": row[6].strftime("%Y-%m-%d %H:%M") if row[6] else "",
                "updated_at": row[7].strftime("%Y-%m-%d %H:%M") if row[7] else "",
            }
        )
    return results


def delete_job(job_id: str) -> None:
    import shutil
    with db_cursor() as cur:
        cur.execute("DELETE FROM memo_chef_jobs WHERE job_id = %s", (job_id,))
    staging_dir = get_storage_root() / f"job_{job_id}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


def get_job(job_id: str) -> dict | None:
    with db_cursor() as cur:
        cur.execute(
            "SELECT job_id, username, status, payload_json, run_id, error_message "
            "FROM memo_chef_jobs WHERE job_id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "job_id": row[0],
        "username": row[1],
        "status": row[2],
        "payload": _restore_json_payload(json.loads(row[3])),
        "run_id": row[4] or "",
        "error_message": row[5] or "",
    }


def get_run_artifact_paths(run_id: str) -> dict[str, str]:
    run_dir = get_run_storage_dir(run_id)
    paths: dict[str, str] = {}
    for name in [
        "memo",
        "change_log",
        "run_manifest",
        "input_memo",
        "input_proforma",
        "input_schedule",
        "input_market_data",
    ]:
        candidates = list(run_dir.glob(f"{name}*"))
        if candidates:
            paths[name] = str(candidates[0])
    return paths


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------

def create_invitation(
    email: str,
    role: str,
    credits_per_week: int,
    invited_by: str,
    expiry_hours: int = 48,
) -> str:
    """Create an invitation and return the token (UUID4)."""
    token = str(uuid.uuid4())
    expires_at = datetime.now() + timedelta(hours=expiry_hours)
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_invitations "
            "(id, email, role, credits_per_week, invited_by, expires_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (token, email.strip().lower(), role, credits_per_week, invited_by, expires_at),
        )
    return token


def get_invitation(token: str) -> dict | None:
    """Return invitation dict or None if not found."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, email, role, credits_per_week, status, invited_by, "
            "created_at, accepted_at, expires_at "
            "FROM memo_chef_invitations WHERE id = %s",
            (token,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "email": row[1],
        "role": row[2],
        "credits_per_week": row[3],
        "status": row[4],
        "invited_by": row[5],
        "created_at": row[6],
        "accepted_at": row[7],
        "expires_at": row[8],
    }


def accept_invitation(token: str, username: str, password: str) -> bool:
    """Accept an invitation: create user account and mark token as used.

    Returns True on success, False if token is invalid/expired/already used
    or username is taken.
    """
    invite = get_invitation(token)
    if invite is None or invite["status"] != "pending":
        return False
    if invite["expires_at"] < datetime.now(invite["expires_at"].tzinfo):
        with db_cursor() as cur:
            cur.execute(
                "UPDATE memo_chef_invitations SET status = 'expired' WHERE id = %s",
                (token,),
            )
        return False

    password_hash = hash_password(password)
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_users (username, password_hash, role, credits_per_week, email) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING username",
            (username, password_hash, invite["role"], invite["credits_per_week"], invite["email"]),
        )
        if cur.fetchone() is None:
            return False
        cur.execute(
            "UPDATE memo_chef_invitations SET status = 'accepted', accepted_at = now() "
            "WHERE id = %s",
            (token,),
        )
    return True


def get_invitations() -> list[dict]:
    """Return all invitations, most recent first."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, email, role, credits_per_week, status, invited_by, "
            "created_at, accepted_at, expires_at "
            "FROM memo_chef_invitations ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "email": row[1],
            "role": row[2],
            "credits_per_week": row[3],
            "status": row[4],
            "invited_by": row[5],
            "created_at": row[6],
            "accepted_at": row[7],
            "expires_at": row[8],
        }
        for row in rows
    ]


def send_invitation_email(email: str, token: str, app_url: str | None = None) -> bool:
    """Send an invitation email via Resend. Returns True on success."""
    import resend

    try:
        api_key = st.secrets["RESEND_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("RESEND_API_KEY not configured in secrets.")
        return False

    resend.api_key = api_key

    if not app_url:
        app_url = st.secrets.get("APP_URL", "https://memochef.streamlit.app")
    invite_url = f"{app_url}?invite={token}"

    try:
        from_addr = st.secrets.get("RESEND_FROM", "Memo Chef <onboarding@resend.dev>")
    except (KeyError, FileNotFoundError):
        from_addr = "Memo Chef <onboarding@resend.dev>"

    html_body = f"""\
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 560px; margin: 0 auto; padding: 32px;">
  <h2 style="color: #1a1a2e; margin-bottom: 8px;">
    Step into Memo City's Hottest Kitchen
  </h2>
  <p style="color: #555; line-height: 1.6;">
    Welcome to <strong>Memo Chef</strong> &mdash; Subtext's in-house tool that
    takes your Excel proforma and PowerPoint IC memo, then updates every metric
    automatically so you never have to copy-paste numbers again.
  </p>
  <p style="color: #555; line-height: 1.6;">
    Upload your files, let the Chef cook, and get back a polished memo with a
    full change log in minutes. Your account is ready &mdash; click below to
    set your password and start your first run.
  </p>
  <a href="{invite_url}"
     style="display: inline-block; background: #6c63ff; color: #fff;
            text-decoration: none; padding: 12px 28px; border-radius: 6px;
            font-weight: 600; margin: 24px 0;">
    Get Started
  </a>
  <p style="color: #999; font-size: 13px; margin-top: 24px;">
    This link expires in 48 hours. If you didn't expect this email,
    you can safely ignore it.
  </p>
</div>"""

    resend.Emails.send({
        "from": from_addr,
        "to": [email],
        "subject": "Step into Memo City's Hottest Kitchen",
        "html": html_body,
    })
    return True


# ============================================================================
# Background Worker
# ============================================================================

def _worker_db_conn() -> psycopg2.extensions.connection:
    """Create a fresh DB connection for the background worker thread."""
    conn = psycopg2.connect(st.secrets["CREDITS_DATABASE_URL"])
    conn.autocommit = True
    return conn


def _reset_stale_jobs(conn: psycopg2.extensions.connection, stale_minutes: int = 30) -> int:
    """Reset jobs stuck in 'running' for more than stale_minutes back to 'queued'."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memo_chef_jobs SET status = 'queued', error_message = 'Reset: stale running job', "
            "updated_at = now() "
            "WHERE status = 'running' AND updated_at < now() - interval '%s minutes' "
            "RETURNING job_id",
            (stale_minutes,),
        )
        rows = cur.fetchall()
    if rows:
        log.info("Reset %d stale running jobs: %s", len(rows), [r[0] for r in rows])
    return len(rows)


def _claim_next_queued_job(conn: psycopg2.extensions.connection) -> dict | None:
    """Atomically claim the oldest queued job using FOR UPDATE SKIP LOCKED."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT job_id, username, payload_json "
            "FROM memo_chef_jobs WHERE status = 'queued' "
            "ORDER BY created_at ASC LIMIT 1 "
            "FOR UPDATE SKIP LOCKED"
        )
        row = cur.fetchone()
        if row is None:
            return None
        job_id, job_username, payload_json = row
        cur.execute(
            "UPDATE memo_chef_jobs SET status = 'running', updated_at = now() "
            "WHERE job_id = %s",
            (job_id,),
        )
    payload = _restore_json_payload(json.loads(payload_json))
    return {"job_id": job_id, "username": job_username, "payload": payload}


def _execute_job_headless(job: dict, api_key: str) -> bool:
    """Execute a job without any Streamlit UI calls (for background worker)."""
    from memo_chef.models import CompUrl, RunRequest
    from memo_chef.pipeline import run_memo_pipeline

    job_id = job["job_id"]
    payload = job["payload"]
    run_id = uuid.uuid4().hex

    try:
        # Use a worker-owned DB connection for status updates
        conn = _worker_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE memo_chef_jobs SET run_id = %s, updated_at = now() WHERE job_id = %s",
                (run_id, job_id),
            )

        run_dir = get_run_storage_dir(run_id)

        # Resolve file paths — support both path-based and bytes-based payloads
        memo_path = payload.get("memo_path")
        if not memo_path or not os.path.isfile(memo_path):
            memo_path = str(run_dir / f"input_memo{os.path.splitext(payload['memo_name'])[1]}")
            with open(memo_path, "wb") as f:
                f.write(payload["memo_bytes"])

        proforma_path = payload.get("proforma_path")
        if not proforma_path or not os.path.isfile(proforma_path):
            proforma_path = str(run_dir / f"input_proforma{os.path.splitext(payload['proforma_name'])[1]}")
            with open(proforma_path, "wb") as f:
                f.write(payload["proforma_bytes"])

        schedule_path = None
        if payload.get("schedule_path") and os.path.isfile(payload["schedule_path"]):
            schedule_path = payload["schedule_path"]
        elif payload.get("schedule_bytes"):
            schedule_path = str(run_dir / f"input_schedule{os.path.splitext(payload['schedule_name'])[1]}")
            with open(schedule_path, "wb") as f:
                f.write(payload["schedule_bytes"])

        market_data_path = None
        if payload.get("market_data_path") and os.path.isfile(payload["market_data_path"]):
            market_data_path = payload["market_data_path"]
        elif payload.get("market_data_bytes"):
            market_data_path = str(run_dir / f"input_market_data{os.path.splitext(payload['market_data_name'])[1]}")
            with open(market_data_path, "wb") as f:
                f.write(payload["market_data_bytes"])

        supplemental_path = None
        supplemental_type = payload.get("supplemental_type")
        if supplemental_type == "url":
            supplemental_path = payload.get("supplemental_name")
        elif payload.get("supplemental_path") and os.path.isfile(payload["supplemental_path"]):
            supplemental_path = payload["supplemental_path"]
        elif payload.get("supplemental_bytes"):
            ext = os.path.splitext(payload["supplemental_name"])[1] if payload.get("supplemental_name") else ".pdf"
            supplemental_path = str(run_dir / f"input_supplemental{ext}")
            with open(supplemental_path, "wb") as f:
                f.write(payload["supplemental_bytes"])

        comp_url_objects = [CompUrl(**cu) for cu in payload.get("comp_urls", [])]

        config_profile_name = payload.get("config_profile_name", "")
        config_override_path = None
        if config_profile_name:
            profiles_dir = Path(__file__).resolve().parent / "config_profiles"
            candidate = profiles_dir / f"{config_profile_name}.yaml"
            if candidate.exists():
                config_override_path = str(candidate)

        request = RunRequest(
            memo_path=memo_path,
            proforma_path=proforma_path,
            schedule_path=schedule_path,
            market_data_path=market_data_path,
            supplemental_path=supplemental_path,
            supplemental_type=supplemental_type,
            supplemental_brief=payload.get("supplemental_brief"),
            comp_urls=comp_url_objects,
            output_dir=str(run_dir),
            api_key=api_key,
            config_path=os.path.join(os.path.dirname(__file__), "config.yaml"),
            config_override_path=config_override_path,
            run_id=run_id,
            property_name=payload.get("property_name"),
            property_rename_to=payload.get("property_rename_to"),
            dry_run=payload.get("dry_run", False),
            skip_validation=payload.get("skip_validation", False),
            use_batch_api=payload.get("use_batch_api", False),
        )

        result = run_memo_pipeline(request)
        (run_dir / f"memo{os.path.splitext(payload['memo_name'])[1]}").write_bytes(result.memo_bytes)
        (run_dir / "change_log.md").write_bytes(result.log_bytes)
        (run_dir / "run_manifest.json").write_bytes(result.manifest_bytes)

        record_run(
            run_id=run_id,
            username=job["username"],
            status=result.manifest.status,
            memo_name=result.manifest.memo_name,
            proforma_name=result.manifest.proforma_name,
            property_name=result.manifest.property_name,
            dry_run=payload.get("dry_run", False),
            skip_validation=payload.get("skip_validation", False),
            change_count=len(result.changes),
            rejected_count=len(result.rejected),
            missed_count=len(result.missed),
            duration_seconds=0,
            warnings=[w.model_dump() for w in result.manifest.warnings],
            input_tokens=result.manifest.counts.get("input_tokens", 0),
            output_tokens=result.manifest.counts.get("output_tokens", 0),
            estimated_cost_microdollars=result.manifest.counts.get("estimated_cost_microdollars", 0),
            slides_inserted=result.manifest.counts.get("slides_inserted", 0),
            confidence_score=(result.manifest.accuracy or {}).get("confidence_score"),
            coverage_pct=(result.manifest.accuracy or {}).get("coverage_pct"),
            correction_rate_pct=(result.manifest.accuracy or {}).get("correction_rate_pct"),
            run_manifest_json=result.manifest_bytes.decode("utf-8") if result.manifest_bytes else None,
            change_log_html=result.log_bytes.decode("utf-8") if result.log_bytes else None,
            _conn=conn,
        )

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE memo_chef_jobs SET status = 'completed', updated_at = now() WHERE job_id = %s",
                (job_id,),
            )
        log.info("Background worker completed job %s (run %s)", job_id, run_id)
        conn.close()
        return True

    except Exception as exc:
        log.exception("Background worker failed job %s: %s", job_id, exc)
        try:
            conn = _worker_db_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memo_chef_jobs SET status = 'failed', error_message = %s, "
                    "updated_at = now() WHERE job_id = %s",
                    (str(exc)[:500], job_id),
                )
            conn.close()
        except Exception:
            log.exception("Failed to update job status for %s", job_id)
        return False


def _worker_loop(api_key: str, poll_interval: int = 10) -> None:
    """Poll for queued jobs and execute them. Runs in a daemon thread."""
    log.info("Background worker thread started (poll every %ds)", poll_interval)
    while True:
        try:
            conn = _worker_db_conn()
            _reset_stale_jobs(conn)
            job = _claim_next_queued_job(conn)
            conn.close()
            if job:
                log.info("Worker claimed job %s", job["job_id"])
                _execute_job_headless(job, api_key)
            else:
                time.sleep(poll_interval)
        except Exception:
            log.exception("Worker loop error")
            time.sleep(poll_interval)


@st.cache_resource
def start_background_worker() -> threading.Thread:
    """Start the background worker daemon thread. Called once per Streamlit process."""
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        log.warning("ANTHROPIC_API_KEY not in secrets — background worker disabled")
        return None
    t = threading.Thread(target=_worker_loop, args=(api_key,), daemon=True, name="memo-chef-worker")
    t.start()
    log.info("Background worker thread launched: %s", t.name)
    return t
