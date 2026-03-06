from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

import psycopg2
import streamlit as st

from app_helpers import hash_password


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
            "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
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
    return conn


def get_storage_root() -> Path:
    root = Path(__file__).resolve().parent / "run_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


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
    try:
        with conn.cursor() as cur:
            yield cur
        return
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
    except Exception:
        pass
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
) -> None:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_runs ("
            " run_id, username, status, memo_name, proforma_name, property_name,"
            " dry_run, skip_validation, change_count, rejected_count, missed_count,"
            " duration_seconds, warnings_json"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id) DO UPDATE SET "
            " status = EXCLUDED.status,"
            " change_count = EXCLUDED.change_count,"
            " rejected_count = EXCLUDED.rejected_count,"
            " missed_count = EXCLUDED.missed_count,"
            " duration_seconds = EXCLUDED.duration_seconds,"
            " warnings_json = EXCLUDED.warnings_json",
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
            ),
        )


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
                "duration_seconds, created_at, warnings_json, approval_status, approved_by "
                "FROM memo_chef_runs WHERE username = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (username, limit),
            )
        else:
            cur.execute(
                "SELECT run_id, username, status, memo_name, proforma_name, property_name, "
                "dry_run, skip_validation, change_count, rejected_count, missed_count, "
                "duration_seconds, created_at, warnings_json, approval_status, approved_by "
                "FROM memo_chef_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        rows = cur.fetchall()
    results = []
    for row in rows:
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
            }
        )
    return results


def get_run_details(run_id: str) -> dict | None:
    with db_cursor() as cur:
        cur.execute(
            "SELECT run_id, username, status, memo_name, proforma_name, property_name, "
            "dry_run, skip_validation, change_count, rejected_count, missed_count, "
            "duration_seconds, created_at, warnings_json, approval_status, approval_notes, "
            "approved_by, approved_at "
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
    }


def save_profile(
    profile_name: str,
    owner_username: str,
    property_name: str | None,
    dry_run: bool,
    skip_validation: bool,
    notes: str | None = None,
) -> None:
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_profiles (profile_name, owner_username, property_name, dry_run, skip_validation, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (profile_name) DO UPDATE SET "
            "owner_username = EXCLUDED.owner_username, "
            "property_name = EXCLUDED.property_name, "
            "dry_run = EXCLUDED.dry_run, "
            "skip_validation = EXCLUDED.skip_validation, "
            "notes = EXCLUDED.notes, "
            "updated_at = now()",
            (profile_name, owner_username, property_name, dry_run, skip_validation, notes),
        )


def get_profiles(owner_username: str | None = None) -> list[dict]:
    with db_cursor() as cur:
        if owner_username:
            cur.execute(
                "SELECT profile_name, owner_username, property_name, dry_run, skip_validation, notes, updated_at "
                "FROM memo_chef_profiles WHERE owner_username = %s "
                "ORDER BY profile_name",
                (owner_username,),
            )
        else:
            cur.execute(
                "SELECT profile_name, owner_username, property_name, dry_run, skip_validation, notes, updated_at "
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
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO memo_chef_jobs (job_id, username, status, payload_json) "
            "VALUES (%s, %s, 'queued', %s) "
            "ON CONFLICT (job_id) DO UPDATE SET payload_json = EXCLUDED.payload_json, "
            "status = 'queued', updated_at = now()",
            (job_id, username, json.dumps(payload)),
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
        payload = json.loads(row[3])
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
    with db_cursor() as cur:
        cur.execute("DELETE FROM memo_chef_jobs WHERE job_id = %s", (job_id,))


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
        "payload": json.loads(row[3]),
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
