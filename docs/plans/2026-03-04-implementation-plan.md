# Implementation Plan — User Management, Invites & Market Workbook

> **Date:** 2026-03-04
> **Author:** @brandon
> **Status:** In Progress

---

## Overview

This plan covers three workstreams that advance the project through the remaining
Phase 1 cleanup and into Phase 2 (Team Ready) territory:

1. **Database-backed user management** — replace secrets.toml with Postgres users table
2. **Email invite system** — admin sends invites, recipients self-register
3. **Market data workbook scaffold** — Excel generation with charts (data sources in Phase 3)

---

## Execution Order

### Step 1: User Management Module (`user_management.py`)

**Files:** New `user_management.py`

- Define `create_tables()` with idempotent DDL for `users`, `invites`, `user_audit_log`
- Implement CRUD: `create_user`, `get_user`, `list_users`, `update_user`, `deactivate_user`
- Implement `verify_user()` with PBKDF2-SHA256 (reuse existing `_hash_password` logic)
- Implement invite functions: `create_invite`, `accept_invite`, `list_invites`
- Implement audit logging: `log_audit`, `get_audit_log`
- Implement `send_invite_email()` using `smtplib` with TLS
- All functions take a `conn` parameter (dependency injection, testable)

### Step 2: Update Authentication in `app.py`

**Files:** `app.py`

- Import from `user_management`
- Call `create_tables()` on DB connection initialization
- Update `check_password()`: query DB first, fall back to secrets.toml
- Handle invite signup: detect `?invite=TOKEN` query param, show signup form

### Step 3: Admin Portal UI in `app.py`

**Files:** `app.py`

- Replace current admin sidebar with tabbed layout (Users / Invites / Audit)
- **Users tab**: Table of all users + edit form (role, credits, active status)
- **Invites tab**: Send invite form + pending invites table
- **Audit tab**: Recent audit log entries
- Add "Add User" form for direct creation (no email required)

### Step 4: Market Data Workbook Module (`market_workbook.py`)

**Files:** New `market_workbook.py`

- Define dataclasses for all data models
- Implement `generate_workbook()` — creates multi-tab .xlsx with openpyxl
- Implement per-tab writers with formatting and charts
- Create `sample_market_data.json` for testing/demos

### Step 5: Update Documentation

**Files:** `CHECKLIST.md`, `ROADMAP.md`, `docs/adr/`

- Add ADR-004: Database-backed user management
- Update CHECKLIST with new tasks and completion status
- Update ROADMAP to reflect Phase 2 progress

---

## Dependencies

```
No new pip packages required:
- psycopg2-binary  (already installed — Postgres)
- openpyxl         (already installed — Excel charts)
- smtplib          (Python stdlib — email)
- secrets          (Python stdlib — token generation)
- dataclasses      (Python stdlib — data models)
```

---

## Testing Strategy

- `user_management.py`: Unit-testable with mock DB connections
- `market_workbook.py`: Test generates valid .xlsx, verify tab names and chart count
- `app.py` changes: Manual testing via Streamlit (UI-level)

---

## Rollback Plan

- User auth falls back to secrets.toml if DB tables don't exist
- Market workbook is a standalone module with no coupling to existing pipeline
- All changes are additive — no existing functionality modified destructively
