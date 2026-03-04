# User Management & Email Invites — Design Doc

> **Date:** 2026-03-04
> **Author:** @brandon
> **Status:** Accepted

---

## Problem

Users are currently defined in `.streamlit/secrets.toml` as static TOML entries.
Adding, removing, or modifying users requires editing a config file, regenerating
password hashes manually, and redeploying. There is no way for an admin to manage
users from the UI, and no mechanism for inviting new team members via email.

### Current State

```toml
# .streamlit/secrets.toml
[users.brandon]
password_hash = "salt_hex:hash_hex"
role = "admin"
credits_per_week = 20
```

**Pain points:**
- Manual password hash generation (CLI command)
- Redeploy required for every user change
- No self-service invite flow
- No user deactivation (only full removal from config)
- No audit trail for user lifecycle events

---

## Goals

| # | Goal | Success Metric |
|---|------|----------------|
| U1 | Admin can add/edit/deactivate users from the web UI | Zero config file edits for user changes |
| U2 | Admin can send email invites with secure signup links | New user onboarded in < 2 minutes |
| U3 | Invite tokens expire and are single-use | No stale/reusable tokens |
| U4 | User lifecycle is auditable | Created/invited/deactivated events logged |
| U5 | Backward compatible | Existing secrets.toml users still work as fallback |

---

## Database Schema

### New Tables

```sql
-- Replaces secrets.toml [users.*] entries
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    username        TEXT UNIQUE NOT NULL,
    display_name    TEXT NOT NULL DEFAULT '',
    email           TEXT UNIQUE,
    password_hash   TEXT NOT NULL,          -- PBKDF2-SHA256 (salt:hash)
    role            TEXT NOT NULL DEFAULT 'user',  -- 'admin', 'user', 'viewer'
    credits_per_week INTEGER NOT NULL DEFAULT 5,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Pending email invitations
CREATE TABLE invites (
    id              SERIAL PRIMARY KEY,
    email           TEXT NOT NULL,
    token           TEXT UNIQUE NOT NULL,   -- 32-byte URL-safe random token
    role            TEXT NOT NULL DEFAULT 'user',
    credits_per_week INTEGER NOT NULL DEFAULT 5,
    invited_by      TEXT NOT NULL,          -- username of admin who sent invite
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL,   -- created_at + 7 days
    accepted_at     TIMESTAMPTZ,            -- NULL until accepted
    is_used         BOOLEAN NOT NULL DEFAULT FALSE
);

-- Audit log for user lifecycle events
CREATE TABLE user_audit_log (
    id              SERIAL PRIMARY KEY,
    actor           TEXT NOT NULL,          -- who performed the action
    action          TEXT NOT NULL,          -- 'created', 'invited', 'activated', 'deactivated', 'role_changed', 'credits_reset'
    target_user     TEXT,                   -- affected username
    details         JSONB,                  -- additional context
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Migration of `credit_usage`

The existing `credit_usage` table stays unchanged — it tracks weekly consumption
and resets automatically. No migration needed.

---

## Roles

| Role | Permissions |
|------|------------|
| `admin` | Full CRUD on users, send invites, reset credits, view audit log, run memos |
| `user` | Run memos (credit-gated), view own profile |
| `viewer` | View-only access to results (future — not implemented in v1) |

---

## Authentication Flow

### Login (updated)

```
1. User enters username + password
2. Check database `users` table first (WHERE username = ? AND is_active = TRUE)
3. If not found, fall back to secrets.toml (backward compat)
4. Verify PBKDF2-SHA256 hash
5. Set session state: username, role, credits_per_week
```

### Invite + Signup Flow

```
1. Admin enters email + role + credits_per_week in Admin Panel
2. System generates a 32-byte URL-safe token, inserts into `invites` table
3. System sends email with signup link: {APP_URL}/invite?token={token}
4. Recipient clicks link → Streamlit shows signup form
5. Recipient chooses username + password
6. System validates token (not expired, not used), creates user, marks invite used
7. User can now log in
```

---

## Email Delivery

### Approach: SMTP via `smtplib`

Use Python's built-in `smtplib` with TLS. Configuration stored in Streamlit secrets.

```toml
# .streamlit/secrets.toml (new entries)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "memochef@company.com"
SMTP_PASSWORD = "app-password"
SMTP_FROM_NAME = "Memo Chef"
APP_URL = "https://memochef.streamlit.app"
```

### Email Template

```
Subject: You're invited to Memo Chef

Hi there,

{inviter_name} has invited you to join Memo Chef — the IC memo automation tool.

Click the link below to set up your account:
{APP_URL}?invite={token}

This invitation expires in 7 days.

— Memo Chef
```

### Fallback

If SMTP is not configured, the admin panel shows the raw invite link for
manual sharing (copy-paste via Slack/Teams/email).

---

## Admin Portal UI

### Sidebar → Admin Panel (expanded)

```
┌─────────────────────────────┐
│ Admin Panel                 │
├─────────────────────────────┤
│ [Users] [Invites] [Audit]   │  ← tab selector
│                             │
│ ── Users Tab ──             │
│ ┌───────┬──────┬───┬──────┐│
│ │ User  │ Role │ $ │ Act  ││
│ ├───────┼──────┼───┼──────┤│
│ │brandon│admin │20 │ ✓    ││
│ │john   │user  │ 5 │ ✓    ││
│ │sarah  │user  │10 │ ✗    ││
│ └───────┴──────┴───┴──────┘│
│                             │
│ [+ Add User]  [Send Invite] │
│                             │
│ ── Edit User ──             │
│ Username: [________]        │
│ Display:  [________]        │
│ Email:    [________]        │
│ Role:     [admin ▾]         │
│ Credits:  [10     ]         │
│ [Save] [Deactivate]        │
│                             │
│ ── Invites Tab ──           │
│ ┌───────┬────────┬────────┐│
│ │ Email │ Status │ Sent   ││
│ ├───────┼────────┼────────┤│
│ │a@b.com│Pending │ 3/4    ││
│ │c@d.com│Accepted│ 3/1    ││
│ └───────┴────────┴────────┘│
│                             │
│ ── Audit Tab ──             │
│ 2026-03-04 brandon created  │
│   user "sarah"              │
│ 2026-03-04 brandon invited  │
│   a@b.com                   │
│ 2026-03-03 brandon reset    │
│   credits for "john"        │
└─────────────────────────────┘
```

---

## Implementation Components

### `user_management.py` (new module)

Contains all user management logic, decoupled from Streamlit:

- `create_tables(conn)` — idempotent DDL
- `get_user(conn, username)` → dict or None
- `list_users(conn)` → list[dict]
- `create_user(conn, username, password, email, role, credits_per_week)` → dict
- `update_user(conn, username, **fields)` → dict
- `deactivate_user(conn, username)` → bool
- `activate_user(conn, username)` → bool
- `verify_user(conn, username, password)` → dict or None
- `create_invite(conn, email, role, credits_per_week, invited_by)` → token
- `accept_invite(conn, token, username, password)` → dict or None
- `list_invites(conn)` → list[dict]
- `log_audit(conn, actor, action, target_user, details)` → None
- `get_audit_log(conn, limit=50)` → list[dict]
- `send_invite_email(email, token, inviter, app_url, smtp_cfg)` → bool

### `app.py` Changes

- Import from `user_management.py`
- Update `check_password()` to query DB first, secrets.toml as fallback
- Replace sidebar admin panel with tabbed UI (Users / Invites / Audit)
- Add invite signup page (shown when `?invite=TOKEN` query param present)
- Add "Add User" form for direct user creation
- Add "Send Invite" form with email input

---

## Security Considerations

- Invite tokens: 32-byte `secrets.token_urlsafe()` — 256 bits of entropy
- Tokens expire after 7 days, single-use
- Passwords: PBKDF2-SHA256, 260K iterations, 16-byte random salt (existing scheme)
- Admin actions logged in `user_audit_log`
- Email addresses stored but not displayed to non-admin users
- SMTP password stored in Streamlit secrets (encrypted at rest on Streamlit Cloud)

---

## Backward Compatibility

During transition:
1. `check_password()` checks DB `users` table first
2. If no match, falls back to `st.secrets["users"]` (existing behavior)
3. Admins can migrate secrets.toml users to DB via "Import from config" button
4. Once all users migrated, secrets.toml `[users.*]` sections can be removed

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| SMTP not configured | Show copyable invite link as fallback |
| DB connection lost | Fall back to secrets.toml auth |
| Token brute force | 256-bit tokens + rate limiting via Streamlit |
| Admin locks themselves out | secrets.toml admin always works as backdoor |
| Email delivery failures | Log failures, show admin toast notification |
