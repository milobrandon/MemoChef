# Tenant Invitation System Design

**Date:** 2026-03-07
**Status:** Approved

## Flow

1. Admin enters email + role + credits in Admin tab, clicks "Send invite"
2. System creates `invitations` row with UUID4 token, sends email via Resend with magic link: `<app-url>?invite=<token>`
3. User clicks link, sees sign-up form (username + password + confirm)
4. Token validated (exists, not expired, not used), account created, token marked accepted
5. Admin sees all invitations with status in "Invitations" sub-tab

## DB Changes

New table `memo_chef_invitations`:
- `id` (UUID4 token), `email`, `role`, `credits_per_week`, `status` (pending/accepted/expired), `invited_by`, `created_at`, `accepted_at`, `expires_at`

Add `email TEXT` column to `memo_chef_users`.

## Email

- Provider: Resend (Python SDK)
- API key in `.streamlit/secrets.toml` as `RESEND_API_KEY`
- From address configurable (default Resend sender for dev)
- Simple HTML email with Subtext branding + CTA button

## Code Changes

- `app_services.py`: `create_invitation()`, `get_invitation()`, `accept_invitation()`, `get_invitations()` + DB migration
- `app.py`: Invite sign-up page (via `st.query_params`), "Invitations" sub-tab in admin
- `requirements.txt`: Add `resend`

## Security

- UUID4 tokens (128-bit random), 48h expiry, single-use
- Password min 6 chars (matching existing)
