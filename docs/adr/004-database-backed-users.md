# ADR-004: Database-Backed User Management

**Status:** Accepted
**Date:** 2026-03-04
**Author:** @brandon

## Context

Users are currently defined in `.streamlit/secrets.toml` as static entries with
pre-computed password hashes. Adding or modifying users requires editing a config
file, running a CLI hash command, and redeploying. This blocks team-wide adoption
(Goal G2) because onboarding a new analyst requires developer intervention.

The credit system already uses Neon Postgres (`credit_usage` table), so the
database infrastructure exists.

## Decision

Move user definitions from `secrets.toml` to Postgres tables (`users`, `invites`,
`user_audit_log`). Provide admin CRUD via the Streamlit UI and an email invite
flow for self-service onboarding.

Keep `secrets.toml` as a fallback authentication source so that at least one admin
can always log in, even if the database is unavailable.

## Consequences

**Easier:**
- Adding/removing users (no redeploy needed)
- Onboarding new team members (email invite, self-service signup)
- Auditing user lifecycle events
- Deactivating users without deleting their credit history

**Harder:**
- Initial migration of existing users from secrets.toml to DB
- Slightly more complex auth flow (DB check + fallback)
- SMTP configuration required for email invites (optional — link fallback exists)
