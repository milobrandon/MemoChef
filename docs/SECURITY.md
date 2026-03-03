# Security Policy

Memo Automator processes confidential financial data. Treat all source files and outputs as sensitive.

## Secrets handling

- Store runtime credentials in `.env` (local) or Streamlit secrets (deployed).
- Never commit real keys, DB URLs, or plaintext credentials.
- Rotate credentials immediately if exposure is suspected.

## Required controls

- `.gitignore` excludes `.env`, deploy secrets files, and uploaded memo/proforma/schedule artifacts.
- CI secret scanning via Gitleaks (`.github/workflows/ci.yml`).
- Local secret checks via `detect-secrets` pre-commit hook.

## If credentials are exposed

1. Revoke/rotate exposed keys immediately.
2. Replace secrets in all deployment targets.
3. Review logs for unauthorized usage.
4. Notify project owner and document incident timeline.

## Data sent to Claude API

- Proforma text extraction
- Memo text extraction
- Optional schedule text extraction

Not sent:
- local filesystem metadata
- local credentials
- raw binary file payloads

## Access control

- CLI: repo + environment access required.
- Streamlit: per-user credentials in secrets.
- Credits and auth state backed by Neon Postgres.

## Fixture policy

- Tests must use synthetic/anonymized fixture data only.
- Never add real client names, addresses, or financials to test assets.

## Dependency hygiene

Run periodically:

```bash
pip audit
pip list --outdated
```

## Reporting security issues

Do not post publicly. Contact `@brandon` directly with:
- affected area
- reproduction details
- potential impact
