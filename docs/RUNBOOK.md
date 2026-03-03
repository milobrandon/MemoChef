# Memo Automator Operations Runbook

Procedures for deploying, operating, and troubleshooting Memo Automator.

## Deployment

## Streamlit Cloud

1. Create app in Streamlit Cloud using `app.py`.
2. Configure secrets in app settings:
   - `ANTHROPIC_API_KEY`
   - `CREDITS_DATABASE_URL`
   - `[users.<username>]` blocks (`password_hash`, `role`, `credits_per_week`)
3. Deploy from `main`.
4. Verify login, credits load, and a dry-run execution.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs:
- Ruff lint
- pytest non-integration tests
- coverage gate
- Gitleaks secret scan

## Local development run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Credentials and key rotation

## Anthropic key

1. Create replacement key in Anthropic console.
2. Update local `.env` and Streamlit Cloud secrets.
3. Revoke old key.
4. Run a smoke test.

## Neon DB password

1. Rotate in Neon console.
2. Update `CREDITS_DATABASE_URL` secret.
3. Verify login and credits operations.

## Credits system notes

- Credits are charged only after successful run completion.
- Charge writes are idempotent by `run_id` for each user/week.
- If credits DB is unavailable, UI disables runs and exposes a retry action in sidebar.

## Troubleshooting

## `ANTHROPIC_API_KEY not set`

- Local: confirm `.env` exists and key is set.
- Streamlit: confirm secret exists in app settings.

## Proforma extraction errors

- Ensure workbook is valid `.xlsx/.xlsm`.
- If formulas return empty values, open in Excel, recalculate, save, rerun.
- Verify configured tab names in `config.yaml`.

## Memo/PPTX read errors

- Confirm file is valid `.pptx` and not password-protected/corrupt.

## Rate limits / API timeouts

- Retry after pause.
- Use `--skip-validation` for faster diagnostic runs.
- For large decks, batching is automatic.

## Credits service down

- Use sidebar "Retry credits service".
- Check Neon connectivity and credential validity.

## Monitoring

- Anthropic usage dashboard for API cost/volume.
- Streamlit logs for runtime errors.
- `CHANGE_LOG.md` telemetry for per-run timing and API call counts.
