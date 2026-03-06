# Memo Automator How-To

This guide is the fastest way to run, debug, and deploy Memo Automator.

## 1. First-Time Setup

1. Install Python 3.9+.
2. Clone the repo and enter it:
   ```bash
   git clone <repo-url>
   cd "g. Memo Automator/v2"
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Configure local secrets:
   ```bash
   copy .env.example .env
   ```
   Set `ANTHROPIC_API_KEY` in `.env`.

## 2. Run the CLI

Basic run:
```bash
python memo_automator.py memo.pptx proforma.xlsm
```

Safe preview (recommended first):
```bash
python memo_automator.py memo.pptx proforma.xlsm --dry-run
```

Useful options:
```bash
python memo_automator.py memo.pptx proforma.xlsm ^
  --config config.yaml ^
  --output-dir output ^
  --property-name "EVER at Reston" ^
  --verbose
```

## 3. Run the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501`, log in with credentials from Streamlit secrets, upload memo/proforma, then click **Generate draft**.

The app now includes:
- `New Run` for the primary workflow
- `Run History` for recent activity and outcomes
- `Operations` for queue execution, health checks, and saved profiles
- an `Admin` tab for user and credit management (admins only)
- downloadable `run_manifest.json` output for each successful run

You can also:
- save common run preferences as named profiles
- add runs to a queue and execute them sequentially
- record approval decisions and reviewer notes per run
- requeue historical runs from stored input artifacts
- inspect persistent jobs and retry failed queue items

## 4. Understand Outputs

Each run produces:
- backup memo (`*_BACKUP_<timestamp>.pptx`)
- `proforma_extract.txt`
- `memo_extract.txt`
- `mappings_raw.json`
- `mappings_validated.json`
- `CHANGE_LOG.md` (includes telemetry: duration and API call counts)
- `run_manifest.json` (stage status, warnings, counts, and artifact locations)

## 5. Debug Fast

If something fails, check in this order:
1. Console logs / Streamlit logs.
2. `CHANGE_LOG.md` for rejected/missed entries and telemetry.
3. `mappings_raw.json` and `mappings_validated.json`.
4. `proforma_extract.txt` to verify source data was actually read.

Common fixes:
- **API key errors**: verify `.env` or Streamlit secrets.
- **Proforma has no values**: open in Excel, recalculate, save, rerun.
- **Rate limit/timeouts**: retry; large runs are batched automatically.
- **Credits unavailable in Streamlit**: use sidebar retry for credits service.

## 6. Test and Quality Gates

Run tests:
```bash
pytest -q
```

Lint:
```bash
ruff check .
```

Coverage run:
```bash
pytest -m "not integration" --cov=memo_automator --cov=app_helpers --cov-report=term-missing
```

## 7. CI and Pre-Commit

CI workflow (`.github/workflows/ci.yml`) runs:
- Ruff
- pytest (non-integration)
- coverage gate
- secret scanning (Gitleaks)

Recommended local hooks:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 8. Prompt Template Workflow

Prompt templates now live in:
- `prompts/mapping_v1.txt`
- `prompts/validation_v1.txt`

When editing prompts:
1. Update template file(s).
2. Run `pytest -q`.
3. Run at least one dry run on representative data.
4. Document behavior changes in PR notes.

## 9. Deploy (Streamlit Cloud)

1. Create app in Streamlit Cloud using `app.py`.
2. Add secrets in dashboard:
   - `ANTHROPIC_API_KEY`
   - `CREDITS_DATABASE_URL`
   - `[users.*]` blocks for auth
3. Push to `main`; Streamlit auto-deploys.
4. Verify login, credit load, and a dry-run execution.

## 10. Security Basics

- Never commit real credentials, memo/proforma files, or private schedules.
- Rotate exposed credentials immediately.
- Keep `.env` local only.
- Run secret scan checks before pushing.
