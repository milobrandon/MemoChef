# Memo Automator

Memo Automator updates Investment Committee (IC) PowerPoint memos from Excel proformas with a two-pass Claude workflow:
1. Mapping pass: identify candidate metric updates.
2. Validation pass: reject/correct mismatches and flag misses.

It preserves deck formatting, creates a backup, and produces a full audit trail.

## Quick Links

- How-to guide: `docs/HOW_TO.md`
- Roadmap: `ROADMAP.md`
- Contributing: `docs/CONTRIBUTING.md`
- Runbook: `docs/RUNBOOK.md`
- Security: `docs/SECURITY.md`

## What It Produces

- `*_BACKUP_<timestamp>.pptx`
- `proforma_extract.txt`
- `memo_extract.txt`
- `mappings_raw.json`
- `mappings_validated.json`
- `CHANGE_LOG.md` (includes run telemetry)
- `run_manifest.json` (stage timings, warnings, counts, output paths)

## Prerequisites

- Python 3.9+
- Anthropic API key

Install:

```bash
pip install -r requirements.txt
```

## Setup

1. Copy env template:

```bash
cp .env.example .env
```

2. Set key in `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-your-key
```

3. Review `config.yaml`.

## CLI Usage

Basic:

```bash
python memo_automator.py memo.pptx proforma.xlsm
```

Dry-run:

```bash
python memo_automator.py memo.pptx proforma.xlsm --dry-run
```

Common options:

- `--config / -c`
- `--output-dir / -o`
- `--skip-validation`
- `--property-name`
- `--verbose / -v`
- `--quiet / -q`

## Streamlit Usage

```bash
streamlit run app.py
```

The Streamlit app now includes:
- a modernized `New Run` workflow
- `Run History` for recent executions
- an `Operations` console for queueing and health checks
- downloadable `run_manifest.json` artifacts
- saved run profiles and approval tracking
- persistent job records and stored run artifacts for reruns
- admin controls for users, credits, and recent activity

## Testing and Quality

```bash
pytest -q
ruff check .
pytest -m "not integration" --cov=memo_automator --cov=app_helpers --cov-report=term-missing
```

## Prompt Templates

Prompt templates are versioned in:

- `prompts/mapping_v1.txt`
- `prompts/validation_v1.txt`

## CI and Security Guardrails

- GitHub Actions workflow: `.github/workflows/ci.yml`
- Lint + tests + coverage gate on push/PR
- Secret scanning in CI (Gitleaks)
- Local pre-commit hooks in `.pre-commit-config.yaml`

## Architecture Notes

The current core remains a monolith (`memo_automator.py`) with staged refactor plans in `ROADMAP.md`.

ADRs:
- `docs/adr/001-monolith-to-packages.md`
- `docs/adr/002-claude-model-selection.md`
- `docs/adr/003-market-data-approach.md`
- `docs/adr/004-externalize-prompt-templates.md`
- `docs/adr/005-pydantic-config-validation.md`
- `docs/adr/006-ci-quality-gates-and-secret-scanning.md`
