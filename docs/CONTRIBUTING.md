# Contributing to Memo Automator

## Quick Start

## Prerequisites

- Python 3.9+
- Git
- Anthropic API key (for live runs)
- Java runtime (only for `.mpp` schedule extraction)

## Local setup

```bash
git clone <repo-url>
cd "g. Memo Automator/v2"
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`.

## Run locally

```bash
python memo_automator.py memo.pptx proforma.xlsm --dry-run
streamlit run app.py
```

## Quality checks

```bash
ruff check .
pytest -q
pytest -m "not integration" --cov=memo_automator --cov=app_helpers --cov-report=term-missing
```

## Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Hooks are configured in `.pre-commit-config.yaml` and include Ruff, Black, and detect-secrets.

## Branching and PRs

- `main`: production-ready
- `dev`: integration branch
- feature branches: `feat/<name>`
- bugfix branches: `fix/<name>`

PR checklist:
1. Explain what changed and why.
2. Include test evidence (commands + results).
3. Include screenshots for UI changes.
4. Confirm lint/tests pass locally.

## Prompt workflow

Prompt templates live in:
- `prompts/mapping_v1.txt`
- `prompts/validation_v1.txt`

When changing prompts:
1. Update template file(s).
2. Run tests and lint.
3. Run at least one dry-run check.
4. Document behavior impact in the PR.

## Security rules

- Never commit credentials (`.env`, raw secrets, private DB URLs).
- Never commit real client memo/proforma/schedule files.
- Use synthetic fixtures only (`tests/fixtures/`).
- Run secret checks before opening a PR.
