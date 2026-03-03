# Contributing to Memo Automator

## Getting Started

### Prerequisites

- Python 3.9+
- An Anthropic API key (see [README](../README.md))
- Java Runtime (only if working with schedule `.mpp` files)
- Git

### Local Setup

```bash
# Clone the repo
git clone <repo-url>
cd "g. Memo Automator/v2"

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API key
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

### Running the Tool

```bash
# CLI
python memo_automator.py memo.pptx proforma.xlsm

# Web UI
streamlit run app.py
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest test_row_inserts.py -v

# With coverage
pytest --cov=memo_automator --cov-report=html
```

---

## Code Style

### Python Standards

- **Formatter:** [Black](https://black.readthedocs.io/) (line length 99)
- **Linter:** [Ruff](https://docs.astral.sh/ruff/) with default rules
- **Type checking:** mypy (strict mode, once type hints are added)
- **Docstrings:** Google style for public functions

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Files/modules | `snake_case.py` | `memo_automator.py` |
| Functions | `snake_case` | `extract_proforma()` |
| Classes | `PascalCase` | `MemoConfig` |
| Constants | `UPPER_SNAKE` | `MAPPING_PROMPT` |
| Config keys | `snake_case` | `max_rows_per_tab` |

### Commit Messages

Use conventional commit format:

```
type: short description

Longer explanation if needed.
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `style`

Examples:
```
feat: add schedule extraction from .mpp files
fix: handle unicode smart quotes in table matching
docs: add RUNBOOK.md with troubleshooting guide
refactor: extract branding logic to formatting/branding.py
test: add pytest fixtures for proforma extraction
```

---

## Branching Strategy

```
main          ← production-ready, always passing
├── dev       ← integration branch for features
│   ├── feat/market-data-ingestion
│   ├── feat/batch-processing
│   └── fix/unicode-table-match
```

- **`main`**: Protected. Requires passing CI + review.
- **`dev`**: Integration branch. Features merge here first.
- **Feature branches**: `feat/<description>` off `dev`.
- **Bug fix branches**: `fix/<description>` off `dev` (or `main` for hotfixes).

---

## Pull Request Process

1. Create a feature branch from `dev`.
2. Make changes. Write/update tests.
3. Run tests locally: `pytest`.
4. Run linter: `ruff check .` and `black --check .`.
5. Push and open a PR against `dev`.
6. Fill in the PR template:
   - **What** changed and **why**
   - **Testing** done
   - **Screenshots** if UI changes
7. Request review from at least one team member.
8. Address review feedback.
9. Merge after approval + CI passes.

### PR Size Guidelines

- Prefer small, focused PRs (< 400 lines changed).
- If a change is large, break it into stacked PRs or explain why in the description.
- Docs-only PRs can be merged with a single approval.

---

## Working with Prompts

The Claude prompt templates (`MAPPING_PROMPT`, `VALIDATION_PROMPT`) are critical to the tool's accuracy. Changes to prompts require extra care:

1. **Never modify prompts without testing** on at least 2 different memo/proforma pairs.
2. **Document prompt changes** in the PR description with before/after examples.
3. **Version prompts** — when prompts move to `prompts/` directory, use filenames like `mapping_v2.txt`.
4. **Regression test** — ensure existing test cases still produce correct mappings.

---

## Reporting Issues

Use the project's issue tracker. Include:

1. **What you expected** to happen.
2. **What actually happened** (error message, incorrect output).
3. **Steps to reproduce** (CLI command or UI steps).
4. **Input files** (anonymized if they contain sensitive financial data).
5. **Logs** (`CHANGE_LOG.md`, console output).
6. **Environment** (Python version, OS, package versions).

---

## Security

- **Never commit API keys** or `.env` files.
- **Never commit real proforma/memo files** — use anonymized test fixtures.
- See [SECURITY.md](SECURITY.md) for the full security policy.
