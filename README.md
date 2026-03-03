# Memo Automator

Automatically updates an Investment Committee (IC) PowerPoint memo with the latest metrics from an Excel proforma. Uses two Claude API calls — one for metric mapping, one for QA validation — to intelligently identify and verify every update across the entire deck.

## What It Does

1. **Creates a backup** of the original memo (timestamped copy).
2. **Extracts data** from specified proforma tabs using `openpyxl` (reads cached formula values). By default extracts from: Executive Summary, Development Summary, Cash Flow, Assumptions, and Comparison.
3. **Extracts content** from every slide in the memo using `python-pptx` (text boxes, tables).
4. **Claude API call #1 — Metric Mapping**: Sends both datasets to Claude, which identifies every metric in the memo that maps to a proforma value and returns structured JSON with exact old/new text replacements.
5. **Claude API call #2 — Validation**: A second Claude call cross-checks the proposed updates: verifies old values match exactly, new values are correctly formatted, catches duplicates, and flags any missed metrics.
6. **Applies text and table updates** to the memo, preserving all existing formatting (fonts, colors, sizes).
7. **Writes a change log** (Markdown) documenting every modification, rejected updates, and potentially missed metrics.

## Prerequisites

- **Python 3.9+**
- An **Anthropic API key** ([get one here](https://console.anthropic.com/))

### Python Dependencies

Install with pip:

```bash
pip install -r requirements.txt
```

Required packages:
| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API client for metric mapping and validation |
| `python-pptx` | Read/write PowerPoint presentations |
| `openpyxl` | Read Excel proforma data (cached values) |
| `python-dotenv` | Load API key from `.env` file |
| `pyyaml` | Parse `config.yaml` settings |

## Setup

### 1. API Key

Copy the example environment file and add your Anthropic API key:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your actual key:

```
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

### 2. Configuration

Edit `config.yaml` to match your project. The defaults work for standard IC memos, but you can adjust:

```yaml
proforma:
  tabs:                        # Excel tabs to extract data from
    - "Executive Summary"
    - "Development Summary"
    - "Cash Flow"
    - "Assumptions"
    - "Comparison"
  max_rows_per_tab: 250        # 0 = read all rows
  max_cols_per_tab: 30         # 0 = read all columns

memo:
  # Scan all slides. Or use a list: [3, 4, 6, 10, 11, 12, 18, 19, 20]
  pages: "all"

claude:
  model: "claude-sonnet-4-20250514"    # or claude-opus-4-20250514 for higher accuracy
  max_tokens: 16000
  temperature: 0
```

## Usage

### Basic

```bash
python memo_automator.py "path/to/memo.pptx" "path/to/proforma.xlsm"
```

### With Options

```bash
# Use a custom config file
python memo_automator.py memo.pptx proforma.xlsm --config my_project.yaml

# Output artifacts to a specific directory
python memo_automator.py memo.pptx proforma.xlsm --output-dir ./output

# Preview changes without modifying the memo
python memo_automator.py memo.pptx proforma.xlsm --dry-run
```

### All Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `memo` | Yes | Path to the PowerPoint memo (`.pptx`) |
| `proforma` | Yes | Path to the Excel proforma (`.xlsx` / `.xlsm`) |
| `--config`, `-c` | No | Path to YAML config (default: `config.yaml` beside the script) |
| `--output-dir`, `-o` | No | Directory for output files (default: same folder as memo) |
| `--dry-run` | No | Show what would change without modifying files |

## Inputs

The script expects two input files:

1. **PowerPoint memo** (`.pptx`) — The IC memo containing metrics, tables, and data pages to be updated.
2. **Excel proforma** (`.xlsx` or `.xlsm`) — The source of truth for all financial metrics. The workbook must have been opened and saved in Excel at least once so that formula values are cached (required by `openpyxl` with `data_only=True`).

## Output Artifacts

After a run, the following files are produced:

| File | Description |
|------|-------------|
| `*_BACKUP_<timestamp>.pptx` | Timestamped backup of the original memo |
| `proforma_extract.txt` | Raw text extraction of proforma data (for debugging) |
| `memo_extract.txt` | Raw text extraction of memo content (for debugging) |
| `mappings_raw.json` | First Claude API response — all proposed metric mappings |
| `mappings_validated.json` | Second Claude API response — validated mappings, rejections, missed items |
| `CHANGE_LOG.md` | Full record of every change applied, rejected updates, and missed metrics |

## How the Claude API Is Used

The script makes **two API calls**, each replacing a step that would normally require human analysis:

1. **Mapping call** — Claude reads the full proforma data and full memo content, then reasons about which memo values correspond to which proforma cells. It returns structured JSON with exact text replacements, preserving dollar signs, commas, decimal precision, and percentage formatting.

2. **Validation call** — Claude reviews the proposed changes against both source documents. It rejects updates where the `old_value` doesn't exactly match the memo text, flags formatting inconsistencies, removes duplicates, and identifies any metrics that were missed in the first pass.

## Notes

- **Full-deck scanning**: By default the script scans every slide in the deck, so no metrics are missed regardless of which page they appear on.
- **Formatting preservation**: Text and table updates are applied at the run level within `python-pptx`, so fonts, colors, and sizes are preserved.
- **Dry run**: Use `--dry-run` to preview all changes before committing them. No files are modified in dry-run mode.
- **Auditability**: The `CHANGE_LOG.md`, `mappings_raw.json`, and `mappings_validated.json` provide a full audit trail of what changed and why.
- **Proforma caching**: If `openpyxl` returns `None` for formula cells, open the proforma in Excel, wait for it to calculate, save, and close before running the script.

## Team Onboarding

New to the project? Here's how to get productive quickly.

### 1. Read These First

| Document | What You'll Learn |
|----------|------------------|
| This README | How the tool works and how to run it |
| [ROADMAP.md](ROADMAP.md) | Project vision, architecture, phased plan |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Code style, branching, PR process |
| [docs/SECURITY.md](docs/SECURITY.md) | API key handling, data sensitivity |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Deployment, key rotation, troubleshooting |

### 2. Set Up Your Environment

```bash
git clone https://github.com/milobrandon/MemoChef.git
cd MemoChef
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # add your ANTHROPIC_API_KEY
```

### 3. Run the Tests

```bash
pytest                       # unit tests (no API key needed)
pytest -m "not integration"  # skip integration tests
```

### 4. Try a Dry Run

```bash
python memo_automator.py memo.pptx proforma.xlsm --dry-run
```

This previews all changes without modifying any files. Review the `CHANGE_LOG.md` output to understand what the tool does.

### 5. Architecture Overview

The codebase is currently a single module (`memo_automator.py`, ~2,500 lines) with clearly separated sections. See [ROADMAP.md](ROADMAP.md) Section 3 for the current architecture diagram and the planned package refactor.

Key decision records:
- [ADR-001](docs/adr/001-monolith-to-packages.md) — Why the monolith hasn't been split yet
- [ADR-002](docs/adr/002-claude-model-selection.md) — Claude model selection strategy
- [ADR-003](docs/adr/003-market-data-approach.md) — Market data integration approach
