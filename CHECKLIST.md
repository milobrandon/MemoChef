# Memo Automator - Execution Checklist

> **Last updated:** 2026-03-06
> **Tracking:** Check boxes as tasks complete. Add dates and notes in the rightmost column.

---

## Phase 0: Foundation (Documentation & Scaffolding)

> **Goal:** Dev-readiness. No business logic changes. All docs and scaffolding.

- [x] Create `ROADMAP.md` - @brandon - 1d - _No deps_ - Done 2026-03-01
- [x] Create `CHECKLIST.md` (this file) - @brandon - 0.5d - _Depends: ROADMAP_ - Done 2026-03-01
- [x] Create `docs/CONTRIBUTING.md` - @brandon - 0.5d - _No deps_ - Done 2026-03-01
- [x] Create `docs/SECURITY.md` - @brandon - 0.5d - _No deps_ - Done 2026-03-01
- [x] Create `docs/RUNBOOK.md` - @brandon - 0.5d - _No deps_ - Done 2026-03-02
- [x] Create `docs/DATA_SOURCES.md` - @brandon - 1d - _No deps_ - Done 2026-03-02
- [x] Create `docs/adr/README.md` (ADR index + template) - @brandon - 0.25d - _No deps_ - Done 2026-03-02
- [x] Create `docs/adr/001-monolith-to-packages.md` - @brandon - 0.25d - _No deps_ - Done 2026-03-02
- [x] Create `docs/adr/002-claude-model-selection.md` - @brandon - 0.25d - _No deps_ - Done 2026-03-02
- [x] Create `docs/adr/003-market-data-approach.md` - @brandon - 0.25d - _No deps_ - Done 2026-03-02
- [x] Update `.gitignore` - @brandon - 0.25d - _No deps_ - Done 2026-03-02
- [x] Update `README.md` with team onboarding section - @brandon - 1d - _Depends: CONTRIBUTING_ - Done 2026-03-02

---

## Phase 1: MVP Polish (Weeks 1-2)

> **Goal:** Stabilize for confident single-user use. Testing and logging.

### Logging & Error Handling

- [x] Replace `print()` with Python `logging` module - @developer - 2d - _No deps_ - Done 2026-03-02
- [x] Add configurable log levels (DEBUG/INFO/WARNING/ERROR) - @developer - 0.5d - _Depends: logging_ - Done 2026-03-02
- [x] Add `--verbose` / `--quiet` CLI flags - @developer - 0.5d - _Depends: logging_ - Done 2026-03-02
- [x] Add graceful error messages for common failures - @developer - 2d - _Depends: logging_ - Done 2026-03-03
  - [x] Missing/invalid API key
  - [x] Malformed Excel (missing tabs, uncached formulas)
  - [x] Malformed PPTX (corrupt, password-protected)
  - [x] API timeout / rate limit exceeded
  - [x] Disk full / permission denied on output

### Input Validation

- [x] Validate file extensions (.pptx, .xlsx/.xlsm, .mpp) - @developer - 0.5d - _No deps_ - Done 2026-03-02
- [x] Validate proforma tab names exist before extraction - @developer - 0.5d - _No deps_ - Done 2026-03-02
- [x] Validate non-empty data extraction (warn if 0 rows) - @developer - 0.5d - _Depends: tab validation_ - Done 2026-03-02
- [x] Validate config.yaml schema on load - @developer - 1d - _No deps_ - Done 2026-03-02

### Testing

- [x] Set up pytest + conftest.py with fixtures - @developer - 1d - _No deps_ - Done 2026-03-02
- [x] Create anonymized test fixtures (mini proforma + memo) - @developer - 1d - _No deps_ - Done 2026-03-03
- [x] Unit tests: `extract_proforma()` - @developer - 1d - _Depends: fixtures_ - Done 2026-03-03
- [x] Unit tests: `extract_memo()` - @developer - 1d - _Depends: fixtures_ - Done 2026-03-03
- [x] Unit tests: `apply_updates()` (table + text) - @developer - 1d - _Depends: fixtures_ - Done 2026-03-03
- [x] Unit tests: `apply_branding()` - @developer - 1d - _Depends: fixtures_ - Done 2026-03-03
- [x] Unit tests: `normalize_layout()` - @developer - 0.5d - _Depends: fixtures_ - Done 2026-03-03
- [x] Unit tests: JSON parsing / truncation recovery - @developer - 0.5d - _No deps_ - Done 2026-03-03
- [x] Integration test: full pipeline with mocked Claude - @developer - 2d - _Depends: all unit tests_ - Done 2026-03-03
- [x] Migrate `test_row_inserts.py` to pytest - @developer - 0.5d - _Depends: pytest setup_ - Done 2026-03-02
- [x] Migrate `test_schedule.py` to pytest - @developer - 0.5d - _Depends: pytest setup_ - Done 2026-03-02

### Cleanup

- [x] Archive `v1/` directory (move or .gitignore) - @developer - 0.25d - _No deps_ - Done 2026-03-03 (N/A: no `v1/` directory present)
- [x] Clean up `a. Sandbox/` artifacts - @developer - 0.25d - _No deps_ - Done 2026-03-03 (`.gitignore` + no tracked artifacts)
- [x] Clean up `x. Old/` scripts - @developer - 0.25d - _No deps_ - Done 2026-03-03 (`.gitignore` + no tracked artifacts)
- [x] Remove or .gitignore `memo_automator_app_sandbox/` - @developer - 0.25d - _No deps_ - Done 2026-03-03

---

## Phase 2: v1.0 - Team Ready (Weeks 3-6)

> **Goal:** Multi-user, CI-enabled, properly structured codebase.

### Refactoring

- [ ] Create `memo_automator/` package structure - @developer - 2d - _Depends: MVP tests passing_
  - [ ] `extraction/proforma.py`
  - [ ] `extraction/memo.py`
  - [ ] `extraction/schedule.py`
  - [ ] `ai/client.py`
  - [ ] `ai/mapping.py`
  - [ ] `ai/validation.py`
  - [ ] `formatting/updates.py`
  - [ ] `formatting/branding.py`
  - [ ] `formatting/layout.py`
  - [ ] `output/changelog.py`
  - [ ] `output/artifacts.py`
  - [ ] `cli.py`
  - [ ] `config.py`
  - [ ] `pipeline.py`
- [x] Create `memo_chef/` package (pipeline wrapper + pydantic models + theme) - @developer - Done 2026-03-05
  - [x] `memo_chef/models.py` - RunRequest, RunManifest, RunResult, StageUpdate pydantic models
  - [x] `memo_chef/pipeline.py` - full pipeline orchestration with retry + checkpoint
  - [x] `memo_chef/theme.py` - Subtext brand CSS + UI helpers
- [x] Extract service layer to `app_services.py` (DB, credits, runs, jobs, profiles) - @developer - Done 2026-03-05
- [x] Extract prompt templates to `prompts/` directory - @developer - 2d - _Depends: package structure_ - Done 2026-03-03
- [ ] Add type hints to all public functions - @developer - 3d - _Depends: package structure_
- [x] Ensure all tests pass post-refactor - @developer - 2d - _Depends: refactor_ - Done 2026-03-06 (66 passed, 1 skipped)

### CI/CD

- [x] Set up GitHub Actions workflow (lint + test on push) - @devops - 1d - _Depends: tests_ - Done 2026-03-03
- [x] Add pre-commit hooks: ruff, black - @devops - 0.5d - _No deps_ - Done 2026-03-03
- [ ] Add pre-commit hook: mypy (type checking) - @devops - 0.5d - _Depends: type hints_
- [x] Set up test coverage reporting - @devops - 0.5d - _Depends: CI_ - Done 2026-03-03
- [ ] Add branch protection rules (require CI pass for merge) - @devops - 0.25d - _Depends: CI_

### Reliability

- [x] API retry logic with exponential backoff - @developer - Done 2026-03-05 (`memo_chef/pipeline.py` `_retry()`)
- [x] Checkpoint/resume for interrupted runs - @developer - Done 2026-03-05 (`memo_chef/pipeline.py` `CheckpointManager`)
- [x] Config validation with pydantic models - @developer - 2d - _Depends: config.py_ - Done 2026-03-03
- [x] Per-project config profiles (`configs/projectA.yaml`) - @developer - 1d - _Depends: config validation_ - Done 2026-03-05

### Team Features

- [x] Streamlit session state for multi-user support - @developer - 2d - _No deps_ - Done 2026-03-05 (session-backed auth + artifact persistence)
- [x] API cost tracking and per-run cost display - @developer - 1d - _Depends: ai/client.py_ - Done 2026-03-05 (run manifest + UI cost metrics)
- [x] Deployment guide (Streamlit Cloud or internal) - @devops - 2d - _No deps_ - Done 2026-03-03
- [x] Team training documentation / walkthrough - @brandon - 2d - _Depends: v1.0 features_ - Done 2026-03-03 (`docs/HOW_TO.md`)

---

## Phase 3: v1.5 - Market Data & Intelligence (Weeks 7-12)

> **Goal:** External data enrichment, batch processing, analytics.

### Market Data

- [ ] Define final market data schema (JSON Schema or pydantic) - @brandon - 2d - _No deps_
- [ ] Build ingestion pipeline framework - @developer - 3d - _Depends: schema_
- [ ] Integrate FRED API (macro indicators) - @developer - 2d - _Depends: ingestion pipeline_
- [ ] Integrate Census/ACS API (demographics) - @developer - 2d - _Depends: ingestion pipeline_
- [ ] Evaluate CoStar/Yardi Matrix API access - @brandon - 2d - _No deps_
- [ ] Commercial data source integration (if licensed) - @developer - 5d - _Depends: license confirmed_
- [ ] Data validation + staleness checks - @developer - 2d - _Depends: ingestion_
- [ ] Extend mapping prompt for market context - @developer - 2d - _Depends: market data available_

### Batch & Analytics

- [ ] Batch processing mode (multi-memo queue) - @developer - 5d - _Depends: v1.0 stable_
- [ ] Run history storage (SQLite or JSON) - @developer - 3d - _Depends: batch mode_
- [ ] Run comparison dashboard - @developer - 3d - _Depends: run history_
- [ ] Accuracy metrics + confidence scoring - @developer - 3d - _Depends: run history_
- [ ] API cost optimization (prompt caching, smart batching) - @developer - 3d - _Depends: cost tracking_

---

## Legend

| Symbol | Meaning |
|--------|---------|
| `[x]` | Complete |
| `[ ]` | Not started |
| `@owner` | Responsible person (placeholder) |
| `2d` | Estimated effort in days |
| _Depends: X_ | Must wait for task X to complete |
| _No deps_ | Can start immediately |
