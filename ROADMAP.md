# Memo Automator â€” Project Roadmap

> **Last updated:** 2026-03-03
> **Status:** Active development
> **Owner:** @brandon

---

## 0. Recent Progress (2026-03-03)

- Added CI matrix (Windows + Linux) with Ruff, pytest, coverage gate, and secret scanning.
- Added strict pydantic-based config validation with clearer field-level errors.
- Externalized Claude prompts to versioned files in prompts/ with runtime fallback loading.
- Added deployment hardening in Streamlit app (credits idempotency, DB retry path, recoverable credits service outage).
- Added run telemetry output to CHANGE_LOG.md (duration + API call counts).

---

## 1. Product Goals & Scope

### Vision

Transform Memo Automator from a single-analyst productivity tool into a reliable, team-wide platform for updating Investment Committee (IC) PowerPoint memos from Excel proformas â€” with full auditability, consistent branding, and optional market data enrichment.

### Goals

| # | Goal | Success Metric |
|---|------|----------------|
| G1 | **Reduce memo update time** from hours to minutes | < 10 min per memo, including review |
| G2 | **Team-wide adoption** across the acquisitions group | 100% of analysts using the tool within 60 days of v1.0 |
| G3 | **Zero silent errors** in financial metrics | 0 undetected mismatches per memo (validated by QA pass) |
| G4 | **Auditability** for every change | Full change log + JSON audit trail on every run |
| G5 | **Consistent brand identity** on all IC deliverables | Subtext theme applied automatically |
| G6 | **Market data integration** (v1.5) | Comps/market context populated from external sources |

### In Scope

- IC memo updates from Excel proformas (.xlsx/.xlsm)
- Microsoft Project schedule integration (.mpp)
- Subtext brand theme application
- Layout normalization and formatting preservation
- Streamlit web UI + CLI
- Change log generation and audit trail
- Market data integration (v1.5)
- Multi-project batch runs (v1.5)

### Out of Scope (for now)

- Memo creation from scratch (green-field generation)
- Non-PowerPoint output formats (Word, PDF)
- Real-time collaboration / multi-user editing
- CRM or deal pipeline integration
- Automated IC scheduling or distribution

---

## 2. User Roles & Workflows

### Roles

| Role | Description | Primary Actions |
|------|-------------|-----------------|
| **Analyst** | Builds proformas, runs the tool, reviews output | Upload files, run automation, review change log, fix edge cases |
| **Associate / VP** | Reviews memos before IC presentation | Inspect change log, spot-check metrics, request re-runs |
| **MD / Partner** | Consumes final memo at IC | Confidence that numbers are correct and branded |
| **Admin** | Manages deployment, API keys, access | Rotate keys, monitor usage/costs, manage config |

### Workflow: Analyst (Primary)

```
1. Finalize proforma in Excel (save to cache formulas)
2. Open Memo Automator (web UI or CLI)
3. Upload: memo.pptx + proforma.xlsm + [optional schedule.mpp]
4. Set property name (if rebranding) and config overrides
5. Click "Run" (or execute CLI command)
6. Review change log:
   - Verify applied changes make sense
   - Check rejected items â€” fix proforma if needed
   - Review missed metrics â€” update manually if critical
7. Open updated memo in PowerPoint for final visual QC
8. Send to Associate/VP for review
```

### Workflow: Associate / VP (Reviewer)

```
1. Receive updated memo + change log from Analyst
2. Review CHANGE_LOG.md:
   - Spot-check 5-10 key metrics against proforma
   - Verify rejected items were intentional
   - Flag any missed metrics
3. Open memo in PowerPoint, check formatting
4. Approve or send back for revisions
```

### Workflow: Admin

```
1. Manage ANTHROPIC_API_KEY rotation (quarterly)
2. Monitor API spend via Anthropic dashboard
3. Update config.yaml defaults when proforma structure changes
4. Deploy new versions to team (pip install or Streamlit Cloud)
5. Respond to issues logged in GitHub/shared tracker
```

---

## 3. Architecture & Tech Debt

### Current Architecture

```
memo_automator.py (2,532 lines â€” monolith)
â”œâ”€â”€ extract_proforma()      â€” openpyxl data extraction
â”œâ”€â”€ extract_schedule()      â€” mpxj/jpype schedule parsing
â”œâ”€â”€ extract_memo()          â€” python-pptx content extraction
â”œâ”€â”€ call_claude_mapping()   â€” Claude API: metric identification
â”œâ”€â”€ call_claude_validation()â€” Claude API: QA cross-check
â”œâ”€â”€ apply_updates()         â€” PPTX text/table modifications
â”œâ”€â”€ apply_branding()        â€” theme + font + color reformat
â”œâ”€â”€ normalize_layout()      â€” title/margin/TOC alignment
â””â”€â”€ write_change_log()      â€” Markdown audit trail

app.py (560 lines)
â””â”€â”€ Streamlit web dashboard ("Memo Chef")
```

### Tech Debt Register

| ID | Debt Item | Severity | Effort | Notes |
|----|-----------|----------|--------|-------|
| TD-1 | **Monolithic module** â€” 2,500+ lines in one file | High | Medium | Split into packages: `extraction/`, `ai/`, `formatting/`, `output/` |
| TD-2 | **No test suite** â€” only 2 ad-hoc test scripts | High | Medium | Need pytest suite with fixtures, mocking, CI integration |
| TD-3 | **No CI/CD pipeline** | Medium | Low | GitHub Actions for lint + test on push |
| TD-4 | **Print-based logging** â€” no structured logging | Medium | Low | Replace with Python `logging` module, configurable levels |
| TD-5 | **Hardcoded prompt strings** â€” 600+ line prompts inline | Medium | Medium | Extract to `prompts/` directory as versioned templates |
| TD-6 | **No input validation** â€” trusts file formats | Medium | Low | Validate file extensions, sheet names, non-empty data |
| TD-7 | **No error recovery** â€” crash on API timeout/failure | Medium | Medium | Retry logic with exponential backoff, checkpoint/resume |
| TD-8 | **Magic numbers** â€” thresholds scattered in code | Low | Low | Move to config.yaml with documentation |
| TD-9 | **No type hints** â€” limited IDE support | Low | Low | Add type annotations incrementally |
| TD-10 | **Sandbox/test artifacts in repo** â€” `a. Sandbox/`, `x. Old/` | Low | Low | Clean up, .gitignore, or archive |

### Proposed Architecture (v1.0+)

```
memo_automator/
â”œâ”€â”€ __init__.py
â”œâ”€â”€ cli.py                  # CLI entry point (argparse)
â”œâ”€â”€ config.py               # Config loading + validation
â”œâ”€â”€ pipeline.py             # Orchestrator (step sequencing)
â”œâ”€â”€ extraction/
â”‚   â”œâ”€â”€ proforma.py         # Excel data extraction
â”‚   â”œâ”€â”€ memo.py             # PPTX content extraction
â”‚   â””â”€â”€ schedule.py         # MPP schedule parsing
â”œâ”€â”€ ai/
â”‚   â”œâ”€â”€ client.py           # Claude API wrapper (retry, rate limit)
â”‚   â”œâ”€â”€ mapping.py          # Metric mapping logic
â”‚   â””â”€â”€ validation.py       # QA validation logic
â”œâ”€â”€ formatting/
â”‚   â”œâ”€â”€ updates.py          # Apply text/table changes
â”‚   â”œâ”€â”€ branding.py         # Theme + font + color
â”‚   â””â”€â”€ layout.py           # Layout normalization
â”œâ”€â”€ output/
â”‚   â”œâ”€â”€ changelog.py        # Change log generation
â”‚   â””â”€â”€ artifacts.py        # JSON/debug file output
â”œâ”€â”€ prompts/
â”‚   â”œâ”€â”€ mapping_v1.txt      # Mapping prompt template
â”‚   â””â”€â”€ validation_v1.txt   # Validation prompt template
â””â”€â”€ market/                  # v1.5: Market data module
    â”œâ”€â”€ sources.py
    â”œâ”€â”€ schema.py
    â””â”€â”€ ingestion.py
```

---

## 4. Implementation Plan â€” Phased

### Phase 0: Foundation (Current Sprint)

**Goal:** Documentation, scaffolding, and dev-readiness without touching business logic.

| Task | Owner | Est. | Depends On |
|------|-------|------|------------|
| Create ROADMAP.md | @brandon | 1d | â€” |
| Create CHECKLIST.md | @brandon | 0.5d | ROADMAP |
| Create CONTRIBUTING.md | @brandon | 0.5d | â€” |
| Create SECURITY.md | @brandon | 0.5d | â€” |
| Create RUNBOOK.md | @brandon | 0.5d | â€” |
| Create DATA_SOURCES.md | @brandon | 1d | â€” |
| Create initial ADRs | @brandon | 0.5d | â€” |
| Set up docs/ folder structure | @brandon | 0.5d | â€” |
| Add .gitignore improvements | @brandon | 0.25d | â€” |

### Phase 1: MVP Polish (Weeks 1-2)

**Goal:** Stabilize current functionality for confident single-user use.

| Task | Owner | Est. | Depends On |
|------|-------|------|------------|
| Add structured logging (replace print) | @developer | 2d | â€” |
| Add input file validation | @developer | 1d | â€” |
| Add proper error messages + graceful failures | @developer | 2d | Logging |
| Write pytest unit tests for extraction | @developer | 3d | â€” |
| Write pytest unit tests for update application | @developer | 2d | â€” |
| Create sample test fixtures (anonymized) | @developer | 1d | â€” |
| Add `--verbose` / `--quiet` CLI flags | @developer | 0.5d | Logging |
| Clean up sandbox/old artifacts | @developer | 0.5d | â€” |
| Update README with team onboarding steps | @brandon | 1d | â€” |

### Phase 2: v1.0 â€” Team Ready (Weeks 3-6)

**Goal:** Multi-user, tested, CI-enabled, documented.

| Task | Owner | Est. | Depends On |
|------|-------|------|------------|
| Refactor into package structure | @developer | 5d | MVP tests passing |
| Extract prompts to versioned template files | @developer | 2d | Package refactor |
| Set up GitHub Actions CI (lint + test) | @devops | 2d | Tests exist |
| Add pre-commit hooks (ruff, black, mypy) | @devops | 1d | CI |
| Add type hints to public interfaces | @developer | 3d | Package refactor |
| API retry logic with exponential backoff | @developer | 2d | Package refactor |
| Checkpoint/resume for interrupted runs | @developer | 3d | Retry logic |
| Config validation + schema (pydantic) | @developer | 2d | Package refactor |
| Per-project config profiles | @developer | 1d | Config validation |
| Streamlit multi-user support (session state) | @developer | 2d | â€” |
| API cost tracking + logging | @developer | 1d | Logging |
| Deployment guide (Streamlit Cloud or internal) | @devops | 2d | â€” |
| Team training documentation | @brandon | 2d | v1.0 features |

### Phase 3: v1.5 â€” Market Data & Intelligence (Weeks 7-12)

**Goal:** Enrich memos with external market data; batch processing.

| Task | Owner | Est. | Depends On |
|------|-------|------|------------|
| Define market data schema (see DATA_SOURCES.md) | @brandon | 2d | â€” |
| Build market data ingestion pipeline | @developer | 5d | Schema |
| CoStar/Yardi Matrix API integration | @developer | 5d | Ingestion pipeline |
| Market data validation + freshness checks | @developer | 3d | Ingestion |
| Extend mapping prompt for market context | @developer | 2d | Market data available |
| Batch processing mode (multi-memo) | @developer | 5d | v1.0 stable |
| Run history + comparison dashboard | @developer | 5d | Batch mode |
| Accuracy metrics + confidence scoring | @developer | 3d | Run history |
| API cost optimization (caching, smart batching) | @developer | 3d | Cost tracking |

---

## 5. Required Project Documents

| Document | Purpose | Status |
|----------|---------|--------|
| `README.md` | User-facing setup + usage guide | Exists (update for team) |
| `ROADMAP.md` | This document â€” project vision + phases | **New** |
| `CHECKLIST.md` | Execution plan with task tracking | **New** |
| `docs/CONTRIBUTING.md` | How to contribute, code style, PR process | **New** |
| `docs/SECURITY.md` | API key handling, data sensitivity, access control | **New** |
| `docs/DATA_SOURCES.md` | Market data sources, schema, ingestion, licensing | **New** |
| `docs/RUNBOOK.md` | Operational procedures, troubleshooting, incident response | **New** |
| `docs/adr/001-monolith-to-packages.md` | ADR: Refactoring from single file to package | **New** |
| `docs/adr/002-claude-model-selection.md` | ADR: Why Sonnet for mapping, model choice rationale | **New** |
| `docs/adr/003-market-data-approach.md` | ADR: Market data integration strategy | **New** |
| `docs/adr/README.md` | ADR index and template | **New** |

---

## 6. Market Data Integration

### Sources Under Consideration

| Source | Data Type | Access Method | Cost | Licensing |
|--------|-----------|---------------|------|-----------|
| **CoStar** | Comps, rent surveys, cap rates | API or manual export | $$$ (enterprise license) | Per-seat; no redistribution |
| **Yardi Matrix** | Rent comps, pipeline, occupancy | API or CSV export | $$ (subscription) | Per-seat; restricted use |
| **RCA / MSCI** | Transaction comps, cap rates | API | $$$ (enterprise) | Institutional license |
| **FRED** | Macro indicators (rates, CPI) | Free REST API | Free | Public domain |
| **Census / ACS** | Demographics, housing stats | Free API | Free | Public domain |
| **Zillow / Realtor** | Residential rent indices | ZTRAX or public data | Free-$$ | Varies; check ToS |
| **Internal deal database** | Historical deal metrics | Direct DB/Excel | Free | Internal |

### Ingestion Approach

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Data Source   â”‚â”€â”€â”€â”€â–¶â”‚ Ingestion    â”‚â”€â”€â”€â”€â–¶â”‚ Validated    â”‚
â”‚ (API/CSV/DB) â”‚     â”‚ Pipeline     â”‚     â”‚ Data Store   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â”‚ - Fetch      â”‚     â”‚ (JSON/SQLite)â”‚
                     â”‚ - Normalize  â”‚     â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
                     â”‚ - Validate   â”‚            â”‚
                     â”‚ - Version    â”‚            â–¼
                     â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                          â”‚ Memo Context â”‚
                                          â”‚ Enrichment   â”‚
                                          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Schema (Conceptual)

```yaml
market_data:
  property:
    name: str
    address: str
    submarket: str
    msa: str
    property_type: str      # multifamily, mixed-use, etc.
  rent_comps:
    - comp_name: str
      distance_mi: float
      unit_type: str        # 1BR, 2BR, Studio
      asking_rent: float
      effective_rent: float
      concessions: str
      occupancy_pct: float
      year_built: int
      as_of_date: date
  sale_comps:
    - comp_name: str
      sale_date: date
      sale_price: float
      price_per_unit: float
      cap_rate: float
      units: int
  submarket_stats:
    avg_rent: float
    vacancy_pct: float
    absorption_units: int
    pipeline_units: int
    rent_growth_yoy: float
    as_of_date: date
  macro:
    sofr_rate: float
    treasury_10yr: float
    cpi_yoy: float
    as_of_date: date
```

### Validation Rules

- All monetary values must be positive and within reasonable bounds
- Dates must be within the last 12 months (configurable staleness threshold)
- Occupancy/vacancy must sum to ~100%
- Cap rates must be between 2% and 15%
- Rent values must be > $0 and < $10,000/unit/month
- Source attribution required for every data point

### Update Cadence

| Data Type | Recommended Cadence | Staleness Threshold |
|-----------|--------------------|--------------------|
| Rent comps | Weekly | 30 days |
| Sale comps | Monthly | 90 days |
| Submarket stats | Monthly | 60 days |
| Macro indicators | Daily (automated) | 7 days |
| Demographics | Annually | 365 days |

### Licensing Considerations

- **CoStar/Yardi**: Enterprise agreements typically prohibit redistribution. Tool must not store or cache licensed data beyond the current session unless the license permits it. Check with legal.
- **Public data (FRED, Census)**: Free to use and cache. Attribute source in outputs.
- **Internal data**: No restrictions, but must be versioned and auditable.
- **Recommendation**: Start with free public sources (FRED, Census) + internal deal database. Add commercial sources after license review.

---

## 7. Risks & Open Questions

### Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | **AI hallucination** â€” Claude maps wrong metric or invents values | Low | Critical | Two-stage validation; human review of change log; dry-run default |
| R2 | **API cost escalation** â€” large decks + batching = many API calls | Medium | Medium | Cost tracking, caching, prompt optimization, token budget alerts |
| R3 | **Proforma format drift** â€” teams use different tab/column layouts | High | Medium | Config profiles per project type; validation of expected tabs |
| R4 | **Breaking API changes** â€” Anthropic model updates change behavior | Low | High | Pin model versions in config; regression test suite |
| R5 | **Data sensitivity** â€” proforma data sent to external API | Medium | High | Document data handling in SECURITY.md; consider on-prem options |
| R6 | **Single point of failure** â€” one developer, no bus factor | High | High | Documentation, CONTRIBUTING.md, team training |
| R7 | **Font availability** â€” Pragmatica not installed on all machines | Medium | Low | Document requirement; PowerPoint auto-substitutes |
| R8 | **Market data licensing** â€” commercial data ToS violations | Medium | High | Legal review before integration; start with free sources |
| R9 | **Large file handling** â€” 100+ slide decks with embedded images | Low | Medium | Memory profiling; streaming extraction; page filtering |
| R10 | **Team adoption resistance** â€” analysts prefer manual workflow | Medium | Medium | Training docs, demo sessions, gradual rollout |

### Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| Q1 | Should we deploy on Streamlit Cloud (public) or internal server? | @devops | Open |
| Q2 | What is the API budget per month for team-wide use? | @brandon | Open |
| Q3 | Do we need SOC 2 / compliance review for sending proforma data to Claude? | @legal | Open |
| Q4 | Which commercial market data sources does the firm already license? | @brandon | Open |
| Q5 | Should we support Google Slides output in addition to PPTX? | @brandon | Deferred |
| Q6 | Do we need multi-language support (memos in languages other than English)? | @brandon | Deferred |
| Q7 | Should the tool auto-detect proforma tab names or require explicit config? | @developer | Open |
| Q8 | What is the acceptable error rate? (0 errors vs. 99% accuracy + manual review) | @brandon | Open |
| Q9 | Should we version prompts independently from code releases? | @developer | Open |
| Q10 | Do we need role-based access control in the Streamlit UI? | @brandon | Open |

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **IC memo** | Investment Committee presentation (PowerPoint deck) |
| **Proforma** | Excel financial model with projected cash flows, returns, unit mix |
| **Mapping** | The process of identifying which memo values correspond to which proforma cells |
| **Validation** | QA step that cross-checks proposed updates against source documents |
| **Branding** | Applying Subtext brand theme (fonts, colors) to the output memo |
| **Change log** | Markdown file documenting every modification, rejection, and missed metric |
| **Dry run** | Preview mode â€” shows changes without modifying any files |
| **Comps** | Comparable properties used for market analysis |
