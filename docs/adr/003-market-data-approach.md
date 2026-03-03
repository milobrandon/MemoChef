# ADR-003: Start with Free Data Sources, Defer Commercial

**Status:** Accepted
**Date:** 2026-03-01
**Author:** @brandon

## Context

The v1.5 roadmap includes market data enrichment — populating memo sections with rent comps, sale comps, submarket stats, and macro indicators from external sources.

Commercial data providers (CoStar, Yardi Matrix, RCA/MSCI) offer the highest-quality CRE data but require enterprise licenses with strict redistribution terms. Free public sources (FRED, Census/ACS) provide useful macro and demographic data but lack property-level comps.

## Decision

1. **Phase 1-2 (now through v1.0):** No external market data. Focus on proforma-to-memo accuracy.
2. **Phase 3 (v1.5):** Integrate free public sources first:
   - **FRED API** — Interest rates (SOFR, 10-year Treasury), CPI, employment.
   - **Census / ACS API** — MSA demographics, housing stats, population growth.
3. **After v1.5:** Evaluate commercial sources only after:
   - Confirming the firm's existing data licenses permit API-based access.
   - Legal review of redistribution terms (the tool sends data to Claude's API).
   - Cost-benefit analysis vs. manual data entry.

## Consequences

- **Positive:** No licensing risk in v1.0 — all data comes from the user's own proforma.
- **Positive:** Free sources are sufficient for macro context (rates, demographics) that analysts already look up manually.
- **Positive:** Establishes the ingestion pipeline architecture before adding complex commercial integrations.
- **Negative:** Property-level comps (rent comps, sale comps) won't be automated until commercial sources are cleared.
- **Mitigation:** Analysts continue to manually enter comps as they do today; the tool handles everything else.
