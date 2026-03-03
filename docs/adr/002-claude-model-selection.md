# ADR-002: Claude Model Selection Strategy

**Status:** Accepted
**Date:** 2026-03-01
**Author:** @brandon

## Context

The Memo Automator makes two Claude API calls per run:

1. **Mapping call** — Identifies which memo metrics correspond to which proforma values and produces structured JSON replacements.
2. **Validation call** — Cross-checks proposed updates for correctness, duplicates, and missed metrics.

Model choice affects accuracy, latency, and cost. The mapping call is the most complex (reasoning over two large documents), while validation is a structured review of already-identified changes.

## Decision

- **Mapping model:** Default to `claude-sonnet-4-6` (best balance of accuracy and speed). Configurable via `config.yaml` → `claude.model`.
- **Validation model:** Default to the same model as mapping. Configurable independently via `claude.validation_model` for users who want to use a cheaper/faster model for QA.
- **Configuration:** Both models are set in `config.yaml`, not hardcoded, so teams can upgrade to Opus for higher accuracy on complex memos or downgrade to Haiku for cost-sensitive batch runs.

```yaml
claude:
  model: "claude-sonnet-4-6"              # mapping
  validation_model: "claude-sonnet-4-6"   # QA pass (can differ)
```

## Consequences

- **Positive:** Sonnet provides strong accuracy at moderate cost for typical IC memos.
- **Positive:** Independent model config allows cost optimization (e.g., Sonnet for mapping, Haiku for validation).
- **Positive:** Easy to upgrade to Opus for high-stakes memos without code changes.
- **Negative:** Model version strings will need updating when Anthropic releases new versions.
- **Mitigation:** Pin model versions in config; document upgrade process in RUNBOOK.md.
