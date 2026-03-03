# ADR-005: Use Pydantic for Strict Configuration Validation

**Status:** Accepted  
**Date:** 2026-03-03  
**Author:** @developer

## Context

Config validation relied on manual checks that were easy to drift from actual
defaults and did not provide structured field-level errors.

## Decision

Adopt `pydantic` models for configuration schema validation with:

- typed nested sections (`proforma`, `memo`, `schedule`, `branding`, `layout`, `claude`)
- default values defined in schema models
- bounds validation (e.g., non-negative integers, temperature 0-1)
- strict unknown-key handling (`extra=\"forbid\"`)

## Consequences

- Positive: clearer and more consistent validation errors.
- Positive: single source of truth for defaults and types.
- Positive: easier future migration to per-project config profiles.
- Negative: introduces a new dependency (`pydantic`).
- Mitigation: dependency is lightweight and widely adopted.
