# ADR-001: Defer Monolith-to-Packages Refactor Until Tests Exist

**Status:** Accepted
**Date:** 2026-03-01
**Author:** @brandon

## Context

`memo_automator.py` is a 2,500+ line monolith containing extraction, AI calls, formatting, and output logic. The conventional approach would be to refactor into a package structure (`extraction/`, `ai/`, `formatting/`, `output/`) immediately.

However, the project currently has only two ad-hoc test scripts and no automated test suite. Refactoring without tests risks introducing regressions that go undetected — the tool processes confidential financial data where silent errors have real consequences.

## Decision

Defer the package refactor (proposed `memo_automator/` structure in ROADMAP.md) until Phase 1 testing is complete. Specifically:

1. **Phase 1** — Add pytest infrastructure, unit tests for key functions, and integration tests with mocked Claude responses.
2. **Phase 2** — Refactor into packages with confidence that tests catch regressions.

In the meantime, keep the monolith as-is but make targeted improvements (logging, input validation, CLI flags) that don't change the module's public API.

## Consequences

- **Positive:** No risk of breaking working functionality during the documentation/testing phase.
- **Positive:** Tests written against the monolith will still apply after refactoring (same function signatures).
- **Negative:** The monolith remains hard to navigate in the short term.
- **Negative:** New contributors may be discouraged by the single large file.
- **Mitigation:** Clear section headers and a table of contents in the module docstring help navigation.
