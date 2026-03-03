# ADR-006: CI Quality Gates and Secret Scanning

**Status:** Accepted  
**Date:** 2026-03-03  
**Author:** @developer

## Context

Quality checks were mostly local and inconsistent across environments. Secret
exposure risk also required automated scanning in CI.

## Decision

Use GitHub Actions CI with:

- OS matrix (`ubuntu-latest`, `windows-latest`)
- Ruff linting
- pytest non-integration suite
- coverage gate
- Gitleaks secret scanning

Also add local pre-commit configuration with Ruff, Black, and detect-secrets.

## Consequences

- Positive: consistent quality checks before merge.
- Positive: better cross-platform confidence.
- Positive: automated detection of committed secrets.
- Negative: slightly longer CI runtime.
- Mitigation: non-integration test scope keeps CI practical.
