# ADR-004: Externalize Prompt Templates to Versioned Files

**Status:** Accepted  
**Date:** 2026-03-03  
**Author:** @developer

## Context

Prompt templates were embedded inline in `memo_automator.py`, making them
hard to diff, review, and version independently from code flow changes.

## Decision

Move prompt templates to:

- `prompts/mapping_v1.txt`
- `prompts/validation_v1.txt`

Load templates at runtime with a safe fallback to inline strings if files are
missing.

## Consequences

- Positive: prompt edits become cleaner and easier to review.
- Positive: enables prompt versioning workflows without changing code logic.
- Positive: easier prompt regression testing.
- Negative: one more file dependency at runtime.
- Mitigation: fallback loader preserves behavior if files are missing.
