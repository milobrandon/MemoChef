# Architecture Decision Records (ADRs)

Lightweight records of significant technical decisions made on this project.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-monolith-to-packages.md) | Defer monolith-to-packages refactor until tests exist | Accepted | 2026-03-01 |
| [002](002-claude-model-selection.md) | Claude model selection strategy | Accepted | 2026-03-01 |
| [003](003-market-data-approach.md) | Start with free data sources, defer commercial | Accepted | 2026-03-01 |
| [004](004-externalize-prompt-templates.md) | Externalize prompt templates to versioned files | Accepted | 2026-03-03 |
| [005](005-pydantic-config-validation.md) | Use pydantic for strict configuration validation | Accepted | 2026-03-03 |
| [006](006-ci-quality-gates-and-secret-scanning.md) | CI quality gates and secret scanning | Accepted | 2026-03-03 |

## Template

When adding a new ADR, copy this template:

```markdown
# ADR-NNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-NNN
**Date:** YYYY-MM-DD
**Author:** @name

## Context

What issue is motivating this decision?

## Decision

What change are we making?

## Consequences

What becomes easier or harder because of this change?
```

## Conventions

- Number ADRs sequentially (`001`, `002`, ...).
- Once accepted, ADRs are immutable. To change a decision, create a new ADR that supersedes the old one.
- Keep ADRs short (about one page max).
