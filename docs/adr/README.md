# Architecture Decision Records (ADRs)

> Lightweight records of significant technical decisions made on this project.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-monolith-to-packages.md) | Defer monolith-to-packages refactor until tests exist | Accepted | 2026-03-01 |
| [002](002-claude-model-selection.md) | Claude model selection strategy | Accepted | 2026-03-01 |
| [003](003-market-data-approach.md) | Start with free data sources, defer commercial | Accepted | 2026-03-01 |

## Template

When adding a new ADR, copy this template:

```markdown
# ADR-NNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-NNN
**Date:** YYYY-MM-DD
**Author:** @name

## Context

What is the issue that we're seeing that is motivating this decision?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?
```

## Conventions

- Number ADRs sequentially (`001`, `002`, ...).
- Once accepted, ADRs are immutable. To change a decision, create a new ADR that supersedes the old one.
- Keep ADRs short — one page max.
