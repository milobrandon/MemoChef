"""
Pre-built Anthropic skills attached to the Memo Chef agent.

Skills are filesystem-based, on-demand expertise. Anthropic ships
maintained xlsx and pptx skills that match what Memo Chef does:
read/write proforma workbooks (xlsx) and edit IC memo decks (pptx).

Attaching them lets the agent load the relevant SKILL.md from the
filesystem only when it's actually working with that file type — no
context cost when not in use.

Docs: https://platform.claude.com/docs/en/managed-agents/skills
"""

from __future__ import annotations

# Pre-built Anthropic skills. Keep this list short — every skill costs
# ~100 tokens of metadata per session even when unused.
SKILLS: list[dict] = [
    {"type": "anthropic", "skill_id": "xlsx"},
    {"type": "anthropic", "skill_id": "pptx"},
]
