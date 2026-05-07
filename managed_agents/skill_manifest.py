"""
Declarative manifest of Memo Chef's custom skills.

`SKILL_DIRS` lists every custom skill directory under
`managed_agents/skills_content/`. The display_title is what shows up in
the org's skill catalog; the directory name doubles as the SKILL.md
`name` field (lowercase-hyphen).

The skill_id assigned by Anthropic when a skill is first created is
persisted to `managed_agents/.skills.json`. Subsequent runs of
`sync_skills.py` use that mapping to publish new versions instead of
creating duplicates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SKILLS_CONTENT_DIR = Path(__file__).resolve().parent / "skills_content"
SKILLS_CACHE_FILE = Path(__file__).resolve().parent / ".skills.json"


@dataclass(frozen=True)
class SkillSpec:
    name: str  # directory name; matches SKILL.md frontmatter `name`
    display_title: str

    @property
    def source_dir(self) -> Path:
        return SKILLS_CONTENT_DIR / self.name


SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec("memo-table-updates", "Memo Table Updates"),
    SkillSpec("image-table-replacement", "Image Table Replacement"),
    SkillSpec("layout-integrity", "Layout Integrity"),
    SkillSpec("memo-changelog", "Memo Changelog"),
    SkillSpec("fireflies-transcripts", "Fireflies Transcripts"),
    SkillSpec("market-workbook", "Market Workbook"),
    SkillSpec("toc-maintenance", "Table of Contents Maintenance"),
)


def load_cache() -> dict[str, str]:
    """Return the {skill_name: skill_id} cache, or an empty dict."""
    if not SKILLS_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(SKILLS_CACHE_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def save_cache(cache: dict[str, str]) -> None:
    SKILLS_CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
