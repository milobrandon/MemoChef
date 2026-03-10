"""Shared backend modules for the Memo Automator experience."""

from .models import CompUrl, RunManifest, RunRequest, RunResult, StageUpdate

# Lazy import: pipeline pulls in memo_automator.py which has heavy deps.
# Importing it eagerly can break the whole package if a dep is missing.


def __getattr__(name: str):
    if name == "run_memo_pipeline":
        from .pipeline import run_memo_pipeline

        return run_memo_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompUrl",
    "RunManifest",
    "RunRequest",
    "RunResult",
    "StageUpdate",
    "run_memo_pipeline",
]
