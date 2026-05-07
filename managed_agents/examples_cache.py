"""
File-id cache for example IC memos.

`upload_example_memos()` used to push every `.pptx` under
`managed_agents/examples/` to the Files API on every single session,
which redundantly transferred tens of MB per run. This module turns
that into a one-time cost: each example is uploaded once, the
returned `file_id` is cached alongside the file's content hash, and
subsequent sessions reference the cached id directly.

Cache shape (`managed_agents/.examples.json`, gitignored):

    {
      "limestone_ic_memo.pptx": {
        "file_id": "file_01abc...",
        "sha256": "<hex digest of the on-disk bytes at upload time>"
      },
      ...
    }

Two things invalidate an entry and force a re-upload:

1. `sha256` of the local file no longer matches what's cached
   (someone replaced the example with a newer version).
2. `get_file(file_id)` returns 404 (the Files API expired the id).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from managed_agents.api_client import get_file, upload_file
from managed_agents.config import EXAMPLES_DIR

EXAMPLES_CACHE_FILE = Path(__file__).resolve().parent / ".examples.json"


@dataclass(frozen=True)
class CachedExample:
    file_id: str
    sha256: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: Path = EXAMPLES_CACHE_FILE) -> dict[str, CachedExample]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, CachedExample] = {}
    for name, entry in raw.items():
        try:
            out[name] = CachedExample(
                file_id=entry["file_id"], sha256=entry["sha256"],
            )
        except (KeyError, TypeError):
            continue
    return out


def save_cache(
    cache: dict[str, CachedExample],
    path: Path = EXAMPLES_CACHE_FILE,
) -> None:
    serializable = {
        name: {"file_id": c.file_id, "sha256": c.sha256}
        for name, c in sorted(cache.items())
    }
    path.write_text(json.dumps(serializable, indent=2) + "\n")


def _ensure_cached(
    pptx_path: Path,
    cache: dict[str, CachedExample],
    *,
    validate_remote: bool,
    upload_fn=upload_file,
    get_fn=get_file,
) -> CachedExample:
    """Return a CachedExample for `pptx_path`, uploading only if needed.

    Mutates `cache` in-place when an entry is added or refreshed; the
    caller is responsible for persisting it.

    When `validate_remote` is True, a hit on local sha256 still
    triggers a `get_file` call to confirm the cached file_id is still
    live (Files API entries can be deleted out from under us). When
    False, we trust the cache and skip the network round-trip — useful
    for tests and for runtime paths where we want zero overhead.
    """
    digest = _hash_file(pptx_path)
    name = pptx_path.name
    existing = cache.get(name)

    if existing is not None and existing.sha256 == digest:
        if not validate_remote:
            return existing
        meta = get_fn(existing.file_id)
        if meta is not None:
            return existing
        # File expired server-side; fall through to re-upload.

    new_id = upload_fn(pptx_path)
    fresh = CachedExample(file_id=new_id, sha256=digest)
    cache[name] = fresh
    return fresh


def resolve_examples(
    *,
    validate_remote: bool = True,
    examples_dir: Path = EXAMPLES_DIR,
    cache_path: Path = EXAMPLES_CACHE_FILE,
    upload_fn=upload_file,
    get_fn=get_file,
) -> list[dict]:
    """Return the resource dicts for every example .pptx, uploading
    anything missing from cache.

    Result shape matches the Managed Agents `resources` field:
    `[{"type": "file", "file_id": "...", "mount_path": "/mnt/examples/X.pptx"}, ...]`
    """
    if not examples_dir.exists():
        return []

    cache = load_cache(cache_path)
    initial_state = dict(cache)
    resources: list[dict] = []

    for path in sorted(examples_dir.glob("*.pptx")):
        cached = _ensure_cached(
            path, cache,
            validate_remote=validate_remote,
            upload_fn=upload_fn, get_fn=get_fn,
        )
        resources.append({
            "type": "file",
            "file_id": cached.file_id,
            "mount_path": f"/mnt/examples/{path.name}",
        })

    if cache != initial_state:
        save_cache(cache, cache_path)

    return resources
