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
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from managed_agents.api_client import get_file, upload_file
from managed_agents.config import EXAMPLES_DIR

EXAMPLES_CACHE_FILE = Path(__file__).resolve().parent / ".examples.json"

# Serialize cache reads + uploads + writes inside a single process so
# two concurrent FastAPI sessions can't both upload the same example
# (orphaning one file_id) or interleave bytes when writing back to
# `.examples.json`. Cross-process serialization would require a real
# file lock; the FastAPI server runs single-process under uvicorn so
# threading.Lock is the right scope for now.
_CACHE_LOCK = threading.Lock()


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
    """Atomically persist the cache.

    Writes to a sibling `.tmp` file then `os.replace()`s it onto the
    target path so a partial write or process kill can't leave the
    cache file truncated. `os.replace` is atomic on both POSIX and
    Windows when source and destination live on the same filesystem.
    """
    serializable = {
        name: {"file_id": c.file_id, "sha256": c.sha256}
        for name, c in sorted(cache.items())
    }
    payload = json.dumps(serializable, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


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
    validate_remote: bool = False,
    examples_dir: Path = EXAMPLES_DIR,
    cache_path: Path = EXAMPLES_CACHE_FILE,
    upload_fn=upload_file,
    get_fn=get_file,
) -> list[dict]:
    """Return the resource dicts for every example .pptx, uploading
    anything missing from cache.

    Result shape matches the Managed Agents `resources` field:
    `[{"type": "file", "file_id": "...", "mount_path": "/mnt/examples/X.pptx"}, ...]`

    Defaults to `validate_remote=False` because the runtime path runs
    on every session and a `get_file()` round-trip per cached entry
    would re-introduce the per-session network cost we're trying to
    eliminate. The bootstrap CLI flips this flag explicitly when the
    operator wants a deep-check.

    Concurrency: holds `_CACHE_LOCK` for the duration of the
    load → upload → save cycle so two sessions can't double-upload
    the same example or interleave bytes in the cache file.

    Partial-failure: persists the cache after each successful upload,
    so a network blip on file 5/10 doesn't lose uploads 1–4.

    Orphan-prune: cache entries whose filename is no longer in the
    examples dir get dropped before saving.
    """
    if not examples_dir.exists():
        return []

    with _CACHE_LOCK:
        cache = load_cache(cache_path)
        seen: set[str] = set()
        resources: list[dict] = []

        try:
            for path in sorted(examples_dir.glob("*.pptx")):
                seen.add(path.name)
                before = cache.get(path.name)
                cached = _ensure_cached(
                    path, cache,
                    validate_remote=validate_remote,
                    upload_fn=upload_fn, get_fn=get_fn,
                )
                if cache.get(path.name) != before:
                    # Persist after each successful (re-)upload so a
                    # later failure can't silently drop earlier work.
                    save_cache(cache, cache_path)
                resources.append({
                    "type": "file",
                    "file_id": cached.file_id,
                    "mount_path": f"/mnt/examples/{path.name}",
                })
        finally:
            # Prune entries for files that no longer exist locally
            # (renamed / deleted), and persist if anything changed.
            stale = set(cache) - seen
            if stale:
                for name in stale:
                    cache.pop(name, None)
                save_cache(cache, cache_path)

    return resources


def invalidate_cache_entries(
    filenames: list[str],
    *,
    cache_path: Path = EXAMPLES_CACHE_FILE,
) -> int:
    """Drop the named filenames from the cache. Returns count removed.

    Use after a session creation fails citing dead file_ids — the
    runtime path trusts the cache by default, so a server-side
    invalidation needs an explicit recovery hook.
    """
    with _CACHE_LOCK:
        cache = load_cache(cache_path)
        removed = 0
        for name in filenames:
            if cache.pop(name, None) is not None:
                removed += 1
        if removed:
            save_cache(cache, cache_path)
        return removed
