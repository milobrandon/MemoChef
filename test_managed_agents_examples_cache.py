"""Unit tests for the examples file_id cache.

Covers:
  - Cache I/O round-trip and corrupt-file handling.
  - First-run miss: upload happens, cache populated.
  - Repeat run with unchanged file: zero uploads, zero get_file calls
    when validate_remote=False; one get_file call (no upload) when
    validate_remote=True and the remote still exists.
  - sha256 invalidation: file content changes → re-upload.
  - 404 invalidation: cached file_id expired → re-upload.
  - resolve_examples produces the same resource shape the API expects.
  - api_client.get_file returns None on 404 (smoke test).
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from managed_agents import api_client
from managed_agents.examples_cache import (
    CachedExample,
    load_cache,
    resolve_examples,
    save_cache,
)


def _write_pptx(path, content: bytes = b"PK\x03\x04 fake pptx bytes"):
    path.write_bytes(content)
    return path


# ── Cache I/O ──────────────────────────────────────────────────────

def test_cache_round_trip(tmp_path):
    cache_path = tmp_path / ".examples.json"
    save_cache({"a.pptx": CachedExample("file_a", "abc")}, cache_path)

    loaded = load_cache(cache_path)
    assert loaded == {"a.pptx": CachedExample("file_a", "abc")}
    # JSON shape is stable so users can grep the file
    raw = json.loads(cache_path.read_text())
    assert raw == {"a.pptx": {"file_id": "file_a", "sha256": "abc"}}


def test_cache_handles_corrupt_file(tmp_path):
    cache_path = tmp_path / ".examples.json"
    cache_path.write_text("{ not json")
    assert load_cache(cache_path) == {}


def test_cache_skips_malformed_entries(tmp_path):
    """A partially-corrupt cache must keep the good entries and drop
    the broken ones — never crash and lose everything."""
    cache_path = tmp_path / ".examples.json"
    cache_path.write_text(json.dumps({
        "good.pptx": {"file_id": "file_g", "sha256": "h"},
        "broken.pptx": {"file_id": "file_b"},  # missing sha256
        "also_broken.pptx": "string instead of object",
    }))
    out = load_cache(cache_path)
    assert out == {"good.pptx": CachedExample("file_g", "h")}


# ── resolve_examples — first-run miss ──────────────────────────────

def test_first_run_uploads_and_populates_cache(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "alpha.pptx")
    _write_pptx(examples / "beta.pptx", content=b"different")
    cache_path = tmp_path / ".examples.json"

    upload_calls: list[str] = []
    get_calls: list[str] = []

    def _upload(path):
        upload_calls.append(path.name)
        return f"file_{path.stem}"

    def _get(file_id):
        get_calls.append(file_id)
        return {"id": file_id}

    resources = resolve_examples(
        examples_dir=examples,
        cache_path=cache_path,
        upload_fn=_upload,
        get_fn=_get,
    )

    assert sorted(upload_calls) == ["alpha.pptx", "beta.pptx"]
    assert get_calls == []  # nothing to validate yet
    assert {r["file_id"] for r in resources} == {"file_alpha", "file_beta"}
    assert all(r["mount_path"].startswith("/mnt/examples/") for r in resources)
    assert all(r["type"] == "file" for r in resources)

    persisted = load_cache(cache_path)
    assert persisted["alpha.pptx"].file_id == "file_alpha"
    assert persisted["beta.pptx"].file_id == "file_beta"


# ── resolve_examples — cache hit ───────────────────────────────────

def test_cache_hit_with_validate_skips_upload(tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "alpha.pptx", content=b"stable bytes")
    cache_path = tmp_path / ".examples.json"

    # Bootstrap by running once with a working uploader/get.
    def _upload(path):
        return "file_alpha"

    def _get(file_id):
        return {"id": file_id}

    resolve_examples(
        examples_dir=examples, cache_path=cache_path,
        upload_fn=_upload, get_fn=_get,
        validate_remote=True,
    )

    # Second run: cache should hit, no uploads even though we still
    # call get_file once per cached entry to validate the remote.
    upload_calls: list[str] = []
    get_calls: list[str] = []

    def _upload_strict(path):
        upload_calls.append(path.name)
        raise AssertionError("must not upload on cache hit")

    def _get_track(file_id):
        get_calls.append(file_id)
        return {"id": file_id}

    resources = resolve_examples(
        examples_dir=examples, cache_path=cache_path,
        upload_fn=_upload_strict, get_fn=_get_track,
        validate_remote=True,
    )

    assert upload_calls == []
    assert get_calls == ["file_alpha"]
    assert resources == [{
        "type": "file",
        "file_id": "file_alpha",
        "mount_path": "/mnt/examples/alpha.pptx",
    }]


def test_cache_hit_no_validate_no_network(tmp_path):
    """`validate_remote=False` must avoid both upload AND get_file —
    used at runtime when we don't want any network roundtrip per
    session."""
    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "alpha.pptx", content=b"bytes")
    cache_path = tmp_path / ".examples.json"
    save_cache({"alpha.pptx": CachedExample("file_alpha", _sha("bytes"))}, cache_path)

    def _fail(*a, **kw):
        raise AssertionError("no network call expected")

    resources = resolve_examples(
        examples_dir=examples, cache_path=cache_path,
        upload_fn=_fail, get_fn=_fail,
        validate_remote=False,
    )
    assert resources[0]["file_id"] == "file_alpha"


# ── Invalidation ───────────────────────────────────────────────────

def test_sha256_change_triggers_reupload(tmp_path):
    """If the on-disk file changed, sha256 won't match the cached
    entry and we must re-upload (not blindly hand back stale id)."""
    examples = tmp_path / "examples"
    examples.mkdir()
    pptx = _write_pptx(examples / "alpha.pptx", content=b"v1")
    cache_path = tmp_path / ".examples.json"
    save_cache(
        {"alpha.pptx": CachedExample("file_old", _sha("v1"))},
        cache_path,
    )

    # Replace the file's bytes — sha256 should diverge from cache.
    pptx.write_bytes(b"v2-different")

    upload_calls: list[str] = []

    def _upload(path):
        upload_calls.append(path.name)
        return "file_new"

    def _get(file_id):
        raise AssertionError("get_file must not run on sha mismatch")

    resources = resolve_examples(
        examples_dir=examples, cache_path=cache_path,
        upload_fn=_upload, get_fn=_get,
    )
    assert upload_calls == ["alpha.pptx"]
    assert resources[0]["file_id"] == "file_new"
    persisted = load_cache(cache_path)
    assert persisted["alpha.pptx"].file_id == "file_new"
    assert persisted["alpha.pptx"].sha256 == _sha("v2-different")


def test_404_triggers_reupload(tmp_path):
    """Cached file_id was deleted server-side. get_file returns None;
    we must re-upload and refresh the cache rather than handing the
    dead id to the API and tanking the session.

    This 404 path requires validate_remote=True — the runtime hot path
    deliberately trusts the cache (see test_runtime_path_skips_get_file)."""
    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "alpha.pptx", content=b"stable")
    cache_path = tmp_path / ".examples.json"
    save_cache(
        {"alpha.pptx": CachedExample("file_dead", _sha("stable"))},
        cache_path,
    )

    upload_calls: list[str] = []

    def _upload(path):
        upload_calls.append(path.name)
        return "file_fresh"

    def _get_404(file_id):
        return None  # API said "no such file"

    resources = resolve_examples(
        examples_dir=examples, cache_path=cache_path,
        upload_fn=_upload, get_fn=_get_404,
        validate_remote=True,
    )
    assert upload_calls == ["alpha.pptx"]
    assert resources[0]["file_id"] == "file_fresh"
    persisted = load_cache(cache_path)
    assert persisted["alpha.pptx"].file_id == "file_fresh"


# ── No-examples-dir ─────────────────────────────────────────────────

def test_missing_examples_dir_returns_empty(tmp_path):
    cache_path = tmp_path / ".examples.json"
    out = resolve_examples(
        examples_dir=tmp_path / "does-not-exist",
        cache_path=cache_path,
        upload_fn=lambda p: "x",
        get_fn=lambda f: None,
    )
    assert out == []


# ── Runtime hot-path skips network ─────────────────────────────────

def test_runtime_path_skips_get_file(tmp_path):
    """The runtime default (validate_remote=False) must not call
    get_file even once when the cache is warm. This is the perf
    contract of the PR: cache hits are free."""
    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "alpha.pptx", content=b"runtime bytes")
    cache_path = tmp_path / ".examples.json"
    save_cache(
        {"alpha.pptx": CachedExample("file_alpha", _sha("runtime bytes"))},
        cache_path,
    )

    def _no_upload(path):
        raise AssertionError("must not upload on cache hit")

    def _no_get(file_id):
        raise AssertionError("runtime path must not validate remotely")

    resources = resolve_examples(
        examples_dir=examples, cache_path=cache_path,
        upload_fn=_no_upload, get_fn=_no_get,
        # explicit, but this is also the default
        validate_remote=False,
    )
    assert resources[0]["file_id"] == "file_alpha"


def test_run_session_upload_example_memos_uses_no_validate(monkeypatch):
    """`run_session.upload_example_memos` is the runtime entry point.
    It must call resolve_examples without forcing validation, otherwise
    every session pays N get_file round-trips."""
    from managed_agents import run_session

    captured: dict = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_session, "resolve_examples", _spy)
    run_session.upload_example_memos()
    # Either the default (False) is used, or the call explicitly passes
    # validate_remote=False. Both are acceptable; what's NOT acceptable
    # is the runtime path opting in to validation.
    if "validate_remote" in captured:
        assert captured["validate_remote"] is False


# ── Partial-failure preservation ───────────────────────────────────

def test_partial_failure_preserves_completed_uploads(tmp_path):
    """If file 3/5 raises mid-loop, files 1 and 2 must be persisted
    so the next run doesn't re-upload them."""
    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "a.pptx", content=b"a")
    _write_pptx(examples / "b.pptx", content=b"b")
    _write_pptx(examples / "c.pptx", content=b"c")
    cache_path = tmp_path / ".examples.json"

    upload_count = {"n": 0}

    def _upload(path):
        upload_count["n"] += 1
        if upload_count["n"] == 3:
            raise RuntimeError("network blew up")
        return f"file_{path.stem}"

    with pytest.raises(RuntimeError):
        resolve_examples(
            examples_dir=examples, cache_path=cache_path,
            upload_fn=_upload, get_fn=lambda f: None,
        )

    persisted = load_cache(cache_path)
    # First two uploads must be on disk despite the third raising.
    assert persisted.get("a.pptx") and persisted["a.pptx"].file_id == "file_a"
    assert persisted.get("b.pptx") and persisted["b.pptx"].file_id == "file_b"
    assert "c.pptx" not in persisted


# ── Orphan pruning ─────────────────────────────────────────────────

def test_orphan_entries_pruned(tmp_path):
    """Cache entries for files no longer in the examples dir must be
    dropped — otherwise the cache leaks across years of curation."""
    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "kept.pptx", content=b"k")
    cache_path = tmp_path / ".examples.json"
    save_cache({
        "kept.pptx": CachedExample("file_keep", _sha("k")),
        "removed.pptx": CachedExample("file_old", _sha("old")),
        "renamed.pptx": CachedExample("file_orig", _sha("orig")),
    }, cache_path)

    resources = resolve_examples(
        examples_dir=examples, cache_path=cache_path,
        upload_fn=lambda p: "should_not_upload",
        get_fn=lambda f: None,
    )

    persisted = load_cache(cache_path)
    assert set(persisted.keys()) == {"kept.pptx"}
    assert [r["file_id"] for r in resources] == ["file_keep"]


# ── Atomic save ────────────────────────────────────────────────────

def test_save_cache_uses_atomic_replace(tmp_path, monkeypatch):
    """save_cache must write to a temp path and os.replace, not
    truncate-write directly. A torn write would erase every cached
    file_id and force re-uploads on every session until bootstrap.
    """
    cache_path = tmp_path / ".examples.json"
    cache_path.write_text(json.dumps({  # pre-existing valid file
        "old.pptx": {"file_id": "file_old", "sha256": "h"},
    }))

    replace_calls: list[tuple] = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        replace_calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr("managed_agents.examples_cache.os.replace", _spy_replace)

    save_cache(
        {"new.pptx": CachedExample("file_new", "h2")},
        cache_path,
    )

    # Exactly one replace, from a .tmp sibling onto the real path.
    assert len(replace_calls) == 1
    src, dst = replace_calls[0]
    assert dst == str(cache_path)
    assert src.endswith(".tmp")
    # Final file is well-formed and contains the new entry.
    assert json.loads(cache_path.read_text()) == {
        "new.pptx": {"file_id": "file_new", "sha256": "h2"},
    }


# ── Concurrency: lock prevents double-upload + corrupt JSON ────────

def test_concurrent_resolve_examples_serializes(tmp_path):
    """Two threads calling resolve_examples on a fresh cache must not
    both upload the same file. The lock serializes them so the second
    thread sees the first's cache entry."""
    import threading

    examples = tmp_path / "examples"
    examples.mkdir()
    _write_pptx(examples / "shared.pptx", content=b"x")
    cache_path = tmp_path / ".examples.json"

    upload_count = {"n": 0}
    upload_lock = threading.Lock()

    def _slow_upload(path):
        with upload_lock:
            upload_count["n"] += 1
            return f"file_v{upload_count['n']}"

    results: list[list[dict]] = []
    errors: list[BaseException] = []

    def _worker():
        try:
            r = resolve_examples(
                examples_dir=examples, cache_path=cache_path,
                upload_fn=_slow_upload, get_fn=lambda f: None,
            )
            results.append(r)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Exactly ONE upload despite 4 concurrent calls.
    assert upload_count["n"] == 1
    # All 4 callers got the same file_id back.
    file_ids = {r[0]["file_id"] for r in results}
    assert file_ids == {"file_v1"}
    # Cache file is valid JSON, not interleaved garbage.
    assert json.loads(cache_path.read_text()) == {
        "shared.pptx": {"file_id": "file_v1", "sha256": _sha("x")},
    }


# ── invalidate_cache_entries ───────────────────────────────────────

def test_invalidate_cache_entries_drops_named_files(tmp_path):
    from managed_agents.examples_cache import invalidate_cache_entries

    cache_path = tmp_path / ".examples.json"
    save_cache({
        "a.pptx": CachedExample("file_a", "h"),
        "b.pptx": CachedExample("file_b", "h"),
        "c.pptx": CachedExample("file_c", "h"),
    }, cache_path)

    n = invalidate_cache_entries(["a.pptx", "missing.pptx"], cache_path=cache_path)
    assert n == 1  # 'missing.pptx' wasn't in cache; only 'a.pptx' counted

    persisted = load_cache(cache_path)
    assert set(persisted.keys()) == {"b.pptx", "c.pptx"}


# ── api_client.get_file 404 contract ───────────────────────────────

class _Resp:
    def __init__(self, status_code: int, body: dict | str = ""):
        self.status_code = status_code
        self.text = json.dumps(body) if isinstance(body, dict) else body
        self._body = body

    def json(self):
        return self._body


class _Client:
    queued: _Resp | None = None

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, *a, **kw):
        return type(self).queued


def test_get_file_returns_none_on_404():
    _Client.queued = _Resp(404, "not found")
    with patch.object(api_client.httpx, "Client", _Client):
        assert api_client.get_file("file_x") is None


def test_get_file_returns_metadata_on_200():
    _Client.queued = _Resp(200, {"id": "file_x", "filename": "a.pptx"})
    with patch.object(api_client.httpx, "Client", _Client):
        out = api_client.get_file("file_x")
    assert out == {"id": "file_x", "filename": "a.pptx"}


# ── helpers ────────────────────────────────────────────────────────

def _sha(content: bytes | str) -> str:
    import hashlib
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()
