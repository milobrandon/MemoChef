# Critical Fixes & Hardening — Planning Document

**Date:** 2026-03-18
**Status:** Proposed
**Priority:** High — blocks production readiness

---

## 1. Problem Statement

A code review of the last 5 commits identified **3 high-severity bugs**, **2 security vulnerabilities**, and **systemic issues** with feature management and test coverage that collectively block production deployment.

### High-Severity Bugs

| # | File | Issue | Impact |
|---|------|-------|--------|
| H1 | `memo_chef/pipeline.py:409` | `"table_structure_updates"` key missing from batch result dict initialization | **Runtime crash** (`KeyError`) on every batch API run that returns table structure updates |
| H2 | `memo_chef/comp_builder.py:188,228` | `add_slide()` appends slide, then `insert_slide_at_position()` moves it — but the original append is never cleaned up | **Duplicate slides** in output deck |
| H3 | `memo_chef/slide_generator.py:240` | Global `inserted` counter applied to all future positions regardless of whether earlier inserts affect them | **Slides inserted at wrong positions** when multiple slides target different locations |

### Security Vulnerabilities

| # | File | Issue | Impact |
|---|------|-------|--------|
| S1 | `memo_chef/models.py:65` | `api_key` stored as plain `str` in Pydantic model | Key exposed via `.model_dump()`, `.json()`, `repr()`, logging, serialization |
| S2 | `tests/test_playwright_ui.py:28`, `docs/TESTING.md:32` | Default password `MemoChef2026` hardcoded in source | Credential leak if same password used in any deployment |

### Systemic Issues

| # | Area | Issue | Impact |
|---|------|-------|--------|
| F1 | `memo_automator.py`, `pipeline.py` | Features disabled via comments/hardcoded skips instead of config flags | Maintenance risk — accidental re-enable, no runtime toggle, unclear state |
| T1 | Multiple modules | Zero test coverage for `slide_generator.py` (579 lines), `comp_builder.py`, `run_final_review`, `run_consistency_check`, format validation, table mutation functions | Regressions go undetected |

---

## 2. Goals

1. **Zero runtime crashes** from known bugs (H1, H2, H3)
2. **No credentials in source code** (S1, S2)
3. **Feature flags in config** replacing comment-disabled code (F1)
4. **Unit tests** for new pipeline stages with mocked API responses (T1)

## 3. Non-Goals

- Refactoring `run_memo_pipeline` into a stage registry (architectural — separate effort)
- Adding CI integration for Playwright tests (infrastructure — separate effort)
- Removing the `rapidfuzz` dependency (low priority)

---

## 4. Approach

### 4.1 Fix H1: Missing `table_structure_updates` initialization

**Root cause:** When `_mapping_with_batch_api` was extended to support `narrative_updates` and `table_structure_updates`, only `narrative_updates` was added to the initial dict on line 409.

**Fix:** Add `"table_structure_updates": []` to the initialization dict.

**Risk:** Minimal — additive change to a dict literal.

### 4.2 Fix H2: Duplicate slide in comp builder

**Root cause:** `_build_comp_slide_from_scratch` calls `prs.slides.add_slide(layout)` which appends the slide to the end of the presentation. Later, `insert_slide_at_position(prs, slide, ...)` moves the slide XML to the correct position, but `add_slide` already registered it in the slide list.

**Fix:** Replace the two-step approach. Use `add_slide` to create the slide (needed to get a valid slide object with relationships), populate it, then use `insert_slide_at_position` to *move* it. The `insert_slide_at_position` function must remove the slide from its current position before inserting at the target position. Verify that `insert_slide_at_position` already handles this (moves rather than copies). If it doesn't, fix it to do a move.

**Risk:** Medium — need to verify `insert_slide_at_position` semantics. Test with a deck that has comp slides.

### 4.3 Fix H3: Multi-slide insertion offset

**Root cause:** `build_and_insert_slides` adds a global `inserted` counter to every future insert position. This is correct *only* when all inserts are at positions after the first insert. If slide A is inserted at position 7 and slide B at position 3, B's position should not be shifted by A's insertion.

**Fix:** Only increment the offset for inserts that occur at or before the current target position. Track inserted positions in a list and compute per-insert offset:

```python
inserted_positions = []
for slide_spec in sorted_slides:
    offset = sum(1 for p in inserted_positions if p <= slide_spec.insert_after_slide - 1)
    target_idx = slide_spec.insert_after_slide - 1 + offset
    # ... insert ...
    inserted_positions.append(target_idx)
```

**Risk:** Low — the sorted order means earlier inserts always have lower indices, so the offset monotonically increases. But the fix is correct for unsorted inputs too.

### 4.4 Fix S1: SecretStr for API key

**Root cause:** `api_key: str` in `RunRequest` model exposes the key in any serialization.

**Fix:**
1. Change `api_key: str` to `api_key: SecretStr` in `RunRequest`
2. Update all call sites that read `request.api_key` to use `request.api_key.get_secret_value()`
3. Verify `SecretStr` is imported from `pydantic`

**Risk:** Medium — need to find and update all consumers of `request.api_key`. A missed call site will get a `SecretStr` object instead of a string, which will fail visibly (not silently).

### 4.5 Fix S2: Remove hardcoded credentials

**Fix:**
1. In `tests/test_playwright_ui.py:28`: Remove the default value from `os.environ.get("MEMO_CHEF_PASS", ...)`. Raise a clear error if the env var is not set.
2. In `docs/TESTING.md`: Replace the password value with `(required, no default)` and add a note about setting it.

**Risk:** Minimal — test-only change. Tests will fail if env var not set, which is the desired behavior.

### 4.6 Fix F1: Feature flags via config

**Root cause:** Three features are disabled by comments/hardcoded skips:
- Auto-split (`pipeline.py:923-927`)
- Footer normalization (`memo_automator.py:3482-3483`)
- Correction retry (`pipeline.py:689-690`)

**Fix:**
1. Add a `FeatureFlags` Pydantic model with boolean fields:
   ```python
   class FeatureFlags(BaseModel):
       auto_split_enabled: bool = False
       footer_normalization_enabled: bool = False
       correction_retry_enabled: bool = False
   ```
2. Load from an optional `feature_flags` section in the existing config YAML
3. Replace comment-disabled code with `if flags.auto_split_enabled:` guards
4. Default all three to `False` to preserve current behavior

**Risk:** Low — defaults maintain current behavior. Config-driven toggle allows runtime changes without code modification.

### 4.7 Fix T1: Add unit tests for new pipeline stages

**Scope:** Add tests with mocked Claude API responses for:
- `run_final_review()` — test APPROVED verdict, critical fixes verdict, JSON parse failure, max_tokens truncation
- `run_consistency_check()` — test pass, fail with corrections, API error handling
- `validate_mapping_formats()` / `_auto_fix_format()` — test dollar signs, commas, percentages, decimals, edge cases
- `_add_table_column()`, `_remove_table_row()`, `_reorder_table_rows()` — test XML manipulation with real PPTX table structures

**Approach:** Use `unittest.mock.patch` to mock the Anthropic client. Create minimal PPTX fixtures with `python-pptx` for table mutation tests.

**Risk:** Low — test-only changes.

---

## 5. Execution Order

The fixes are independent and can be implemented in parallel, but should be merged in this order for clean review:

| Phase | Items | Rationale |
|-------|-------|-----------|
| 1 | H1, H2, H3 | Fix runtime crashes first |
| 2 | S1, S2 | Security hardening |
| 3 | F1 | Feature flag infrastructure |
| 4 | T1 | Test coverage (validates all above) |

---

## 6. Validation

- All existing tests pass (`pytest`)
- New tests pass for each fix
- Manual smoke test: run pipeline with a deck that triggers batch API, comp slides, and multi-slide insertion
- `git grep -i "memochef2026"` returns zero results after S2
- `api_key` no longer appears in any `.model_dump()` output after S1
