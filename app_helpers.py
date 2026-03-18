"""Testable helper logic shared by the Streamlit app."""

import hashlib
import secrets


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return ``salt_hex:hash_hex`` using PBKDF2-SHA256."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Validate password against stored ``salt_hex:hash_hex``."""
    try:
        salt_hex, hash_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        bytes.fromhex(hash_hex)
    except (AttributeError, ValueError):
        return False

    candidate = hash_password(password, salt)
    return secrets.compare_digest(candidate, stored_hash)


def should_disable_fire_button(
    memo_file: object | None,
    proforma_file: object | None,
    remaining_credits: int,
    credits_error: str | None,
) -> bool:
    """Return True when the run button should be disabled."""
    return (
        not (memo_file and proforma_file)
        or remaining_credits <= 0
        or credits_error is not None
    )


def build_change_report_html(changes: list[dict], manifest: dict | None = None) -> str:
    """Build a branded Before vs After HTML report from a list of change records.

    Each change dict has: page, type, location, old, new, source.
    Returns a self-contained HTML string styled with Subtext brand colors.
    """
    import html as html_mod
    from collections import defaultdict

    # Group changes by page
    by_page: dict[int, list[dict]] = defaultdict(list)
    for c in changes:
        by_page[c.get("page", 0)].append(c)

    # Type icons
    type_icons = {
        "table": "grid_on",
        "text": "text_fields",
        "narrative": "auto_stories",
        "row_insert": "add_circle",
        "table_structure": "build",
        "chart": "bar_chart",
    }
    type_colors = {
        "table": "#c1d100",     # lime green
        "text": "#f7f1e3",      # beige
        "narrative": "#a95818", # birch
        "row_insert": "#16352e",# everest green
        "table_structure": "#512213",  # brown
        "chart": "#c1d100",     # lime green
    }

    counts = manifest.get("counts", {}) if manifest else {}
    accuracy = manifest.get("accuracy", {}) if manifest else {}

    lines = [
        '<div class="change-report">',
        '<style>',
        '.change-report { font-family: "Pragmatica Book", "Inter", sans-serif; }',
        '.change-report .report-header { background: linear-gradient(135deg, #16352e 0%, #2b2825 100%); '
        'padding: 24px 28px; border-radius: 12px; margin-bottom: 20px; }',
        '.change-report .report-header h2 { color: #c1d100; margin: 0 0 4px 0; font-size: 20px; '
        'font-family: "Pragmatica Bold", "Inter", sans-serif; }',
        '.change-report .report-header .subtitle { color: #bfb8a8; font-size: 13px; }',
        '.change-report .report-stats { display: flex; gap: 16px; margin: 12px 0; flex-wrap: wrap; }',
        '.change-report .stat-chip { background: rgba(193,209,0,0.12); color: #c1d100; '
        'padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }',
        '.change-report .page-section { margin-bottom: 16px; }',
        '.change-report .page-header { color: #c1d100; font-size: 14px; font-weight: 700; '
        'padding: 8px 0; border-bottom: 1px solid rgba(193,209,0,0.2); margin-bottom: 8px; }',
        '.change-report .change-row { display: grid; grid-template-columns: 60px 80px 1fr 20px 1fr; '
        'gap: 8px; padding: 6px 8px; border-radius: 6px; margin-bottom: 4px; align-items: start; '
        'background: rgba(43,40,37,0.5); font-size: 12px; }',
        '.change-report .change-type { color: #bfb8a8; font-size: 11px; text-transform: uppercase; '
        'letter-spacing: 0.5px; }',
        '.change-report .change-loc { color: #bfb8a8; font-size: 11px; overflow: hidden; '
        'text-overflow: ellipsis; white-space: nowrap; }',
        '.change-report .old-val { color: #e88; background: rgba(255,80,80,0.08); padding: 2px 6px; '
        'border-radius: 4px; text-decoration: line-through; word-break: break-word; }',
        '.change-report .arrow { color: #c1d100; font-size: 14px; text-align: center; }',
        '.change-report .new-val { color: #c1d100; background: rgba(193,209,0,0.08); padding: 2px 6px; '
        'border-radius: 4px; font-weight: 600; word-break: break-word; }',
        '.change-report .source-tag { color: #bfb8a8; font-size: 10px; opacity: 0.7; '
        'margin-top: 2px; }',
        '</style>',
        '<div class="report-header">',
        '<h2>Before vs After Report</h2>',
        f'<div class="subtitle">{len(changes)} changes across {len(by_page)} slides</div>',
        '<div class="report-stats">',
    ]

    # Stats chips
    type_counts: dict[str, int] = defaultdict(int)
    for c in changes:
        type_counts[c.get("type", "unknown")] += 1
    for t, n in sorted(type_counts.items()):
        lines.append(f'<span class="stat-chip">{n} {t}</span>')

    if accuracy.get("confidence_score"):
        lines.append(
            f'<span class="stat-chip">Confidence: {accuracy["confidence_score"]:.0f}%</span>'
        )
    review_score = counts.get("final_review_score")
    if review_score:
        lines.append(f'<span class="stat-chip">QA Score: {review_score}</span>')

    lines.append('</div></div>')

    # Changes grouped by page
    for page in sorted(by_page.keys()):
        page_changes = by_page[page]
        lines.append('<div class="page-section">')
        lines.append(f'<div class="page-header">Slide {page}</div>')
        for c in page_changes:
            ctype = c.get("type", "unknown")
            loc = html_mod.escape(str(c.get("location", ""))[:50])
            old = html_mod.escape(str(c.get("old", ""))[:120])
            new = html_mod.escape(str(c.get("new", ""))[:120])
            lines.append('<div class="change-row">')
            lines.append(f'<span class="change-type">{ctype}</span>')
            lines.append(f'<span class="change-loc" title="{loc}">{loc}</span>')
            lines.append(f'<span class="old-val">{old}</span>')
            lines.append('<span class="arrow">&rarr;</span>')
            lines.append(f'<span class="new-val">{new}</span>')
            lines.append('</div>')
        lines.append('</div>')

    lines.append('</div>')
    return "\n".join(lines)
