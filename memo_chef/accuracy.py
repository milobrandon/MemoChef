"""Accuracy metrics and confidence scoring for pipeline runs."""
from __future__ import annotations


def compute_accuracy_metrics(
    raw: dict,
    validated: dict,
    results: list[dict],
) -> dict:
    """Compute accuracy metrics from pipeline outputs.

    Args:
        raw: Raw mappings dict (table_updates, text_updates, row_inserts, chart_updates).
        validated: Validation output (rejected, corrections, missed).
        results: List of change records from apply_updates (with match_quality).

    Returns:
        Dict with confidence_score (0-100) and component metrics.
    """
    total = (
        len(raw.get("table_updates", []))
        + len(raw.get("text_updates", []))
        + len(raw.get("row_inserts", []))
        + len(raw.get("chart_updates", []))
    )

    if total == 0:
        return {
            "confidence_score": 0.0,
            "coverage_pct": 0.0,
            "rejection_rate_pct": 0.0,
            "correction_rate_pct": 0.0,
            "miss_rate_pct": 0.0,
            "match_quality_pct": 0.0,
            "total_mappings": 0,
            "rejected": 0,
            "corrections": 0,
            "missed": 0,
        }

    rejected = len(validated.get("rejected", []))
    corrections = len(validated.get("corrections", []))
    missed = len(validated.get("missed", []))

    degraded = sum(
        1 for r in results
        if str(r.get("match_quality", "")).startswith("degraded")
    )

    total_with_missed = max(total + missed, 1)
    coverage = (total_with_missed - missed) / total_with_missed
    acceptance = (total - rejected) / max(total, 1)
    correction_quality = 1 - corrections / max(total, 1)
    match_quality = (len(results) - degraded) / max(len(results), 1) if results else 1.0
    miss_quality = 1 - missed / total_with_missed

    confidence = (
        coverage * 30
        + acceptance * 25
        + correction_quality * 20
        + match_quality * 15
        + miss_quality * 10
    )

    return {
        "confidence_score": round(confidence, 1),
        "coverage_pct": round(coverage * 100, 1),
        "rejection_rate_pct": round(rejected / max(total, 1) * 100, 1),
        "correction_rate_pct": round(corrections / max(total, 1) * 100, 1),
        "miss_rate_pct": round(missed / total_with_missed * 100, 1),
        "match_quality_pct": round(match_quality * 100, 1),
        "total_mappings": total,
        "rejected": rejected,
        "corrections": corrections,
        "missed": missed,
    }
