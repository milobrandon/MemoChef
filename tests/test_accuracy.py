"""Tests for accuracy metrics and confidence scoring."""
import pytest
from memo_chef.accuracy import compute_accuracy_metrics


def test_perfect_run_scores_100():
    """All mappings accepted, none missed = 100 confidence."""
    raw = {
        "table_updates": [{"old_value": "x"}] * 10,
        "text_updates": [{"old_text": "y"}] * 5,
        "row_inserts": [],
        "chart_updates": [],
    }
    validated = {"rejected": [], "corrections": [], "missed": []}
    results = [{"match_quality": "exact"}] * 15

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["confidence_score"] == 100.0
    assert metrics["rejection_rate_pct"] == 0.0
    assert metrics["miss_rate_pct"] == 0.0


def test_half_rejected_lowers_score():
    """50% rejection rate should lower confidence significantly."""
    raw = {
        "table_updates": [{"old_value": "x"}] * 10,
        "text_updates": [],
        "row_inserts": [],
        "chart_updates": [],
    }
    validated = {
        "rejected": [{"idx": i} for i in range(5)],
        "corrections": [],
        "missed": [],
    }
    results = [{"match_quality": "exact"}] * 5

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["confidence_score"] < 90
    assert metrics["rejection_rate_pct"] == 50.0


def test_all_degraded_matches_lowers_score():
    """All degraded matches should lower the match quality component."""
    raw = {
        "table_updates": [{"old_value": "x"}] * 10,
        "text_updates": [],
        "row_inserts": [],
        "chart_updates": [],
    }
    validated = {"rejected": [], "corrections": [], "missed": []}
    results = [{"match_quality": "degraded_pass_2"}] * 10

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["match_quality_pct"] == 0.0
    assert metrics["confidence_score"] < 90


def test_empty_run_returns_zero():
    """No mappings at all should score 0."""
    raw = {"table_updates": [], "text_updates": [], "row_inserts": [], "chart_updates": []}
    validated = {"rejected": [], "corrections": [], "missed": []}
    results = []

    metrics = compute_accuracy_metrics(raw, validated, results)
    assert metrics["confidence_score"] == 0.0
    assert metrics["total_mappings"] == 0
