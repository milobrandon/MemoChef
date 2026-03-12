"""Tests for run analytics aggregation."""
import json
from unittest.mock import patch, MagicMock

import pytest


def _mock_rows():
    """Simulate query results from memo_chef_runs."""
    return [
        {
            "run_id": "r1", "username": "alice", "status": "completed",
            "change_count": 42, "rejected_count": 2, "missed_count": 1,
            "duration_seconds": 120.0, "estimated_cost_microdollars": 50000,
            "confidence_score": 92.5, "coverage_pct": 95.0,
            "warnings_json": json.dumps([{"stage": "validation", "message": "truncation"}]),
            "created_at": "2026-03-10T10:00:00",
        },
        {
            "run_id": "r2", "username": "bob", "status": "completed",
            "change_count": 30, "rejected_count": 0, "missed_count": 3,
            "duration_seconds": 90.0, "estimated_cost_microdollars": 40000,
            "confidence_score": 88.0, "coverage_pct": 90.0,
            "warnings_json": json.dumps([]),
            "created_at": "2026-03-11T14:00:00",
        },
    ]


def test_analytics_total_runs():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert result["total_runs"] == 2


def test_analytics_total_cost():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert result["total_cost_usd"] == pytest.approx(0.09, abs=0.01)


def test_analytics_avg_confidence():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert result["avg_confidence"] == pytest.approx(90.25, abs=0.1)


def test_analytics_by_user():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    users = {u["username"]: u for u in result["by_user"]}
    assert "alice" in users
    assert "bob" in users
    assert users["alice"]["runs"] == 1


def test_analytics_warning_counts():
    from app_services import _compute_analytics
    result = _compute_analytics(_mock_rows())
    assert any(w["warning"] == "truncation" for w in result["warning_counts"])


def test_analytics_empty_input():
    from app_services import _compute_analytics
    result = _compute_analytics([])
    assert result["total_runs"] == 0
    assert result["avg_confidence"] == 0.0
