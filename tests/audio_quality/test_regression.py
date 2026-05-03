"""Tier 3: Regression baseline comparison tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.audio_quality.suites.regression import (
    BaselineManager,
    load_baseline,
    save_baseline,
)


def test_save_and_load_baseline_roundtrip():
    """Saving and loading a baseline should preserve all data."""
    baseline = {
        "version": "test-v1",
        "commit": "abc1234",
        "branch": "test",
        "date": "2026-05-03",
        "config": {"enable_post_processing": False},
        "samples": {
            "hello": {
                "text": "Hello world",
                "dnsmos_sig": 3.92,
                "dnsmos_ovrl": 3.78,
            }
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_baseline.json"
        save_baseline(baseline, path)
        loaded = load_baseline(path)

        assert loaded["version"] == "test-v1"
        assert loaded["samples"]["hello"]["dnsmos_sig"] == 3.92


def test_baseline_manager_compare_detects_regression():
    """BaselineManager.compare should flag regressions."""
    manager = BaselineManager(regression_threshold_pct=5.0)

    baseline_scores = {"dnsmos_sig": 3.92, "dnsmos_ovrl": 3.78, "silence_ratio": 0.08}
    current_scores = {"dnsmos_sig": 3.30, "dnsmos_ovrl": 3.80, "silence_ratio": 0.10}

    result = manager.compare("hello", baseline_scores, current_scores)

    assert result["passed"] is False
    assert "dnsmos_sig" in result["regressed_metrics"]
    assert "dnsmos_ovrl" not in result["regressed_metrics"]  # improved


def test_baseline_manager_compare_all_pass():
    """When all metrics improve or stay stable, should pass."""
    manager = BaselineManager(regression_threshold_pct=5.0)

    baseline_scores = {"dnsmos_sig": 3.50, "dnsmos_ovrl": 3.50}
    current_scores = {"dnsmos_sig": 3.60, "dnsmos_ovrl": 3.48}

    result = manager.compare("hello", baseline_scores, current_scores)

    assert result["passed"] is True
    assert len(result["regressed_metrics"]) == 0


def test_baseline_manager_compare_baseline_file_sample_not_found():
    """When comparing a sample not in baseline, should return error result."""
    manager = BaselineManager(regression_threshold_pct=5.0)

    baseline = {
        "version": "test-v1",
        "commit": "abc",
        "branch": "test",
        "date": "2026-05-03",
        "samples": {"other_sample": {"dnsmos_sig": 3.5}},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_baseline(baseline, path)

        result = manager.compare_baseline_file(
            path, {"missing_sample": {"dnsmos_sig": 3.0}}, "missing_sample"
        )

        assert result["passed"] is False
        assert "not found in baseline" in result["details"]
