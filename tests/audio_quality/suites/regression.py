"""Tier 3: Regression baseline management.

Handles saving, loading, and comparing audio quality scores against
stored baselines. Designed for A/B comparison across branches and commits.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from tests.audio_quality.scorers.scorer_registry import (
    compare_scores,
    format_regression_details,
)

logger = logging.getLogger(__name__)


def load_baseline(path: Path) -> dict:
    """Load a baseline JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(baseline: dict, path: Path):
    """Save a baseline JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)


def create_baseline(
    version: str,
    commit: str,
    branch: str,
    config: dict,
    samples: dict,
) -> dict:
    """Create a new baseline dictionary in the standard format."""
    return {
        "version": version,
        "commit": commit,
        "branch": branch,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": config,
        "samples": samples,
    }


class BaselineManager:
    """Manages baseline comparison for regression detection.

    Produces agent-parseable results:
    - Which metrics regressed, with exact baseline/current/delta values
    - Pass/fail determination based on configurable threshold
    - Actionable details string a coding agent can use to diagnose issues
    """

    def __init__(self, regression_threshold_pct: float = 5.0):
        self.threshold_pct = regression_threshold_pct

    def compare(
        self,
        sample_name: str,
        baseline_scores: dict,
        current_scores: dict,
    ) -> dict:
        """Compare current scores against baseline for one sample.

        Returns:
            {
                "sample_name": str,
                "passed": bool,
                "regressed_metrics": list[str],
                "deltas": dict,
                "details": str,
            }
        """
        deltas = compare_scores(baseline_scores, current_scores, self.threshold_pct)
        regressed = [k for k, v in deltas.items() if v["regressed"]]
        passed = len(regressed) == 0

        if passed:
            details = f"All metrics within {self.threshold_pct}% of baseline for {sample_name}."
        else:
            details = format_regression_details(
                sample_name, baseline_scores, current_scores, self.threshold_pct
            )

        return {
            "sample_name": sample_name,
            "passed": passed,
            "regressed_metrics": regressed,
            "deltas": deltas,
            "details": details,
        }

    def compare_baseline_file(
        self,
        baseline_path: Path,
        current_scores: dict,
        sample_name: Optional[str] = None,
    ) -> dict:
        """Load a baseline file and compare against current scores.

        If sample_name is provided, compares only that sample.
        Otherwise compares all samples in the baseline.
        """
        baseline = load_baseline(baseline_path)

        if sample_name:
            if sample_name not in baseline["samples"]:
                return {
                    "sample_name": sample_name,
                    "passed": False,
                    "regressed_metrics": [],
                    "deltas": {},
                    "details": f"Sample '{sample_name}' not found in baseline file: {baseline_path}",
                }
            return self.compare(
                sample_name,
                baseline["samples"][sample_name],
                current_scores,
            )

        # Compare all samples
        results = []
        for name, scores in baseline["samples"].items():
            if name in current_scores:
                results.append(self.compare(name, scores, current_scores[name]))

        return {
            "passed": all(r["passed"] for r in results),
            "results": results,
            "baseline_info": {
                "version": baseline["version"],
                "commit": baseline["commit"],
                "branch": baseline["branch"],
                "date": baseline["date"],
            },
        }
