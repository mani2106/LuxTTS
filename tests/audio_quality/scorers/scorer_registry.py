"""Scorer registry: compose metric suites, run collections, produce agent-parseable reports.

All test outputs are designed to be easily understood by both humans and coding agents:
- Structured JSON with descriptive field names
- Actionable failure messages with baseline/current/delta/threshold
- Summary reports with PASS/FAIL per sample per metric
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Structured result for a single sample's evaluation.

    Designed to be serialized to JSON and parsed by coding agents.
    """

    sample_name: str
    scores: dict
    passed: bool
    details: str = ""
    baseline_scores: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "sample_name": self.sample_name,
            "scores": self.scores,
            "passed": self.passed,
            "details": self.details,
        }
        if self.baseline_scores is not None:
            d["baseline_scores"] = self.baseline_scores
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compare_scores(
    baseline: dict,
    current: dict,
    threshold_pct: float = 5.0,
) -> dict:
    """Compare current scores against baseline and flag regressions.

    Args:
        baseline: Baseline scores (e.g., from stored JSON)
        current: Current run scores
        threshold_pct: Flag if any metric drops more than this percentage

    Returns:
        Dict mapping metric name to {
            "baseline": float,
            "current": float,
            "delta": float,
            "delta_pct": float,
            "regressed": bool,
        }
    """
    deltas = {}
    for key in baseline:
        if key not in current:
            continue
        b = baseline[key]
        c = current[key]
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
            continue

        delta = c - b
        if abs(b) < 1e-10:
            continue
        delta_pct = (delta / abs(b)) * 100
        regressed = delta_pct < -threshold_pct

        deltas[key] = {
            "baseline": b,
            "current": c,
            "delta": delta,
            "delta_pct": delta_pct,
            "regressed": regressed,
        }
    return deltas


def format_regression_details(
    sample_name: str,
    baseline: dict,
    current: dict,
    threshold_pct: float = 5.0,
) -> str:
    """Produce actionable failure message for a regression.

    Format: "REGRESSION in {sample}: {metric} dropped from {baseline} to {current} ({delta}%): {description}"
    """
    deltas = compare_scores(baseline, current, threshold_pct)
    regressed = {k: v for k, v in deltas.items() if v["regressed"]}

    if not regressed:
        return f"All metrics within {threshold_pct}% of baseline."

    lines = []
    for key, info in sorted(regressed.items(), key=lambda x: x[1]["delta_pct"]):
        lines.append(
            f"  {key}: {info['baseline']:.2f} -> {info['current']:.2f} "
            f"({info['delta_pct']:+.1f}%, threshold: -{threshold_pct}%)"
        )

    header = f"REGRESSION in {sample_name}: {len(regressed)} metric(s) dropped >{threshold_pct}%"
    return header + "\n" + "\n".join(lines)


def format_report(results: list) -> str:
    """Format a list of ScoreResults into a human + agent readable report.

    Designed so a coding agent can scan the output and immediately understand:
    - Which samples passed/failed
    - What regressed and by how much
    - What to fix
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AUDIO QUALITY EVALUATION REPORT")
    lines.append("=" * 60)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    lines.append(f"\nSummary: {passed}/{len(results)} PASSED, {failed} FAILED\n")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.sample_name}")
        if not result.passed:
            for detail_line in result.details.split("\n"):
                lines.append(f"  {detail_line}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def results_to_json(results: list, path: str):
    """Save results to a JSON file for programmatic consumption."""
    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
