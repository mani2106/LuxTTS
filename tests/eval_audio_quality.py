"""CLI runner for audio quality evaluation.

Usage:
    python tests/eval_audio_quality.py --generate-baselines   # Create/update baseline
    python tests/eval_audio_quality.py --compare-baselines     # Compare current vs baseline
    python tests/eval_audio_quality.py --report                # Full JSON report
    python tests/eval_audio_quality.py --list-cases            # List all test cases
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.audio_quality.suites.fast_ci import FAST_CI_CASES
from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES
from tests.audio_quality.suites.regression import BaselineManager, create_baseline, load_baseline, save_baseline
from tests.audio_quality.scorers.scorer_registry import ScoreResult, format_report, results_to_json

BASELINES_DIR = Path(__file__).parent / "audio_quality" / "baselines"


def list_cases():
    """List all test cases across all tiers."""
    print("\n=== Tier 1: Fast CI (CPU) ===")
    for case in FAST_CI_CASES:
        tags = []
        if case.has_vocalization_tags:
            tags.append("vocalization")
        if case.is_batch:
            tags.append(f"batchx{case.batch_count}")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {case.name}: {case.description}{tag_str}")
        print(f"    Text: \"{case.text[:60]}{'...' if len(case.text) > 60 else ''}\"")

    print("\n=== Tier 2: Full Eval (GPU) ===")
    for case in FULL_EVAL_CASES:
        tags = []
        if not case.enable_post_processing:
            tags.append("no-postproc")
        if case.has_vocalization_tags:
            tags.append("vocalization")
        if case.is_batch:
            tags.append(f"batchx{case.batch_count}")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        print(f"  {case.name}: {case.description}{tag_str}")
        print(f"    Text: \"{case.text[:60]}{'...' if len(case.text) > 60 else ''}\"")


def compare_baselines(baseline_name: str = "master_baseline", threshold: float = 5.0):
    """Compare current results against a stored baseline."""
    baseline_path = BASELINES_DIR / f"{baseline_name}.json"
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found: {baseline_path}")
        print(f"Run --generate-baselines first.")
        sys.exit(1)

    baseline = load_baseline(baseline_path)
    print(f"Loaded baseline: {baseline['version']} (commit {baseline['commit']}, {baseline['date']})")
    print(f"Contains {len(baseline['samples'])} sample(s)")

    manager = BaselineManager(regression_threshold_pct=threshold)

    print(f"\nComparing with {threshold}% regression threshold...")
    print("(Note: This compares stored scores. For fresh generation, use Tier 2 tests.)\n")

    # For now, just display the baseline scores
    for name, scores in baseline["samples"].items():
        print(f"  {name}:")
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.3f}")
            else:
                print(f"    {metric}: {value}")

    print(f"\nTo compare against current output, run:")
    print(f"  pytest tests/audio_quality/test_full_eval.py -v -m gpu")


def generate_baselines(
    version: str = "v1",
    baseline_name: str = "master_baseline",
):
    """Generate a baseline file from current test results."""
    import subprocess

    # Get current commit
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"
        branch = "unknown"

    print(f"Generating baseline: {baseline_name}")
    print(f"Version: {version}")
    print(f"Commit: {commit}")
    print(f"Branch: {branch}")
    print()
    print("To generate baselines with actual scores:")
    print("  1. Run: pytest tests/audio_quality/test_full_eval.py -v -m gpu --json-report")
    print(f"  2. Pipe results to: python tests/eval_audio_quality.py --save-baselines < results.json")
    print()
    print("Or create an empty baseline template:")
    template = create_baseline(
        version=version,
        commit=commit,
        branch=branch,
        config={
            "enable_post_processing": False,
            "num_steps": 4,
            "guidance_scale": 3.0,
        },
        samples={},
    )

    out_path = BASELINES_DIR / f"{baseline_name}.json"
    save_baseline(template, out_path)
    print(f"  Created template: {out_path}")
    print(f"  Fill in 'samples' with actual scores from Tier 2 evaluation.")


def main():
    parser = argparse.ArgumentParser(description="LuxTTS Audio Quality Evaluation CLI")
    parser.add_argument("--generate-baselines", action="store_true", help="Create/update baseline file")
    parser.add_argument("--compare-baselines", action="store_true", help="Compare current vs baseline")
    parser.add_argument("--list-cases", action="store_true", help="List all test cases")
    parser.add_argument("--baseline-name", default="master_baseline", help="Baseline file name (without .json)")
    parser.add_argument("--threshold", type=float, default=5.0, help="Regression threshold (%%)")
    parser.add_argument("--version", default="v1", help="Baseline version label")

    args = parser.parse_args()

    if args.list_cases:
        list_cases()
    elif args.generate_baselines:
        generate_baselines(version=args.version, baseline_name=args.baseline_name)
    elif args.compare_baselines:
        compare_baselines(baseline_name=args.baseline_name, threshold=args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
