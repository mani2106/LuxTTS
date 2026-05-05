"""LuxTTS Audio Quality Evaluation Pipeline.

Orchestrates the full evaluation workflow:
    python tests/run_audio_eval.py                  # Full pipeline
    python tests/run_audio_eval.py --skip-generate  # Score + compare only (skip GPU generation)
    python tests/run_audio_eval.py --no-sim         # Skip speaker similarity
    python tests/run_audio_eval.py --baseline NAME  # Compare against specific baseline
    python tests/run_audio_eval.py --threshold PCT  # Regression threshold (default: 5.0)

Outputs structured JSON to stdout with all results for downstream interpretation.

Exit codes:
    0 — All steps passed
    1 — Gate failures or regressions detected
    2 — Pipeline error (missing deps, bad config)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

BASELINES_DIR = Path(__file__).parent / "audio_quality" / "baselines"
OUTPUT_DIR = Path(__file__).parent / "audio_quality" / "output"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
LATEST_SCORES_PATH = BASELINES_DIR / "latest_scores.json"


def _run(cmd, label):
    """Run a subprocess command, capture output, return (success, stdout, stderr)."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {label}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode == 0, result.stdout, result.stderr


def step_generate():
    """Generate audio samples via pytest GPU tests."""
    cmd = [
        str(VENV_PYTHON), "-m", "pytest",
        "tests/audio_quality/", "-m", "gpu", "-v",
        "--tb=short",
    ]
    success, stdout, stderr = _run(cmd, "Generate Audio Samples")
    if not success:
        return {
            "step": "generate",
            "passed": False,
            "error": "Gate tests failed. Check output above for details.",
        }

    if not MANIFEST_PATH.exists():
        return {
            "step": "generate",
            "passed": False,
            "error": f"Manifest not found at {MANIFEST_PATH} after generation.",
        }

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    return {
        "step": "generate",
        "passed": True,
        "samples_generated": len(manifest),
        "speakers": sorted(set(e["speaker"] for e in manifest)),
        "test_types": sorted(set(e["test_name"] for e in manifest)),
    }


def step_score(skip_similarity=False):
    """Score all generated samples."""
    cmd = [str(VENV_PYTHON), "tests/eval_audio_quality.py", "score"]
    if skip_similarity:
        cmd.append("--no-sim")

    success, stdout, stderr = _run(cmd, "Score Audio Samples")
    if not success:
        return {
            "step": "score",
            "passed": False,
            "error": "Scoring failed. Check output above.",
        }

    if not LATEST_SCORES_PATH.exists():
        return {
            "step": "score",
            "passed": False,
            "error": f"Scores file not found at {LATEST_SCORES_PATH}.",
        }

    with open(LATEST_SCORES_PATH, encoding="utf-8") as f:
        scores = json.load(f)

    # Summarize scores
    metrics = {}
    for entry in scores:
        for key, value in entry["scores"].items():
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(value)

    averages = {k: round(sum(v) / len(v), 3) for k, v in metrics.items()}

    # Group by speaker
    by_speaker = {}
    for entry in scores:
        speaker = entry["sample_name"].rsplit("_", 1)[0] if "_" in entry["sample_name"] else entry["sample_name"]
        by_speaker.setdefault(speaker, []).append(entry["scores"])

    speaker_averages = {}
    for speaker, score_list in by_speaker.items():
        speaker_metrics = {}
        for s in score_list:
            for key, value in s.items():
                if isinstance(value, (int, float)):
                    speaker_metrics.setdefault(key, []).append(value)
        speaker_averages[speaker] = {
            k: round(sum(v) / len(v), 3) for k, v in speaker_metrics.items()
        }

    # Separate raw vs post-processed
    raw_scores = [e for e in scores if "raw_tts" in e["sample_name"]]
    postproc_scores = [e for e in scores if "basic_speech" in e["sample_name"]]
    vocalization_scores = [e for e in scores if "vocalization_" in e["sample_name"]]
    batch_scores = [e for e in scores if "batch_" in e["sample_name"]]

    return {
        "step": "score",
        "passed": True,
        "total_samples": len(scores),
        "averages": averages,
        "by_speaker": speaker_averages,
        "groups": {
            "raw": [e["sample_name"] for e in raw_scores],
            "post_processed": [e["sample_name"] for e in postproc_scores],
            "vocalization": [e["sample_name"] for e in vocalization_scores],
            "batch": [e["sample_name"] for e in batch_scores],
        },
    }


def step_compare(baseline_name="master_baseline", threshold=5.0):
    """Compare scores against baseline."""
    baseline_path = BASELINES_DIR / f"{baseline_name}.json"
    if not baseline_path.exists():
        return {
            "step": "compare",
            "passed": False,
            "skipped": True,
            "error": f"Baseline '{baseline_name}' not found at {baseline_path}. "
                     f"Run with save-baseline to create one.",
        }

    cmd = [
        str(VENV_PYTHON), "tests/eval_audio_quality.py", "compare",
        f"--baseline={baseline_name}",
        f"--threshold={threshold}",
    ]

    success, stdout, stderr = _run(cmd, f"Compare Against Baseline ({baseline_name})")

    # Parse regression info from stdout
    regressions = []
    lines = stdout.split("\n") if stdout else []
    current_sample = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("FAIL "):
            current_sample = stripped.replace("FAIL ", "")
        elif current_sample and "->" in stripped and "%" in stripped:
            regressions.append({
                "sample": current_sample,
                "detail": stripped.strip(),
            })
        elif stripped.startswith("PASS "):
            current_sample = None

    return {
        "step": "compare",
        "passed": success,
        "baseline": baseline_name,
        "threshold_pct": threshold,
        "regressions": regressions,
        "regression_count": len(regressions),
    }


def main():
    parser = argparse.ArgumentParser(description="LuxTTS Audio Quality Evaluation Pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip audio generation step (score + compare only)")
    parser.add_argument("--no-sim", action="store_true",
                        help="Skip speaker similarity scoring (faster)")
    parser.add_argument("--baseline", default="master_baseline",
                        help="Baseline name to compare against (default: master_baseline)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold percentage (default: 5.0)")
    args = parser.parse_args()

    if not VENV_PYTHON.exists():
        print(f"ERROR: Virtual environment not found at {VENV_PYTHON}", file=sys.stderr)
        print("Run: uv venv && uv pip install -r requirements.txt", file=sys.stderr)
        sys.exit(2)

    results = {
        "pipeline": "audio-quality-eval",
        "args": vars(args),
    }

    # Step 1: Generate
    if args.skip_generate:
        print("Skipping generation (--skip-generate)")
        if not MANIFEST_PATH.exists():
            print(f"ERROR: No manifest at {MANIFEST_PATH}. Run without --skip-generate first.", file=sys.stderr)
            sys.exit(2)
        results["generate"] = {"step": "generate", "passed": True, "skipped": True}
    else:
        results["generate"] = step_generate()
        if not results["generate"]["passed"]:
            results["success"] = False
            print(json.dumps(results, indent=2))
            sys.exit(1)

    # Step 2: Score
    results["score"] = step_score(skip_similarity=args.no_sim)
    if not results["score"]["passed"]:
        results["success"] = False
        print(json.dumps(results, indent=2))
        sys.exit(1)

    # Step 3: Compare
    results["compare"] = step_compare(
        baseline_name=args.baseline,
        threshold=args.threshold,
    )

    overall_passed = (
        results["generate"]["passed"]
        and results["score"]["passed"]
        and results["compare"]["passed"]
    )
    results["success"] = overall_passed

    # Output structured JSON
    print(f"\n{'=' * 60}")
    print("PIPELINE RESULT")
    print(f"{'=' * 60}")
    print(json.dumps(results, indent=2))

    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
