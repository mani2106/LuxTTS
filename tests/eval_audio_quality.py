"""LuxTTS Audio Quality Evaluation CLI.

Three commands for agent-driven quality analysis:
    python tests/eval_audio_quality.py score              # Score all audio in manifest
    python tests/eval_audio_quality.py score --no-sim     # Skip speaker similarity (faster)
    python tests/eval_audio_quality.py compare            # Compare scores to baseline
    python tests/eval_audio_quality.py save-baseline NAME # Save current scores as baseline

Workflow:
    1. pytest tests/audio_quality/ -m gpu              # Generate audio, run gates
    2. python tests/eval_audio_quality.py score         # Score generated audio
    3. python tests/eval_audio_quality.py compare       # Compare against baseline
    4. python tests/eval_audio_quality.py save-baseline master_v2  # If scores improved
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.audio_quality.scorers.scorer_registry import ScoreResult, results_to_json
from tests.audio_quality.suites.regression import BaselineManager, load_baseline, save_baseline

BASELINES_DIR = Path(__file__).parent / "audio_quality" / "baselines"
OUTPUT_DIR = Path(__file__).parent / "audio_quality" / "output"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
LATEST_SCORES_PATH = BASELINES_DIR / "latest_scores.json"


def _get_git_info():
    """Get current commit and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        commit = "unknown"
        branch = "unknown"
    return commit, branch


def cmd_score(skip_similarity=False):
    """Score all audio files listed in the manifest."""
    import librosa

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run GPU tests first: pytest tests/audio_quality/ -m gpu")
        sys.exit(1)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        print("ERROR: Manifest is empty. No audio to score.")
        sys.exit(1)

    print("SCORING RESULTS")
    print("=" * 70)
    print(f"Scoring {len(manifest)} audio samples...\n")

    results = []

    for entry in manifest:
        sample_name = f"{entry['test_name']}_{entry['speaker']}"
        audio_path = Path(entry["audio_path"])

        if not audio_path.exists():
            print(f"  SKIP {sample_name}: audio file not found at {audio_path}")
            continue

        audio, sr = librosa.load(str(audio_path), sr=48000)
        audio = audio.astype(np.float32)
        scores = {}

        # DNSMOS (CPU, always available)
        try:
            from tests.audio_quality.scorers.versa_scorer import score_dnsmos
            dnsmos = score_dnsmos(audio, sr)
            scores.update(dnsmos)
        except (ImportError, RuntimeError) as e:
            print(f"  WARN: DNSMOS unavailable for {sample_name}: {e}")

        # Speaker similarity (GPU recommended)
        if not skip_similarity:
            ref_path = entry.get("speaker_ref_path")
            if ref_path and Path(ref_path).exists():
                try:
                    from tests.audio_quality.scorers.versa_scorer import score_speaker_similarity
                    ref_audio, _ = librosa.load(ref_path, sr=48000)
                    ref_audio = ref_audio.astype(np.float32)
                    sim = score_speaker_similarity(audio, ref_audio, sr, use_gpu=True)
                    scores["speaker_similarity"] = sim["speaker_similarity"]
                except (ImportError, RuntimeError) as e:
                    print(f"  WARN: Speaker similarity unavailable for {sample_name}: {e}")

        # Silence artifacts
        try:
            from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts
            artifacts = detect_silence_artifacts(audio, sr)
            scores["trailing_silence_ms"] = artifacts["trailing_silence_ms"]
            scores["silence_ratio"] = artifacts["silence_ratio"]
            scores["peak_amplitude"] = artifacts["peak_amplitude"]
        except ImportError:
            pass

        scores["duration_s"] = entry["duration_s"]

        # Format console output
        sig = scores.get("dnsmos_sig", 0)
        bak = scores.get("dnsmos_bak", 0)
        ovrl = scores.get("dnsmos_ovrl", 0)
        sim_str = f"SIM={scores.get('speaker_similarity', 0):.2f}" if "speaker_similarity" in scores else "SIM=N/A"
        print(f"  {sample_name:40s} SIG={sig:.2f} BAK={bak:.2f} OVRL={ovrl:.2f}  {sim_str}")

        results.append(ScoreResult(
            sample_name=sample_name,
            scores=scores,
            passed=True,
            details="Scored successfully.",
        ))

    # Save results
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    results_to_json(results, str(LATEST_SCORES_PATH))
    print(f"\nScores saved to {LATEST_SCORES_PATH}")
    print("Next: python tests/eval_audio_quality.py compare")


def cmd_compare(baseline_name="master_baseline", threshold=5.0):
    """Compare latest scores against a stored baseline."""
    if not LATEST_SCORES_PATH.exists():
        print("ERROR: No scores to compare. Run 'score' first.")
        sys.exit(1)

    baseline_path = BASELINES_DIR / f"{baseline_name}.json"
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found: {baseline_path}")
        print(f"Run 'save-baseline {baseline_name}' to create one.")
        sys.exit(1)

    with open(LATEST_SCORES_PATH, encoding="utf-8") as f:
        latest_data = json.load(f)

    baseline = load_baseline(baseline_path)
    baseline_samples = baseline.get("samples", {})

    # Build current scores dict: sample_name -> scores
    current = {}
    for entry in latest_data:
        current[entry["sample_name"]] = entry["scores"]

    print("REGRESSION CHECK")
    print("=" * 70)
    print(f"Comparing {len(current)} samples against {baseline_name}")
    print(f"  Baseline: {baseline.get('version', '?')} (commit {baseline.get('commit', '?')}, {baseline.get('date', '?')})")
    print(f"  Threshold: {threshold}%\n")

    manager = BaselineManager(regression_threshold_pct=threshold)
    all_results = []
    any_regressed = False

    for sample_name, scores in current.items():
        if sample_name not in baseline_samples:
            print(f"  NEW  {sample_name} (not in baseline)")
            continue

        result = manager.compare(sample_name, baseline_samples[sample_name], scores)
        all_results.append(result)

        if result["passed"]:
            print(f"  PASS {sample_name}")
        else:
            any_regressed = True
            print(f"  FAIL {sample_name}")
            for line in result["details"].split("\n"):
                print(f"       {line}")

    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    print(f"\n{passed}/{total} samples within {threshold}% of baseline")

    if any_regressed:
        sys.exit(1)


def cmd_save_baseline(name):
    """Save current scores as a named baseline with full reproducibility metadata."""
    if not LATEST_SCORES_PATH.exists():
        print("ERROR: No scores to save. Run 'score' first.")
        sys.exit(1)

    with open(LATEST_SCORES_PATH, encoding="utf-8") as f:
        latest_data = json.load(f)

    commit, branch = _get_git_info()

    # Extract generation config from manifest if available
    gen_config = {}
    speakers_used = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest:
            first_entry = manifest[0]
            gen_config = first_entry.get("generation_config", {})
            for entry in manifest:
                speakers_used.add(entry.get("speaker", ""))

    # Convert list of ScoreResult dicts to samples dict
    samples = {}
    for entry in latest_data:
        samples[entry["sample_name"]] = entry["scores"]

    baseline = {
        "version": f"{name}",
        "commit": commit,
        "branch": branch,
        "date": _get_date(),
        "generation_config": gen_config,
        "speakers": sorted(s for s in speakers_used if s),
        "samples": samples,
    }

    out_path = BASELINES_DIR / f"{name}.json"
    save_baseline(baseline, out_path)
    print(f"Baseline saved to {out_path}")
    print(f"  {len(samples)} samples, commit {commit}, branch {branch}")
    if gen_config:
        print(f"  Generation config: steps={gen_config.get('num_steps')}, "
              f"guidance={gen_config.get('guidance_scale')}, seed={gen_config.get('seed')}")
    print(f"  Speakers: {', '.join(sorted(speakers_used)) if speakers_used else 'unknown'}")


def _get_date():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "score":
        skip_sim = "--no-sim" in sys.argv
        cmd_score(skip_similarity=skip_sim)
    elif command == "compare":
        baseline_name = "master_baseline"
        threshold = 5.0
        for arg in sys.argv[2:]:
            if arg.startswith("--baseline="):
                baseline_name = arg.split("=", 1)[1]
            elif arg.startswith("--threshold="):
                threshold = float(arg.split("=", 1)[1])
        cmd_compare(baseline_name=baseline_name, threshold=threshold)
    elif command == "save-baseline":
        if len(sys.argv) < 3:
            print("Usage: python tests/eval_audio_quality.py save-baseline <name>")
            sys.exit(1)
        cmd_save_baseline(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print("Commands: score, compare, save-baseline")
        sys.exit(1)


if __name__ == "__main__":
    main()
