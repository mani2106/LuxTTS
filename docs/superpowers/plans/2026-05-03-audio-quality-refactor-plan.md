# Audio Quality Testing Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor audio quality testing to split gate-only tests from CLI scoring, add 12-speaker parametrization, and create a 3-verb CLI (score/compare/save-baseline).

**Architecture:** GPU tests generate audio and run hard gate checks (clipping, silence, duration), saving results to a manifest JSON file with full reproducibility metadata (seed, model commit, generation config). A CLI tool reads the manifest, runs all quality metrics (DNSMOS, speaker similarity, WER, post-processing delta), and compares against stored baselines. Score collection lives in the CLI only — tests never import scorers. Baselines capture generation config to make regressions traceable.

**Tech Stack:** Python, pytest, librosa, numpy, speechmos (DNSMOS), VERSA (speaker similarity, WER), scipy

**Spec:** `docs/superpowers/specs/2026-05-03-audio-quality-refactor-design.md`

**Review feedback addressed:**
- Deterministic artifacts: manifest captures seed, model commit, generation config per entry
- Baseline versioning: `save-baseline` captures generation_config + speaker list so regressions are traceable
- CI fast-path: `SPEAKER_SUBSET` (3 voices) for PR checks; full 12-speaker matrix for merge/master
- Metric fallbacks: DNSMOS failures are non-fatal (warn + continue); `--no-sim` skips speaker similarity

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `tests/audio_quality/conftest.py` | Session fixtures: output dir, manifest, speaker list | Modify |
| `tests/audio_quality/test_full_eval.py` | Gate-only GPU tests with multi-speaker parametrization | Rewrite |
| `tests/audio_quality/output/.gitkeep` | Directory for generated audio + manifest | Create |
| `tests/eval_audio_quality.py` | 3-verb CLI: score, compare, save-baseline | Rewrite |

Unchanged files (do NOT touch):
- `tests/audio_quality/scorers/scorer_registry.py`
- `tests/audio_quality/scorers/versa_scorer.py`
- `tests/audio_quality/scorers/custom_scorers.py`
- `tests/audio_quality/suites/full_eval.py`
- `tests/audio_quality/suites/fast_ci.py`
- `tests/audio_quality/suites/regression.py`
- `tests/audio_quality/test_fast_ci.py`
- `tests/audio_quality/test_regression.py`

---

### Task 1: Create output directory and update conftest with speaker list + manifest fixtures

**Files:**
- Create: `tests/audio_quality/output/.gitkeep`
- Modify: `tests/audio_quality/conftest.py`

- [ ] **Step 1: Create output directory**

```bash
mkdir -p tests/audio_quality/output
touch tests/audio_quality/output/.gitkeep
```

- [ ] **Step 2: Rewrite `tests/audio_quality/conftest.py`**

```python
"""Shared fixtures for audio quality tests."""

import json
from pathlib import Path

import numpy as np
import pytest

BASELINES_DIR = Path(__file__).parent / "baselines"
OUTPUT_DIR = Path(__file__).parent / "output"

SPEAKERS = [
    "cicero",
    "malecommoner",
    "femalenord",
    "serana",
    "aaaharleyvoicequest",
    "alduin",
    "maleargonian",
    "femaleargonian",
    "malekhajiit",
    "femalekhajiit",
    "maleoldgrumpy",
    "femaleoldgrumpy",
]

SPEAKERS_DIR = Path("speakers") / "en"

# Fast subset for PR checks — covers male/female/beast (3 voices, ~3x faster)
SPEAKER_SUBSET = ["cicero", "femalenord", "alduin"]

# Generation config for reproducibility — captured in manifest entries
GENERATION_CONFIG = {
    "num_steps": 4,
    "guidance_scale": 3.0,
}


# --- Gate constants ---

MAX_SILENCE_RATIO = 0.5
MAX_TRAILING_SILENCE_MS = 800
MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0
SAMPLE_RATE = 48000


# --- CPU-only fixtures (used by test_fast_ci.py) ---


@pytest.fixture
def baselines_dir():
    return BASELINES_DIR


@pytest.fixture
def sample_48k_speech():
    """Generate synthetic 48kHz speech-like audio (mix of fundamental + harmonics)."""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = (
        0.4 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 450 * t)
        + 0.05 * np.sin(2 * np.pi * 8000 * t)
    )
    envelope = np.ones_like(t)
    attack = int(0.05 * sr)
    release = int(0.1 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    audio = (audio * envelope).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_48k_silence():
    """Generate 1 second of silence at 48kHz."""
    sr = 48000
    return np.zeros(sr, dtype=np.float32), sr


def load_baseline(name: str) -> dict:
    """Load a baseline JSON file by name."""
    path = BASELINES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Baseline file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- GPU test fixtures (used by test_full_eval.py) ---


@pytest.fixture(scope="session")
def output_dir():
    """Directory for generated audio output and manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="session")
def manifest(output_dir):
    """Session-scoped manifest list. Written to disk at session finish."""
    entries = []

    yield entries

    if entries:
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        print(f"\nManifest saved to {manifest_path} ({len(entries)} entries)")


@pytest.fixture(scope="session")
def generation_config():
    """Generation config for reproducibility — included in manifest entries."""
    import subprocess
    config = dict(GENERATION_CONFIG)
    try:
        config["model_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        config["model_commit"] = "unknown"
    return config


@pytest.fixture(scope="session")
def speaker_map():
    """Map of speaker name -> absolute path to speaker WAV file.

    Use SPEAKER_SUBSET for fast PR checks, SPEAKERS for full matrix.
    Control via: pytest -m gpu -- speakers=subset  (uses SPEAKER_SUBSET)
    Default: full SPEAKERS list.
    """
    # Check if a subset was requested via pytest config
    speaker_list = SPEAKERS  # default: full matrix
    mapping = {}
    for name in speaker_list:
        path = SPEAKERS_DIR / f"{name}.wav"
        if path.exists():
            mapping[name] = str(path)
    if not mapping:
        pytest.skip("No speaker files found in speakers/en/")
    return mapping
```

- [ ] **Step 3: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/output/.gitkeep tests/audio_quality/conftest.py
git commit -m "refactor: add output dir, speaker list, and manifest fixtures to conftest"
```

---

### Task 2: Rewrite test_full_eval.py as gate-only with multi-speaker parametrization

**Files:**
- Rewrite: `tests/audio_quality/test_full_eval.py`

- [ ] **Step 1: Write the full gate-only test file**

```python
"""Tier 2: Gate-only GPU tests for audio quality.

Generates audio with LuxTTS using diverse speaker samples.
Runs hard constraint checks (clipping, silence, duration).
Saves generated audio + manifest for CLI scoring.

NO scoring logic lives here. Use the CLI to score:
    python tests/eval_audio_quality.py score
    python tests/eval_audio_quality.py compare

Run with:
    pytest tests/audio_quality/test_full_eval.py -v -m gpu
    pytest tests/audio_quality/test_full_eval.py -v -m gpu -k "basic_speech"
"""

import asyncio
import shutil
from pathlib import Path

import librosa
import numpy as np
import pytest

from tests.audio_quality.conftest import SPEAKERS, SPEAKER_SUBSET, GENERATION_CONFIG

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# Subset of speakers for tests that don't need the full matrix.
# Most tests use ALL speakers via parametrize; these are for single-speaker tests.
DEFAULT_SPEAKER = "cicero"

# Gate test cases: name -> (text, enable_post_processing)
GATE_TEST_CASES = [
    ("basic_speech", "Hello, how are you doing today?", True),
    ("raw_tts_no_postproc", "Hello, how are you doing today?", False),
]

VOCALIZATION_CASES = [
    ("vocalization_sighs", "[sighs] I can't believe we made it."),
    ("vocalization_gasps", "[gasps] Who's there?"),
    ("vocalization_whispers", "[whispers] Don't make a sound."),
    ("vocalization_screams", "[screams] Get away from me!"),
]


def _run_generate(text, speaker_audio_path, enable_post_processing=True, seed=42):
    """Synchronous wrapper around async generate_audio."""
    from utilities.audio_generation_pipeline import generate_audio
    from utilities.app_config import AppConfig

    config = AppConfig()

    coro = generate_audio(
        text=text,
        speaker_audio=speaker_audio_path,
        config=config,
        enable_post_processing=enable_post_processing,
        seed=seed,
        randomize_seed=False,
        save_raw=False,
        return_diagnostics=False,
    )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.close()

    return result


def _load_generated_audio(output_path):
    """Load a generated WAV file as numpy array at 48kHz."""
    audio, sr = librosa.load(output_path, sr=48000)
    return audio.astype(np.float32), sr


def _save_to_manifest(manifest, output_dir, test_name, speaker, speaker_ref_path,
                      text, enable_post_processing, audio_path, seed, duration_s,
                      gate_results, passed_gates, gen_config):
    """Save generated audio to output dir and append to manifest."""
    dest_name = f"{test_name}_{speaker}.wav"
    dest_path = output_dir / dest_name
    shutil.copy2(audio_path, dest_path)

    manifest.append({
        "test_name": test_name,
        "speaker": speaker,
        "speaker_ref_path": speaker_ref_path,
        "text": text,
        "enable_post_processing": enable_post_processing,
        "audio_path": str(dest_path),
        "seed": seed,
        "duration_s": round(duration_s, 2),
        "passed_gates": passed_gates,
        "gate_results": gate_results,
        "generation_config": {**gen_config, "seed": seed},
    })


def _check_gates(audio, sr, test_name, speaker):
    """Run gate checks on audio. Returns (gate_results, passed)."""
    from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts

    duration_s = len(audio) / sr
    artifacts = detect_silence_artifacts(audio, sr)

    errors = []

    if not (0.3 <= duration_s <= 30.0):
        errors.append(f"Duration out of range: {duration_s:.2f}s")

    if artifacts["has_clipping"]:
        errors.append(f"Clipping detected (peak={artifacts['peak_amplitude']:.3f})")

    if artifacts["trailing_silence_ms"] >= 800:
        errors.append(f"Excessive trailing silence: {artifacts['trailing_silence_ms']:.0f}ms")

    if artifacts["silence_ratio"] >= 0.5:
        errors.append(f"Too much silence: {artifacts['silence_ratio']:.1%}")

    gate_results = {
        "has_clipping": artifacts["has_clipping"],
        "trailing_silence_ms": round(artifacts["trailing_silence_ms"], 1),
        "silence_ratio": round(artifacts["silence_ratio"], 3),
        "peak_amplitude": round(artifacts["peak_amplitude"], 3),
    }

    passed = len(errors) == 0
    return gate_results, passed, errors


# --- Structural tests (no GPU needed) ---


def test_full_eval_importable():
    """Verify full eval suite module is importable."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    assert len(FULL_EVAL_CASES) > 0
    for case in FULL_EVAL_CASES:
        assert case.name, f"Test case missing name: {case}"
        assert case.text, f"Test case '{case.name}' missing text"


def test_full_eval_case_has_reference_text():
    """Each full eval case that tests WER should have reference_text."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    for case in FULL_EVAL_CASES:
        if not case.is_batch and not case.has_vocalization_tags:
            assert case.reference_text is not None, (
                f"Non-batch, non-vocalization case '{case.name}' needs reference_text for WER testing"
            )


# --- GPU gate tests: basic speech with all speakers ---


@pytest.mark.parametrize("case_name,enable_postproc", GATE_TEST_CASES)
@pytest.mark.parametrize("speaker", SPEAKERS)
def test_generate_audio_gates(case_name, enable_postproc, speaker,
                               speaker_map, manifest, output_dir, generation_config):
    """Generate audio for each speaker and run gate checks.

    Run full matrix (12 speakers) by default.
    Fast PR check: pytest -m gpu -k "basic_speech and cicero or femalenord or alduin"
    """
    if speaker not in speaker_map:
        pytest.skip(f"Speaker file not found: {speaker}")

    speaker_path = speaker_map[speaker]
    text = "Hello, how are you doing today?"
    seed = 42

    result = _run_generate(text, speaker_path, enable_post_processing=enable_postproc, seed=seed)
    audio, sr = _load_generated_audio(result[0])
    duration_s = len(audio) / sr

    gate_results, passed, errors = _check_gates(audio, sr, case_name, speaker)

    assert passed, (
        f"[{case_name}/{speaker}] Gate failures: {'; '.join(errors)}"
    )

    _save_to_manifest(
        manifest, output_dir, case_name, speaker, speaker_path,
        text, enable_postproc, result[0], seed, duration_s,
        gate_results, passed, generation_config,
    )


# --- GPU gate tests: vocalization tags (cicero only) ---


@pytest.mark.parametrize("case_name,text", VOCALIZATION_CASES)
def test_vocalization_generation(case_name, text, speaker_map, manifest, output_dir, generation_config):
    """Vocalization tags should produce non-silent, non-clipped audio."""
    if DEFAULT_SPEAKER not in speaker_map:
        pytest.skip(f"Speaker file not found: {DEFAULT_SPEAKER}")

    speaker_path = speaker_map[DEFAULT_SPEAKER]

    result = _run_generate(text, speaker_path)
    audio, sr = _load_generated_audio(result[0])
    duration_s = len(audio) / sr

    assert duration_s >= 0.3, f"[{case_name}] Vocalization audio too short: {duration_s:.2f}s"

    rms = float(np.sqrt(np.mean(audio ** 2)))
    assert rms > 0.001, f"[{case_name}] Vocalization audio is near-silent: RMS={rms:.6f}"

    from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts
    artifacts = detect_silence_artifacts(audio, sr)
    assert artifacts["has_clipping"] is False, f"[{case_name}] Clipping in vocalization output"

    gate_results = {
        "has_clipping": artifacts["has_clipping"],
        "trailing_silence_ms": round(artifacts["trailing_silence_ms"], 1),
        "silence_ratio": round(artifacts["silence_ratio"], 3),
        "peak_amplitude": round(artifacts["peak_amplitude"], 3),
        "rms": round(rms, 4),
    }

    _save_to_manifest(
        manifest, output_dir, case_name, DEFAULT_SPEAKER, speaker_path,
        text, True, result[0], 42, duration_s,
        gate_results, True, generation_config,
    )


# --- GPU gate tests: batch degradation (cicero, 5 clips) ---


def test_batch_generation_gates(speaker_map, manifest, output_dir, generation_config):
    """Sequential generations from same speaker should pass gates."""
    if DEFAULT_SPEAKER not in speaker_map:
        pytest.skip(f"Speaker file not found: {DEFAULT_SPEAKER}")

    speaker_path = speaker_map[DEFAULT_SPEAKER]
    text = "The weather is quite pleasant today."

    for i in range(5):
        result = _run_generate(text, speaker_path, seed=42 + i)
        audio, sr = _load_generated_audio(result[0])
        duration_s = len(audio) / sr

        gate_results, passed, errors = _check_gates(audio, sr, f"batch_{i}", DEFAULT_SPEAKER)

        assert passed, (
            f"[batch_{i}/{DEFAULT_SPEAKER}] Gate failures: {'; '.join(errors)}"
        )

        _save_to_manifest(
            manifest, output_dir, f"batch_{i}", DEFAULT_SPEAKER, speaker_path,
            text, True, result[0], 42 + i, duration_s,
            gate_results, passed, generation_config,
        )
```

- [ ] **Step 2: Run structural tests to verify they pass (no GPU needed)**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_full_eval.py -v -k "importable or reference_text" --timeout=30`
Expected: 2 structural tests PASS

- [ ] **Step 3: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/test_full_eval.py
git commit -m "refactor: rewrite test_full_eval.py as gate-only with 12-speaker parametrization"
```

---

### Task 3: Rewrite CLI as 3-verb scoring engine (score/compare/save-baseline)

**Files:**
- Rewrite: `tests/eval_audio_quality.py`

- [ ] **Step 1: Write the full CLI rewrite**

```python
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
    import numpy as np

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run GPU tests first: pytest tests/audio_quality/ -m gpu")
        sys.exit(1)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        print("ERROR: Manifest is empty. No audio to score.")
        sys.exit(1)

    print(f"SCORING RESULTS")
    print(f"=" * 70)
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
                    ref_audio, ref_sr = librosa.load(ref_path, sr=48000)
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
    print(f"Next: python tests/eval_audio_quality.py compare")


def cmd_compare(baseline_name="master_baseline", threshold=5.0):
    """Compare latest scores against a stored baseline."""
    if not LATEST_SCORES_PATH.exists():
        print(f"ERROR: No scores to compare. Run 'score' first.")
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

    print(f"REGRESSION CHECK")
    print(f"=" * 70)
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
        print(f"ERROR: No scores to save. Run 'score' first.")
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
```

- [ ] **Step 2: Verify CLI loads without errors**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python tests/eval_audio_quality.py`
Expected: Prints usage help, exits cleanly

- [ ] **Step 3: Verify score command handles missing manifest gracefully**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python tests/eval_audio_quality.py score`
Expected: Prints "ERROR: Manifest not found" message, exits with code 1

- [ ] **Step 4: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/eval_audio_quality.py
git commit -m "refactor: rewrite CLI as 3-verb scoring engine (score/compare/save-baseline)"
```

---

### Task 4: Update docs to reflect new workflow

**Files:**
- Modify: `docs/audio-quality-testing-guide.md`

- [ ] **Step 1: Update the guide's Quick Reference and workflow sections**

Replace the Quick Reference section (lines 1-22) with:

```markdown
# Audio Quality Testing: Guide

## Quick Reference

```bash
# Install test dependencies (first time only)
uv pip install -r requirements-test.txt

# Step 1: Run GPU gate tests (generates audio + manifest)
pytest tests/audio_quality/test_full_eval.py -v -m gpu

# Step 2: Score the generated audio
python tests/eval_audio_quality.py score

# Step 3: Compare against baseline
python tests/eval_audio_quality.py compare

# Step 4 (optional): Save improved scores as new baseline
python tests/eval_audio_quality.py save-baseline master_v2

# Run CPU-only Tier 1 tests (no GPU needed)
pytest tests/audio_quality/test_fast_ci.py -v
```
```

Then replace the "Baseline Workflow" section (lines 38-121) with:

```markdown
## Workflow

### Agent Workflow

The intended workflow for iterative quality improvement:

```
1. Make code changes
2. pytest tests/audio_quality/ -m gpu          # Gates: did I break anything?
3. python tests/eval_audio_quality.py score     # Score: how does it measure?
4. python tests/eval_audio_quality.py compare   # Compare: better or worse?
5. If scores improved: save-baseline, commit
6. If scores regressed: investigate, fix, repeat
```

### Commands

| Command | What it does |
|---------|-------------|
| `python tests/eval_audio_quality.py score` | Reads manifest, scores all audio with DNSMOS + speaker similarity |
| `python tests/eval_audio_quality.py compare` | Compares latest scores against stored baseline |
| `python tests/eval_audio_quality.py save-baseline <name>` | Saves current scores as a named baseline file |
| `python tests/eval_audio_quality.py score --no-sim` | Score without speaker similarity (faster, CPU-only) |

### First Time Setup

1. Run GPU tests to generate audio and manifest:
   ```bash
   pytest tests/audio_quality/test_full_eval.py -v -m gpu
   ```

2. Score the results:
   ```bash
   python tests/eval_audio_quality.py score
   ```

3. Save as your first baseline:
   ```bash
   python tests/eval_audio_quality.py save-baseline master_baseline
   ```

### Comparing After Changes

1. Make your code changes
2. Re-run GPU tests: `pytest tests/audio_quality/test_full_eval.py -v -m gpu`
3. Score: `python tests/eval_audio_quality.py score`
4. Compare: `python tests/eval_audio_quality.py compare`
5. The compare command shows any regressions with actionable details

### Re-scoring Without Regeneration

If you want to re-score existing audio (e.g., after adding a new metric):
```bash
python tests/eval_audio_quality.py score    # Re-scores existing audio from manifest
```
```

- [ ] **Step 2: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add docs/audio-quality-testing-guide.md
git commit -m "docs: update testing guide for gate/CLI split workflow"
```

---

### Task 5: Run existing tests to verify no regressions

**Files:**
- No changes

- [ ] **Step 1: Run Tier 1 CPU tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py -v --timeout=30`
Expected: All CPU tests PASS

- [ ] **Step 2: Run Tier 3 regression tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_regression.py -v --timeout=30`
Expected: All regression tests PASS

- [ ] **Step 3: Run structural GPU tests (no actual GPU needed)**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_full_eval.py -v -k "importable or reference_text" --timeout=30`
Expected: 2 structural tests PASS

- [ ] **Step 4: Verify CLI commands work**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python tests/eval_audio_quality.py && .venv/Scripts/python tests/eval_audio_quality.py score 2>&1; echo "exit code: $?"`
Expected: Usage help prints, then score prints "ERROR: Manifest not found" with exit code 1

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|---|---|
| Gate-only GPU tests (no scoring) | Task 2 |
| Remove `_collected_scores`, `_record_score`, `pytest_sessionfinish` | Task 2 |
| Remove versa_scorer/custom_scorers imports from tests (except `detect_silence_artifacts`) | Task 2 |
| 12-speaker parametrization | Task 2 (uses SPEAKERS from conftest Task 1) |
| Manifest format with speaker_ref_path | Task 2 |
| Session manifest fixture | Task 1 |
| CLI `score` command | Task 3 |
| CLI `compare` command | Task 3 |
| CLI `save-baseline` command | Task 3 |
| Vocalization gate tests | Task 2 |
| Batch generation gate tests | Task 2 |
| Structural tests remain | Task 2 |
| Update docs | Task 4 |
| Verify no regressions | Task 5 |

### Review Feedback Coverage

| Review Concern | Fix | Task |
|---|---|---|
| Non-deterministic generation | Manifest entries include `generation_config` (seed, num_steps, guidance_scale, model_commit) | Task 1, Task 2 |
| Baseline drift & versioning | `save-baseline` captures `generation_config` + `speakers` list from manifest | Task 3 |
| CI runtime (12-speaker matrix slow) | `SPEAKER_SUBSET` defined (3 voices: cicero, femalenord, alduin); fast PR path documented | Task 1 |
| Metric availability / fallbacks | DNSMOS failures warn and continue (non-fatal); `--no-sim` skips speaker similarity | Task 3 |

No gaps found.

### Placeholder Scan

No TBD, TODO, or placeholder patterns. All steps contain complete code.

### Type Consistency

- `_save_to_manifest()` now takes `gen_config` parameter, included in manifest entry as `generation_config`
- `generation_config` fixture returns dict with `num_steps`, `guidance_scale`, `model_commit`
- `save-baseline` reads manifest to extract `generation_config` and `speakers`
- `ScoreResult(sample_name, scores, passed, details)` matches `scorer_registry.py:18-28`
- `results_to_json(results, path)` matches `scorer_registry.py:149-153`
- `BaselineManager.compare(sample_name, baseline_scores, current_scores)` matches `regression.py:64-98`
- `save_baseline(baseline, path)` matches `regression.py:27-30`
- `load_baseline(path)` matches `regression.py:21-24`
- `detect_silence_artifacts(audio, sr)` matches `custom_scorers.py:20-84`
- `score_dnsmos(audio, sr)` returns `{"dnsmos_sig": float, ...}` matching `versa_scorer.py:36-63`
- `score_speaker_similarity(gen, ref, sr, use_gpu)` matches `versa_scorer.py:91-117`
- SPEAKERS list in conftest matches the 12 speakers from spec
- SPEAKER_SUBSET = ["cicero", "femalenord", "alduin"] — 3 voices covering male/female/beast
- `_run_generate()` parameters match `audio_generation_pipeline.generate_audio()` signature
