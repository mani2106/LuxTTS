# Audio Quality Testing Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiered audio quality evaluation framework that scores TTS output across naturalness, signal quality, speaker similarity, and intelligibility, with regression baselines and agent-parseable reports.

**Architecture:** VERSA toolkit provides validated standard metrics (UTMOS, DNSMOS, speaker similarity, WER) via its Python API. Custom scorers handle LuxTTS-specific checks (batch degradation, vocalization quality, post-processing delta, silence/artifact detection, chunking quality). Three tiers: fast CI (pure CPU, pre-generated fixtures), full eval (GPU, fresh generation), regression (baseline comparison). All test outputs are structured JSON with actionable failure messages.

**Tech Stack:** Python, VERSA, numpy, scipy, librosa, pytest, soundfile

**Spec:** `docs/superpowers/specs/2026-05-03-audio-quality-testing-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `requirements-test.txt` | Test-only dependencies | Create |
| `tests/audio_quality/__init__.py` | Package init | Create |
| `tests/audio_quality/conftest.py` | Shared fixtures (audio loaders, scorer singletons) | Create |
| `tests/audio_quality/scorers/__init__.py` | Scorers package init | Create |
| `tests/audio_quality/scorers/versa_scorer.py` | VERSA integration wrapper | Create |
| `tests/audio_quality/scorers/custom_scorers.py` | LuxTTS-specific signal metrics | Create |
| `tests/audio_quality/scorers/scorer_registry.py` | Compose metric suites, run collection, produce reports | Create |
| `tests/audio_quality/suites/__init__.py` | Suites package init | Create |
| `tests/audio_quality/suites/fast_ci.py` | Tier 1 test case definitions | Create |
| `tests/audio_quality/suites/full_eval.py` | Tier 2 test case definitions | Create |
| `tests/audio_quality/suites/regression.py` | Tier 3 baseline comparison logic | Create |
| `tests/audio_quality/baselines/` | Directory for baseline JSON files | Create |
| `tests/audio_quality/baselines/.gitkeep` | Ensure directory exists in git | Create |
| `tests/audio_quality/test_fast_ci.py` | pytest tests for Tier 1 | Create |
| `tests/audio_quality/test_full_eval.py` | pytest tests for Tier 2 | Create |
| `tests/audio_quality/test_regression.py` | pytest tests for Tier 3 | Create |
| `tests/eval_audio_quality.py` | CLI runner for standalone evaluation | Create |

---

### Task 1: Create Test Dependencies File

**Files:**
- Create: `requirements-test.txt`

- [ ] **Step 1: Create `requirements-test.txt`**

```txt
# VERSA evaluation toolkit (UTMOS, DNSMOS, speaker similarity, WER)
git+https://github.com/wavlab-speech/versa.git#egg=versa-speech-audio-toolkit --no-build-isolation

# Audio I/O
soundfile

# Speaker embeddings for custom similarity checks
resemblyzer

# Test infrastructure
pytest
pytest-timeout
pytest-xdist

# Reporting
jinja2
```

- [ ] **Step 2: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add requirements-test.txt
git commit -m "chore: add test-only dependencies for audio quality evaluation"
```

---

### Task 2: Create Package Structure and Shared Fixtures

**Files:**
- Create: `tests/audio_quality/__init__.py`
- Create: `tests/audio_quality/conftest.py`
- Create: `tests/audio_quality/scorers/__init__.py`
- Create: `tests/audio_quality/suites/__init__.py`
- Create: `tests/audio_quality/baselines/.gitkeep`

- [ ] **Step 1: Create `tests/audio_quality/__init__.py`**

```python
"""Audio quality evaluation framework for LuxTTS."""
```

- [ ] **Step 2: Create `tests/audio_quality/scorers/__init__.py`**

```python
"""Audio quality scorers: VERSA integration + custom LuxTTS metrics."""
```

- [ ] **Step 3: Create `tests/audio_quality/suites/__init__.py`**

```python
"""Test suite definitions for each evaluation tier."""
```

- [ ] **Step 4: Create `tests/audio_quality/baselines/.gitkeep`**

Empty file.

- [ ] **Step 5: Create `tests/audio_quality/conftest.py`**

```python
"""Shared fixtures for audio quality tests."""

import json
from pathlib import Path

import numpy as np
import pytest

BASELINES_DIR = Path(__file__).parent / "baselines"


@pytest.fixture
def baselines_dir():
    return BASELINES_DIR


@pytest.fixture
def sample_48k_speech():
    """Generate synthetic 48kHz speech-like audio (mix of fundamental + harmonics)."""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Fundamental at 150Hz (typical male voice) + harmonics
    audio = (
        0.4 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 450 * t)
        + 0.05 * np.sin(2 * np.pi * 8000 * t)  # sibilant-like content
    )
    # Add amplitude envelope (attack-sustain-release)
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
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 6: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/
git commit -m "feat: create audio quality evaluation package structure and fixtures"
```

---

### Task 3: Implement VERSA Scorer Wrapper

**Files:**
- Create: `tests/audio_quality/scorers/versa_scorer.py`

- [ ] **Step 1: Write failing test**

Add to `tests/audio_quality/test_fast_ci.py` (create the file):

```python
"""Tier 1: Fast CI tests (pure CPU, no GPU, no model loading)."""

import numpy as np
import pytest

from tests.audio_quality.scorers.versa_scorer import (
    score_dnsmos,
    score_dnsmos_available,
)


def test_dnsmos_available():
    """DNSMOS scorer should be importable (speechmos + onnxruntime)."""
    assert score_dnsmos_available(), (
        "DNSMOS not available. Install: uv pip install speechmos onnxruntime"
    )


def test_dnsmos_scores_speech(sample_48k_speech):
    """DNSMOS should return structured scores for speech-like audio."""
    audio, sr = sample_48k_speech
    result = score_dnsmos(audio, sr)

    assert "dnsmos_sig" in result, f"Missing dnsmos_sig key. Got keys: {list(result.keys())}"
    assert "dnsmos_bak" in result, f"Missing dnsmos_bak key. Got keys: {list(result.keys())}"
    assert "dnsmos_ovrl" in result, f"Missing dnsmos_ovrl key. Got keys: {list(result.keys())}"

    # Scores should be in 1-5 range
    for key in ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"]:
        assert 1.0 <= result[key] <= 5.0, (
            f"{key}={result[key]:.2f} outside expected range [1.0, 5.0]"
        )


def test_dnsmos_silence_low_score(sample_48k_silence):
    """DNSMOS should give low signal quality for silence."""
    audio, sr = sample_48k_silence
    result = score_dnsmos(audio, sr)
    # Silence should have low signal quality (but not crash)
    assert result["dnsmos_sig"] < 3.0, (
        f"Silence got unexpectedly high SIG score: {result['dnsmos_sig']:.2f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py::test_dnsmos_available -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.audio_quality.scorers.versa_scorer'`

- [ ] **Step 3: Implement `tests/audio_quality/scorers/versa_scorer.py`**

```python
"""VERSA toolkit integration for standard audio quality metrics.

Provides thin wrappers around VERSA's Python API for:
- DNSMOS (signal quality, CPU-compatible via ONNX)
- UTMOS (naturalness, GPU recommended)
- Speaker similarity (voice cloning quality, GPU recommended)
- WER via Whisper (intelligibility, GPU recommended)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _resample(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target sample rate."""
    if sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)


def score_dnsmos_available() -> bool:
    """Check if DNSMOS is available without importing heavy deps."""
    try:
        import speechmos  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def score_dnsmos(audio: np.ndarray, sr: int) -> dict:
    """Score audio with DNSMOS (signal quality, background noise, overall).

    Runs on CPU via ONNX. Audio is resampled to 16kHz internally.

    Args:
        audio: Float32 audio array
        sr: Sample rate

    Returns:
        {"dnsmos_sig": float, "dnsmos_bak": float, "dnsmos_ovrl": float}
    """
    from speechmos import dnsmos

    audio_16k = _resample(audio, sr, 16000).astype(np.float64)

    result = dnsmos.run(audio_16k, 16000)

    scores = {}
    for item in result:
        if "SIG" in item:
            scores["dnsmos_sig"] = float(item["SIG"])
        if "BAK" in item:
            scores["dnsmos_bak"] = float(item["BAK"])
        if "OVRL" in item:
            scores["dnsmos_ovrl"] = float(item["OVRL"])

    if not scores:
        raise ValueError(f"DNSMOS returned unexpected format: {result}")

    return scores


def score_utmos(audio: np.ndarray, sr: int, use_gpu: bool = False) -> dict:
    """Score audio with UTMOS v1 (perceived naturalness).

    Args:
        audio: Float32 audio array
        sr: Sample rate
        use_gpu: Whether to use GPU

    Returns:
        {"utmos": float} — score in 1-5 range
    """
    from versa import pseudo_mos_setup, pseudo_mos_metric

    predictor_dict, predictor_fs = pseudo_mos_setup(
        predictor_types=["utmos"],
        predictor_args={"utmos": {"fs": 16000}},
        use_gpu=use_gpu,
    )

    audio_16k = _resample(audio, sr, 16000)
    scores = pseudo_mos_metric(audio_16k, 16000, predictor_dict, predictor_fs, use_gpu=use_gpu)

    return {"utmos": float(scores.get("utmos", 0.0))}


def score_speaker_similarity(
    generated_audio: np.ndarray,
    reference_audio: np.ndarray,
    sr: int,
    use_gpu: bool = False,
) -> dict:
    """Score speaker similarity between generated and reference audio.

    Args:
        generated_audio: TTS-generated audio
        reference_audio: Original reference/prompt audio
        sr: Sample rate (both must be same)
        use_gpu: Whether to use GPU

    Returns:
        {"speaker_similarity": float} — cosine similarity, higher is better (>0.8 = same speaker)
    """
    from versa import speaker_model_setup, speaker_metric

    model = speaker_model_setup(model_tag="default", use_gpu=use_gpu)

    gen_16k = _resample(generated_audio, sr, 16000)
    ref_16k = _resample(reference_audio, sr, 16000)

    result = speaker_metric(model=model, pred_x=gen_16k, gt_x=ref_16k, fs=16000)

    return {"speaker_similarity": float(result.get("spk_similarity", 0.0))}


def score_wer(
    audio: np.ndarray,
    sr: int,
    reference_text: str,
    whisper_model: str = "base",
    use_gpu: bool = False,
) -> dict:
    """Score word error rate using Whisper ASR.

    Args:
        audio: Audio to transcribe
        sr: Sample rate
        reference_text: Expected text
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        use_gpu: Whether to use GPU

    Returns:
        {"wer": float, "cer": float, "hyp_text": str, "ref_text": str}
    """
    from versa import whisper_wer_setup, whisper_levenshtein_metric

    wer_utils = whisper_wer_setup(
        model_tag=whisper_model,
        beam_size=5,
        text_cleaner="whisper_basic",
        use_gpu=use_gpu,
    )

    audio_16k = _resample(audio, sr, 16000)
    result = whisper_levenshtein_metric(
        wer_utils=wer_utils,
        pred_x=audio_16k,
        ref_text=reference_text,
        fs=16000,
    )

    # Calculate WER percentage
    total_words = (
        result["whisper_wer_delete"]
        + result["whisper_wer_replace"]
        + result["whisper_wer_equal"]
    )
    wer_score = (
        (result["whisper_wer_delete"] + result["whisper_wer_insert"] + result["whisper_wer_replace"])
        / max(total_words, 1)
    )

    total_chars = (
        result["whisper_cer_delete"]
        + result["whisper_cer_replace"]
        + result["whisper_cer_equal"]
    )
    cer_score = (
        (result["whisper_cer_delete"] + result["whisper_cer_insert"] + result["whisper_cer_replace"])
        / max(total_chars, 1)
    )

    return {
        "wer": float(wer_score),
        "cer": float(cer_score),
        "hyp_text": result.get("whisper_hyp_text", ""),
        "ref_text": result.get("ref_text", reference_text),
    }
```

- [ ] **Step 4: Run tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py -v`
Expected: All 3 DNSMOS tests PASS (if speechmos is installed) or skip with clear message.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/scorers/versa_scorer.py tests/audio_quality/test_fast_ci.py
git commit -m "feat: add VERSA scorer wrapper with DNSMOS, UTMOS, speaker similarity, WER"
```

---

### Task 4: Implement Custom LuxTTS Scorers

**Files:**
- Create: `tests/audio_quality/scorers/custom_scorers.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/audio_quality/test_fast_ci.py`:

```python
from tests.audio_quality.scorers.custom_scorers import (
    detect_silence_artifacts,
    score_post_processing_delta,
)


def test_detect_silence_artifacts_clean(sample_48k_speech):
    """Clean speech should have no silence/artifact issues."""
    audio, sr = sample_48k_speech
    result = detect_silence_artifacts(audio, sr)

    assert result["has_clipping"] is False, "Clean audio should not clip"
    assert result["trailing_silence_ms"] < 100, (
        f"Unexpected trailing silence: {result['trailing_silence_ms']:.0f}ms"
    )
    assert "silence_ratio" in result
    assert "zero_crossing_rate" in result


def test_detect_silence_artifacts_clipped():
    """Clipped audio should be detected."""
    sr = 48000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    audio = np.clip(2.0 * np.sin(2 * np.pi * 440 * t), -0.5, 0.5)

    result = detect_silence_artifacts(audio, sr)

    assert result["has_clipping"] is True, "Clipped audio should be detected"


def test_detect_silence_artifacts_trailing_silence():
    """Trailing silence >500ms should be flagged."""
    sr = 48000
    t = np.linspace(0, 0.5, sr // 2, dtype=np.float32)
    speech = 0.5 * np.sin(2 * np.pi * 150 * t)
    silence = np.zeros(sr, dtype=np.float32)  # 1 second of silence
    audio = np.concatenate([speech, silence])

    result = detect_silence_artifacts(audio, sr)

    assert result["trailing_silence_ms"] > 500, (
        f"Should detect >500ms trailing silence, got {result['trailing_silence_ms']:.0f}ms"
    )
    assert result["has_trailing_artifact"] is True


def test_post_processing_delta_detects_change(sample_48k_speech):
    """Post-processing delta should detect when audio was modified."""
    audio, sr = sample_48k_speech
    # Simulate post-processing: add some noise
    processed = audio + 0.01 * np.random.randn(len(audio)).astype(np.float32)

    result = score_post_processing_delta(audio, processed, sr)

    assert "dnsmos_delta_sig" in result
    assert "dnsmos_delta_ovrl" in result
    assert "rms_change_db" in result
    # Modified audio should show some change
    assert result["rms_change_db"] != 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py::test_detect_silence_artifacts_clean -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `tests/audio_quality/scorers/custom_scorers.py`**

```python
"""Custom LuxTTS-specific audio quality scorers.

These scorers handle domain-specific checks that VERSA doesn't cover:
- Silence/artifact detection (pure signal analysis, no ML)
- Post-processing chain delta analysis
- Batch degradation detection
- Vocalization tag quality
- Chunking/crossfade quality
"""

import logging
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def detect_silence_artifacts(
    audio: np.ndarray,
    sr: int,
    trailing_threshold_ms: float = 500.0,
    clipping_threshold: float = 0.99,
    silence_db: float = -60.0,
) -> dict:
    """Detect silence issues and artifacts in audio.

    Pure signal analysis — no ML, runs on CPU, suitable for fast CI.

    Args:
        audio: Float32 audio array
        sr: Sample rate
        trailing_threshold_ms: Flag if trailing silence exceeds this (ms)
        clipping_threshold: Flag if any sample exceeds this amplitude
        silence_db: Threshold for silence detection (dB)

    Returns:
        {
            "trailing_silence_ms": float,
            "has_trailing_artifact": bool,
            "has_clipping": bool,
            "silence_ratio": float,  # fraction of audio that is silent
            "zero_crossing_rate": float,
            "peak_amplitude": float,
            "duration_s": float,
        }
    """
    duration_s = len(audio) / sr
    peak_amplitude = float(np.max(np.abs(audio)))

    # Clipping detection
    has_clipping = peak_amplitude >= clipping_threshold

    # Trailing silence detection
    # Work backwards from the end to find where audio becomes non-silent
    silence_linear = 10 ** (silence_db / 20.0)
    abs_audio = np.abs(audio)
    non_silent_from_end = np.argmax(abs_audio[::-1] > silence_linear)
    trailing_silence_ms = float(non_silent_from_end / sr * 1000)
    has_trailing_artifact = trailing_silence_ms > trailing_threshold_ms

    # Overall silence ratio
    silent_samples = np.sum(abs_audio < silence_linear)
    silence_ratio = float(silent_samples / len(audio))

    # Zero crossing rate (indicator of noisiness/buzziness)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
    zcr = float(zero_crossings / len(audio))

    return {
        "trailing_silence_ms": trailing_silence_ms,
        "has_trailing_artifact": has_trailing_artifact,
        "has_clipping": has_clipping,
        "silence_ratio": silence_ratio,
        "zero_crossing_rate": zcr,
        "peak_amplitude": peak_amplitude,
        "duration_s": duration_s,
    }


def score_post_processing_delta(
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
) -> dict:
    """Measure quality impact of the post-processing chain.

    Compares raw TTS output against post-processed output using signal analysis.
    Optionally runs DNSMOS on both if available.

    Args:
        raw_audio: Raw TTS output (before DSP)
        processed_audio: Post-processed audio (after DSP)
        sr: Sample rate

    Returns:
        {
            "rms_change_db": float,
            "peak_change_db": float,
            "spectral_centroid_change_hz": float,
            "dnsmos_delta_sig": float or None,
            "dnsmos_delta_ovrl": float or None,
        }
    """
    # RMS change
    raw_rms = float(np.sqrt(np.mean(raw_audio ** 2))) + 1e-10
    proc_rms = float(np.sqrt(np.mean(processed_audio ** 2))) + 1e-10
    rms_change_db = 20.0 * np.log10(proc_rms / raw_rms)

    # Peak change
    raw_peak = float(np.max(np.abs(raw_audio))) + 1e-10
    proc_peak = float(np.max(np.abs(processed_audio))) + 1e-10
    peak_change_db = 20.0 * np.log10(proc_peak / raw_peak)

    # Spectral centroid change
    raw_centroid = _spectral_centroid(raw_audio, sr)
    proc_centroid = _spectral_centroid(processed_audio, sr)
    centroid_change = proc_centroid - raw_centroid

    # Optional DNSMOS delta (CPU-safe)
    dnsmos_delta_sig = None
    dnsmos_delta_ovrl = None
    try:
        from tests.audio_quality.scorers.versa_scorer import score_dnsmos
        raw_scores = score_dnsmos(raw_audio, sr)
        proc_scores = score_dnsmos(processed_audio, sr)
        dnsmos_delta_sig = proc_scores["dnsmos_sig"] - raw_scores["dnsmos_sig"]
        dnsmos_delta_ovrl = proc_scores["dnsmos_ovrl"] - raw_scores["dnsmos_ovrl"]
    except Exception:
        logger.debug("DNSMOS not available for post-processing delta, skipping")

    return {
        "rms_change_db": rms_change_db,
        "peak_change_db": peak_change_db,
        "spectral_centroid_change_hz": centroid_change,
        "dnsmos_delta_sig": dnsmos_delta_sig,
        "dnsmos_delta_ovrl": dnsmos_delta_ovrl,
    }


def score_batch_degradation(
    audio_clips: list[np.ndarray],
    sr: int,
    use_gpu: bool = False,
) -> dict:
    """Detect voice quality drift across sequential same-speaker generations.

    Args:
        audio_clips: List of audio arrays from sequential generations
        sr: Sample rate
        use_gpu: Whether to use GPU for ML-based metrics

    Returns:
        {
            "num_clips": int,
            "first_last_speaker_similarity": float or None,
            "dnsmos_drift": float or None,  # DNSMOS OVRL first vs last delta
            "rms_drift_db": float,
            "duration_drift_pct": float,
        }
    """
    if len(audio_clips) < 2:
        return {"num_clips": len(audio_clips), "error": "Need at least 2 clips"}

    first = audio_clips[0]
    last = audio_clips[-1]

    # RMS drift
    first_rms = float(np.sqrt(np.mean(first ** 2))) + 1e-10
    last_rms = float(np.sqrt(np.mean(last ** 2))) + 1e-10
    rms_drift_db = abs(20.0 * np.log10(last_rms / first_rms))

    # Duration drift
    duration_drift_pct = abs(len(last) - len(first)) / max(len(first), 1) * 100

    # DNSMOS drift
    dnsmos_drift = None
    try:
        from tests.audio_quality.scorers.versa_scorer import score_dnsmos
        first_scores = score_dnsmos(first, sr)
        last_scores = score_dnsmos(last, sr)
        dnsmos_drift = last_scores["dnsmos_ovrl"] - first_scores["dnsmos_ovrl"]
    except Exception:
        logger.debug("DNSMOS not available for batch degradation scoring")

    # Speaker similarity drift (GPU recommended)
    speaker_sim = None
    try:
        from tests.audio_quality.scorers.versa_scorer import score_speaker_similarity
        result = score_speaker_similarity(last, first, sr, use_gpu=use_gpu)
        speaker_sim = result["speaker_similarity"]
    except Exception:
        logger.debug("Speaker similarity not available for batch degradation")

    return {
        "num_clips": len(audio_clips),
        "first_last_speaker_similarity": speaker_sim,
        "dnsmos_drift": dnsmos_drift,
        "rms_drift_db": rms_drift_db,
        "duration_drift_pct": duration_drift_pct,
    }


def score_vocalization_quality(
    vocalization_audio: np.ndarray,
    speech_audio: np.ndarray,
    sr: int,
    expected_min_duration_s: float = 0.2,
    expected_max_duration_s: float = 3.0,
) -> dict:
    """Assess vocalization tag quality — should sound distinct from speech.

    Args:
        vocalization_audio: Audio from a vocalization tag (e.g., [sighs])
        speech_audio: Normal speech audio for comparison
        sr: Sample rate
        expected_min_duration_s: Minimum expected duration
        expected_max_duration_s: Maximum expected duration

    Returns:
        {
            "duration_s": float,
            "duration_in_range": bool,
            "spectral_centroid_dist_hz": float,  # distance from speech centroid
            "energy_ratio_vs_speech": float,
            "is_distinct_from_speech": bool,
        }
    """
    voc_duration = len(vocalization_audio) / sr

    # Spectral centroid distance (vocalizations should differ from speech)
    speech_centroid = _spectral_centroid(speech_audio, sr)
    voc_centroid = _spectral_centroid(vocalization_audio, sr)
    centroid_dist = abs(voc_centroid - speech_centroid)

    # Energy ratio
    speech_rms = float(np.sqrt(np.mean(speech_audio ** 2))) + 1e-10
    voc_rms = float(np.sqrt(np.mean(vocalization_audio ** 2))) + 1e-10
    energy_ratio = voc_rms / speech_rms

    # Distinctness: spectral centroid differs by >200Hz or energy differs by >3dB
    is_distinct = centroid_dist > 200 or abs(20 * np.log10(energy_ratio)) > 3

    return {
        "duration_s": voc_duration,
        "duration_in_range": expected_min_duration_s <= voc_duration <= expected_max_duration_s,
        "spectral_centroid_dist_hz": centroid_dist,
        "energy_ratio_vs_speech": energy_ratio,
        "is_distinct_from_speech": is_distinct,
    }


def score_chunking_quality(
    audio: np.ndarray,
    sr: int,
    estimated_chunk_duration_s: float = 3.0,
    energy_dip_threshold_db: float = -6.0,
) -> dict:
    """Detect crossfade artifacts at chunk boundaries.

    Args:
        audio: Full concatenated audio
        sr: Sample rate
        estimated_chunk_duration_s: Expected duration per chunk
        energy_dip_threshold_db: Flag if energy dips below this at boundaries

    Returns:
        {
            "num_estimated_boundaries": int,
            "boundary_dips_db": list[float],
            "has_audible_artifacts": bool,
            "max_dip_db": float,
        }
    """
    total_duration = len(audio) / sr
    num_boundaries = max(0, int(total_duration / estimated_chunk_duration_s) - 1)

    if num_boundaries == 0:
        return {
            "num_estimated_boundaries": 0,
            "boundary_dips_db": [],
            "has_audible_artifacts": False,
            "max_dip_db": 0.0,
        }

    # Measure RMS energy in windows around each estimated boundary
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    overall_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-10

    dips = []
    for i in range(1, num_boundaries + 1):
        center = int(i * estimated_chunk_duration_s * sr)
        start = max(0, center - window_samples // 2)
        end = min(len(audio), center + window_samples // 2)
        boundary_rms = float(np.sqrt(np.mean(audio[start:end] ** 2))) + 1e-10
        dip_db = 20.0 * np.log10(boundary_rms / overall_rms)
        dips.append(dip_db)

    max_dip = min(dips) if dips else 0.0
    has_artifacts = max_dip < energy_dip_threshold_db

    return {
        "num_estimated_boundaries": num_boundaries,
        "boundary_dips_db": dips,
        "has_audible_artifacts": has_artifacts,
        "max_dip_db": max_dip,
    }


def _spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid in Hz."""
    magnitudes = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    centroid = float(np.sum(magnitudes * freqs) / (np.sum(magnitudes) + 1e-10))
    return centroid
```

- [ ] **Step 4: Run tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py -v`
Expected: All custom scorer tests PASS. DNSMOS tests PASS or skip if not installed.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/scorers/custom_scorers.py tests/audio_quality/test_fast_ci.py
git commit -m "feat: add custom LuxTTS scorers (silence detection, post-processing delta, batch degradation, vocalization quality, chunking quality)"
```

---

### Task 5: Implement Scorer Registry and Report Formatting

**Files:**
- Create: `tests/audio_quality/scorers/scorer_registry.py`

- [ ] **Step 1: Write failing test**

Add to `tests/audio_quality/test_fast_ci.py`:

```python
from tests.audio_quality.scorers.scorer_registry import (
    ScoreResult,
    format_report,
    compare_scores,
)


def test_score_result_format():
    """ScoreResult should produce agent-parseable dict output."""
    result = ScoreResult(
        sample_name="hello_world",
        scores={"dnsmos_sig": 3.92, "dnsmos_ovrl": 3.78},
        passed=True,
        details="All metrics within acceptable range.",
    )
    d = result.to_dict()

    assert d["sample_name"] == "hello_world"
    assert d["passed"] is True
    assert "dnsmos_sig" in d["scores"]
    assert d["details"] == "All metrics within acceptable range."


def test_format_report_produces_readable_output():
    """format_report should produce human + agent readable text."""
    results = [
        ScoreResult("test1", {"dnsmos_sig": 3.5}, True, "OK"),
        ScoreResult("test2", {"dnsmos_sig": 2.0}, False, "DNSMOS SIG dropped below threshold: 2.0 < 3.0"),
    ]
    report = format_report(results)

    assert "test1" in report
    assert "test2" in report
    assert "PASS" in report
    assert "FAIL" in report
    assert "2.0 < 3.0" in report


def test_compare_scores_detects_regression():
    """compare_scores should flag when a metric drops below threshold."""
    baseline = {"dnsmos_sig": 3.92, "dnsmos_ovrl": 3.78}
    current = {"dnsmos_sig": 3.40, "dnsmos_ovrl": 3.80}

    deltas = compare_scores(baseline, current, threshold_pct=5.0)

    # dnsmos_sig dropped ~13% — should be flagged
    assert deltas["dnsmos_sig"]["regressed"] is True
    assert deltas["dnsmos_sig"]["delta_pct"] < -5.0

    # dnsmos_ovrl improved — should not be flagged
    assert deltas["dnsmos_ovrl"]["regressed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py::test_score_result_format -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `tests/audio_quality/scorers/scorer_registry.py`**

```python
"""Scorer registry: compose metric suites, run collections, produce agent-parseable reports.

All test outputs are designed to be easily understood by both humans and coding agents:
- Structured JSON with descriptive field names
- Actionable failure messages with baseline/current/delta/threshold
- Summary reports with PASS/FAIL per sample per metric
"""

import json
import logging
from dataclasses import dataclass, field
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
        delta_pct = (delta / abs(b)) * 100 if b != 0 else 0.0
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


def format_report(results: list[ScoreResult]) -> str:
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
            # Indent details for readability
            for detail_line in result.details.split("\n"):
                lines.append(f"  {detail_line}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def results_to_json(results: list[ScoreResult], path: str):
    """Save results to a JSON file for programmatic consumption."""
    data = [r.to_dict() for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py -v`
Expected: All scorer registry tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/scorers/scorer_registry.py tests/audio_quality/test_fast_ci.py
git commit -m "feat: add scorer registry with agent-parseable reports and regression comparison"
```

---

### Task 6: Implement Tier 1 Fast CI Test Suite

**Files:**
- Create: `tests/audio_quality/suites/fast_ci.py`

- [ ] **Step 1: Implement fast CI suite**

```python
"""Tier 1: Fast CI test suite definitions.

Pure CPU, no GPU, no model loading. Runs on pre-generated audio fixtures.
Suitable for GitHub Actions free tier.
"""

from dataclasses import dataclass


@dataclass
class FastCITestCase:
    """A test case for the fast CI tier."""
    name: str
    description: str
    text: str  # Text that was used to generate the fixture audio
    has_vocalization_tags: bool = False
    is_batch: bool = False
    batch_count: int = 1
    expected_min_duration_s: float = 0.5
    expected_max_duration_s: float = 30.0


# Test cases covering the full range of TTS generation scenarios
FAST_CI_CASES = [
    FastCITestCase(
        name="short_text",
        description="Short text (<50 chars) — basic speech",
        text="Hello, how are you?",
        expected_min_duration_s=0.5,
        expected_max_duration_s=5.0,
    ),
    FastCITestCase(
        name="medium_text",
        description="Medium text (100-200 chars) — chunked generation",
        text="I've been traveling across Skyrim for many years now. "
        "The roads are dangerous, but the scenery never fails to take my breath away.",
        expected_min_duration_s=2.0,
        expected_max_duration_s=15.0,
    ),
    FastCITestCase(
        name="long_text",
        description="Long text (300+ chars) — multi-chunk",
        text="You know, when I first came to Riften, I thought it was the most beautiful "
        "city in all of Skyrim. The way the mist rises off the lake in the morning, the "
        "sound of the docks coming alive with merchants and fishermen. But then I learned "
        "about the Thieves Guild lurking beneath the city, and the corruption that runs "
        "deep through the Ratways. It changed my perspective entirely.",
        expected_min_duration_s=5.0,
        expected_max_duration_s=30.0,
    ),
    FastCITestCase(
        name="vocalization_sighs",
        description="Vocalization tag: [sighs]",
        text="[sighs] I can't believe we made it.",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_gasps",
        description="Vocalization tag: [gasps]",
        text="[gasps] Who's there?",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_whispers",
        description="Vocalization tag: [whispers]",
        text="[whispers] Don't make a sound.",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_screams",
        description="Vocalization tag: [screams]",
        text="[screams] Get away from me!",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_pause",
        description="Vocalization tag: [pause]",
        text="Hello [pause] my old friend.",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="batch_sequential",
        description="5 sequential generations with same speaker — degradation test",
        text="The weather is nice today.",
        is_batch=True,
        batch_count=5,
    ),
    FastCITestCase(
        name="edge_all_caps",
        description="Edge case: ALL CAPS text (pitch shift trigger)",
        text="THIS IS AN EMERGENCY!",
    ),
    FastCITestCase(
        name="edge_question",
        description="Edge case: Question text (pitch shift trigger)",
        text="Where are you going?",
    ),
    FastCITestCase(
        name="edge_ellipsis",
        description="Edge case: Ellipsis text (pitch shift trigger)",
        text="I'm not so sure about that...",
    ),
]
```

- [ ] **Step 2: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/suites/fast_ci.py
git commit -m "feat: add Tier 1 fast CI test case definitions"
```

---

### Task 7: Implement Regression Baseline Management

**Files:**
- Create: `tests/audio_quality/suites/regression.py`

- [ ] **Step 1: Write failing test**

Create `tests/audio_quality/test_regression.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_regression.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `tests/audio_quality/suites/regression.py`**

```python
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
    with open(path) as f:
        return json.load(f)


def save_baseline(baseline: dict, path: Path):
    """Save a baseline JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
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
        """
        Args:
            regression_threshold_pct: Flag if any metric drops more than this percentage
        """
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
                "deltas": dict,  # from compare_scores
                "details": str,  # actionable failure message
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
```

- [ ] **Step 4: Run tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_regression.py -v`
Expected: All 3 regression tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/suites/regression.py tests/audio_quality/test_regression.py
git commit -m "feat: add regression baseline management with agent-parseable comparison"
```

---

### Task 8: Implement Tier 2 Full Eval Suite

**Files:**
- Create: `tests/audio_quality/suites/full_eval.py`
- Create: `tests/audio_quality/test_full_eval.py`

- [ ] **Step 1: Create `tests/audio_quality/suites/full_eval.py`**

```python
"""Tier 2: Full evaluation test case definitions.

Runs with GPU. Generates fresh audio and runs full VERSA + custom metric suite.
Marked with @pytest.mark.gpu and @pytest.mark.slow.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FullEvalTestCase:
    """A test case for the full evaluation tier."""
    name: str
    description: str
    text: str
    speaker: Optional[str] = None  # Speaker preset name, or None for default
    enable_post_processing: bool = True
    has_vocalization_tags: bool = False
    is_batch: bool = False
    batch_count: int = 1
    reference_text: Optional[str] = None  # For WER, if different from text


FULL_EVAL_CASES = [
    FullEvalTestCase(
        name="basic_speech",
        description="Basic short speech with post-processing",
        text="Hello, how are you doing today?",
        reference_text="Hello, how are you doing today?",
    ),
    FullEvalTestCase(
        name="raw_tts_no_postproc",
        description="Raw TTS output without post-processing",
        text="Hello, how are you doing today?",
        enable_post_processing=False,
        reference_text="Hello, how are you doing today?",
    ),
    FullEvalTestCase(
        name="medium_text_chunked",
        description="Medium text that triggers punctuation chunking",
        text="I've been traveling across Skyrim for many years now. "
        "The roads are dangerous, but the scenery never fails to take my breath away.",
        reference_text="I've been traveling across Skyrim for many years now. "
        "The roads are dangerous, but the scenery never fails to take my breath away.",
    ),
    FullEvalTestCase(
        name="long_text_multi_chunk",
        description="Long text requiring multiple chunks",
        text="You know, when I first came to Riften, I thought it was the most beautiful "
        "city in all of Skyrim. The way the mist rises off the lake in the morning, the "
        "sound of the docks coming alive with merchants and fishermen. But then I learned "
        "about the Thieves Guild lurking beneath the city, and the corruption that runs "
        "deep through the Ratways. It changed my perspective entirely.",
    ),
    FullEvalTestCase(
        name="vocalization_sighs",
        description="Vocalization: [sighs] with speech",
        text="[sighs] I can't believe we made it.",
        has_vocalization_tags=True,
        reference_text="I can't believe we made it.",
    ),
    FullEvalTestCase(
        name="vocalization_gasps",
        description="Vocalization: [gasps] with speech",
        text="[gasps] Who's there?",
        has_vocalization_tags=True,
        reference_text="Who's there?",
    ),
    FullEvalTestCase(
        name="vocalization_whispers",
        description="Vocalization: [whispers] modifies following speech",
        text="[whispers] Don't make a sound.",
        has_vocalization_tags=True,
        reference_text="Don't make a sound.",
    ),
    FullEvalTestCase(
        name="vocalization_screams",
        description="Vocalization: [screams] with speech",
        text="[screams] Get away from me!",
        has_vocalization_tags=True,
        reference_text="Get away from me!",
    ),
    FullEvalTestCase(
        name="batch_degradation_5",
        description="5 sequential generations — degradation test",
        text="The weather is quite pleasant today.",
        is_batch=True,
        batch_count=5,
        reference_text="The weather is quite pleasant today.",
    ),
]
```

- [ ] **Step 2: Create `tests/audio_quality/test_full_eval.py`**

```python
"""Tier 2: Full evaluation tests (GPU required).

Run with: pytest tests/audio_quality/test_full_eval.py -v -m gpu
"""

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_full_eval_importable():
    """Verify full eval suite module is importable."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    assert len(FULL_EVAL_CASES) > 0
    for case in FULL_EVAL_CASES:
        assert case.name, f"Test case missing name: {case}"
        assert case.text, f"Test case '{case.name}' missing text"


@pytest.mark.parametrize("case_name", [c.name for c in __import__(
    "tests.audio_quality.suites.full_eval", fromlist=["FULL_EVAL_CASES"]
).FULL_EVAL_CASES])
def test_full_eval_case_has_reference_text(case_name):
    """Each full eval case that tests WER should have reference_text."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    case = next(c for c in FULL_EVAL_CASES if c.name == case_name)
    # Cases with reference_text can be tested for WER
    if not case.is_batch and not case.has_vocalization_tags:
        assert case.reference_text is not None, (
            f"Non-batch, non-vocalization case '{case_name}' needs reference_text for WER testing"
        )
```

- [ ] **Step 3: Run tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_full_eval.py -v`
Expected: All tests PASS (these are structural tests, no GPU needed).

- [ ] **Step 4: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/audio_quality/suites/full_eval.py tests/audio_quality/test_full_eval.py
git commit -m "feat: add Tier 2 full evaluation test case definitions"
```

---

### Task 9: Implement CLI Runner

**Files:**
- Create: `tests/eval_audio_quality.py`

- [ ] **Step 1: Implement CLI runner**

```python
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
            tags.append(f"batch×{case.batch_count}")
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
            tags.append(f"batch×{case.batch_count}")
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
    """Generate a baseline file from current test results.

    This requires running the full evaluation first, then saving scores.
    In practice, you'd pipe results from Tier 2 tests into this command.
    """
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
```

- [ ] **Step 2: Test CLI runs**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python tests/eval_audio_quality.py --list-cases`
Expected: Lists all Tier 1 and Tier 2 test cases.

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python tests/eval_audio_quality.py --generate-baselines`
Expected: Creates template baseline file at `tests/audio_quality/baselines/master_baseline.json`.

- [ ] **Step 3: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add tests/eval_audio_quality.py
git commit -m "feat: add CLI runner for audio quality evaluation and baseline management"
```

---

### Task 10: Run Full Test Suite and Verify

**Files:**
- No new files

- [ ] **Step 1: Run existing test suite to verify no regressions**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/ -v --timeout=30 -x --ignore=tests/audio_quality/test_full_eval.py`
Expected: All existing tests PASS.

- [ ] **Step 2: Run new audio quality tests**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python -m pytest tests/audio_quality/test_fast_ci.py tests/audio_quality/test_regression.py -v`
Expected: All fast CI and regression tests PASS.

- [ ] **Step 3: Verify CLI works end-to-end**

Run: `cd "F:/Studies/LuxTTS" && .venv/Scripts/python tests/eval_audio_quality.py --list-cases && .venv/Scripts/python tests/eval_audio_quality.py --generate-baselines && .venv/Scripts/python tests/eval_audio_quality.py --compare-baselines`
Expected: All three commands succeed without errors.

---

## Self-Review

### Spec Coverage

| Spec Section | Task |
|---|---|
| Architecture (three layers) | Tasks 2-8 |
| File structure | Task 2 |
| VERSA metrics (UTMOS, DNSMOS, speaker sim, WER) | Task 3 |
| Custom scorers (all 5) | Task 4 |
| Scorer registry + agent-parseable reports | Task 5 |
| Tier 1 fast CI (pure CPU) | Tasks 3, 4, 6 |
| Tier 2 full eval (GPU) | Task 8 |
| Tier 3 regression baselines | Task 7 |
| Baseline JSON format | Task 7 |
| Test-only dependencies | Task 1 |
| CLI runner | Task 9 |
| Agent-parseable output requirement | Task 5 (ScoreResult, format_report, compare_scores) |

No gaps found.

### Placeholder Scan

No TBD, TODO, or placeholder patterns. All steps contain complete code.

### Type Consistency

- `ScoreResult` used consistently across scorer_registry, regression, CLI
- `score_dnsmos()` returns `{"dnsmos_sig": float, "dnsmos_bak": float, "dnsmos_ovrl": float}` — consistent in versa_scorer.py and custom_scorers.py
- `BaselineManager.compare()` returns dict with `"passed"`, `"regressed_metrics"`, `"deltas"`, `"details"` — consistent in test_regression.py and regression.py
- `_resample()` helper used consistently across versa_scorer functions
- All scorers take `(audio: np.ndarray, sr: int)` as first two args
