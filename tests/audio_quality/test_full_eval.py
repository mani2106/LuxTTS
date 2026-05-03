"""Tier 2: Full evaluation tests (GPU required).

Generates fresh audio with LuxTTS, scores it with VERSA + custom metrics.
Uses speaker samples from speakers/en/ as voice cloning references.

Two categories of tests:
- GATE tests: Binary pass/fail on hard quality constraints (clipping, silence, duration).
- MEASUREMENT tests: Record scores to JSON for regression tracking. These FAIL if
  scores drop below minimum floors, but their primary purpose is to capture actual
  values for baseline comparison.

Run with:
    pytest tests/audio_quality/test_full_eval.py -v -m gpu
    pytest tests/audio_quality/test_full_eval.py -v -m gpu -k "basic_speech"
"""

import asyncio
import json
from pathlib import Path

import librosa
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SPEAKERS_DIR = Path("speakers") / "en"
SAMPLE_RATE = 48000
SCORES_OUTPUT = Path("tests/audio_quality/baselines/latest_scores.json")

# Hard floors — anything below these is a clear quality problem.
# These are NOT aspirational targets. They represent "clearly broken" levels.
MIN_DNSMOS_SIG = 2.0
MIN_DNSMOS_OVRL = 2.0
MIN_DNSMOS_BAK = 2.0
MAX_SILENCE_RATIO = 0.5
MAX_TRAILING_SILENCE_MS = 800
MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0
MIN_SPEAKER_SIMILARITY = 0.2
MIN_WER_PASS = 0.5


# --- Score collection ---

_collected_scores = {}


def _record_score(test_name: str, scores: dict):
    """Record scores for JSON output at session end."""
    _collected_scores[test_name] = scores


def pytest_sessionfinish(session, exitstatus):
    """Write collected scores to JSON file after all tests complete."""
    if _collected_scores:
        SCORES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(SCORES_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(_collected_scores, f, indent=2)
        print(f"\nScores saved to {SCORES_OUTPUT}")


# --- Fixtures ---


@pytest.fixture(scope="session")
def speaker_audio_path():
    """Path to a speaker reference sample for voice cloning."""
    speaker = SPEAKERS_DIR / "cicero.wav"
    if not speaker.exists():
        pytest.skip(f"Speaker file not found: {speaker}")
    return str(speaker)


@pytest.fixture(scope="session")
def speaker_audio_array(speaker_audio_path):
    """Load speaker reference audio as numpy array at 48kHz."""
    audio, sr = librosa.load(speaker_audio_path, sr=SAMPLE_RATE)
    return audio, sr


def _run_generate(text, speaker_audio_path, enable_post_processing=True, save_raw=False, seed=42, randomize_seed=False):
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
        randomize_seed=randomize_seed,
        save_raw=save_raw,
        return_diagnostics=False,
    )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(coro)
    finally:
        loop.close()

    return result


def _load_generated_audio(output_path):
    """Load a generated WAV file as numpy array."""
    audio, sr = librosa.load(output_path, sr=SAMPLE_RATE)
    return audio.astype(np.float32), sr


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


# --- GPU generation + scoring tests ---


@pytest.mark.parametrize("case_name", [
    "basic_speech",
    "raw_tts_no_postproc",
])
def test_generate_and_score_quality(case_name, speaker_audio_path):
    """Generate audio, run gate checks, and record DNSMOS scores.

    Gate checks (must pass): no clipping, reasonable silence, valid duration.
    Measurement (recorded): DNSMOS SIG/OVRL/BAK scores saved to JSON.
    """
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES
    from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts
    from tests.audio_quality.scorers.versa_scorer import score_dnsmos

    case = next(c for c in FULL_EVAL_CASES if c.name == case_name)

    result = _run_generate(
        case.text, speaker_audio_path,
        enable_post_processing=case.enable_post_processing,
    )
    output_path = result[0]
    audio, sr = _load_generated_audio(output_path)
    duration_s = len(audio) / sr

    # --- Gate checks (hard constraints) ---
    assert MIN_DURATION_S <= duration_s <= MAX_DURATION_S, (
        f"[{case_name}] Duration out of range: {duration_s:.2f}s"
    )

    artifacts = detect_silence_artifacts(audio, sr)
    assert artifacts["has_clipping"] is False, (
        f"[{case_name}] Clipping detected! peak={artifacts['peak_amplitude']:.3f}"
    )
    assert artifacts["trailing_silence_ms"] < MAX_TRAILING_SILENCE_MS, (
        f"[{case_name}] Excessive trailing silence: {artifacts['trailing_silence_ms']:.0f}ms"
    )
    assert artifacts["silence_ratio"] < MAX_SILENCE_RATIO, (
        f"[{case_name}] Too much silence: {artifacts['silence_ratio']:.1%}"
    )

    # --- Measurement (record and floor-check) ---
    scores = score_dnsmos(audio, sr)

    assert scores["dnsmos_sig"] >= MIN_DNSMOS_SIG, (
        f"[{case_name}] DNSMOS SIG below hard floor: {scores['dnsmos_sig']:.2f} < {MIN_DNSMOS_SIG}"
    )
    assert scores["dnsmos_ovrl"] >= MIN_DNSMOS_OVRL, (
        f"[{case_name}] DNSMOS OVRL below hard floor: {scores['dnsmos_ovrl']:.2f} < {MIN_DNSMOS_OVRL} — "
        f"post-processing may be degrading quality"
    )
    assert scores["dnsmos_bak"] >= MIN_DNSMOS_BAK, (
        f"[{case_name}] DNSMOS BAK below hard floor: {scores['dnsmos_bak']:.2f} < {MIN_DNSMOS_BAK}"
    )

    # Record all scores for regression tracking
    _record_score(case_name, {
        "text": case.text,
        "enable_post_processing": case.enable_post_processing,
        "duration_s": round(duration_s, 2),
        **scores,
        **artifacts,
    })


def test_post_processing_impact(speaker_audio_path):
    """Measure exact quality impact of the post-processing chain.

    This test GENERATES audio both ways and records the delta.
    It fails if post-processing degrades signal quality by more than 0.5 DNSMOS points.
    """
    from tests.audio_quality.scorers.custom_scorers import score_post_processing_delta

    text = "Hello, how are you doing today?"

    raw_result = _run_generate(text, speaker_audio_path, enable_post_processing=False)
    proc_result = _run_generate(text, speaker_audio_path, enable_post_processing=True)

    raw_audio, sr = _load_generated_audio(raw_result[0])
    proc_audio, _ = _load_generated_audio(proc_result[0])

    delta = score_post_processing_delta(raw_audio, proc_audio, sr)

    # Gate: post-processing should not drastically change volume
    assert abs(delta["rms_change_db"]) < 6.0, (
        f"Post-processing changed RMS by {delta['rms_change_db']:+.1f}dB"
    )

    # Gate: post-processing should not hurt signal quality by more than 0.5 DNSMOS points
    if delta["dnsmos_delta_sig"] is not None:
        assert delta["dnsmos_delta_sig"] > -0.5, (
            f"Post-processing hurt DNSMOS SIG by {delta['dnsmos_delta_sig']:+.2f} — "
            f"investigate DSP chain"
        )

    # Record delta for tracking
    _record_score("post_processing_delta", {
        "text": text,
        **delta,
    })


def test_speaker_similarity(speaker_audio_path, speaker_audio_array):
    """Score voice cloning accuracy — similarity between reference and generated voice.

    This is a MEASUREMENT test. The score is recorded for regression tracking.
    It fails only if similarity drops below the hard floor (0.2 = clearly different voice).
    """
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES
    from tests.audio_quality.scorers.versa_scorer import score_speaker_similarity

    case = next(c for c in FULL_EVAL_CASES if c.name == "basic_speech")

    result = _run_generate(case.text, speaker_audio_path)
    generated_audio, sr = _load_generated_audio(result[0])

    sim = score_speaker_similarity(generated_audio, speaker_audio_array[0], sr, use_gpu=True)

    assert sim["speaker_similarity"] >= MIN_SPEAKER_SIMILARITY, (
        f"Speaker similarity below hard floor: {sim['speaker_similarity']:.3f} < {MIN_SPEAKER_SIMILARITY} — "
        f"voice cloning may have failed entirely"
    )

    _record_score("speaker_similarity", {
        "text": case.text,
        "speaker": "cicero",
        **sim,
    })


def test_intelligibility_wer(speaker_audio_path):
    """Score word error rate — generated speech should be intelligible."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES
    from tests.audio_quality.scorers.versa_scorer import score_wer

    case = next(c for c in FULL_EVAL_CASES if c.name == "raw_tts_no_postproc")

    result = _run_generate(
        case.text, speaker_audio_path,
        enable_post_processing=False,
    )
    audio, sr = _load_generated_audio(result[0])

    wer_result = score_wer(audio, sr, case.reference_text, use_gpu=True)

    assert wer_result["wer"] < MIN_WER_PASS, (
        f"WER too high: {wer_result['wer']:.1%} — "
        f"expected: '{wer_result['ref_text']}', "
        f"got: '{wer_result['hyp_text']}'"
    )

    _record_score("intelligibility_wer", {
        "text": case.text,
        "reference_text": case.reference_text,
        "wer": wer_result["wer"],
        "cer": wer_result["cer"],
        "hyp_text": wer_result["hyp_text"],
    })


@pytest.mark.parametrize("case_name", [
    "vocalization_sighs",
    "vocalization_gasps",
    "vocalization_whispers",
    "vocalization_screams",
])
def test_vocalization_generation(case_name, speaker_audio_path):
    """Vocalization tags should produce non-silent, non-clipped audio."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES
    from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts

    case = next(c for c in FULL_EVAL_CASES if c.name == case_name)

    result = _run_generate(case.text, speaker_audio_path)
    audio, sr = _load_generated_audio(result[0])
    duration_s = len(audio) / sr

    assert duration_s >= 0.3, f"[{case_name}] Vocalization audio too short: {duration_s:.2f}s"

    rms = float(np.sqrt(np.mean(audio ** 2)))
    assert rms > 0.001, f"[{case_name}] Vocalization audio is near-silent: RMS={rms:.6f}"

    artifacts = detect_silence_artifacts(audio, sr)
    assert artifacts["has_clipping"] is False, f"[{case_name}] Clipping in vocalization output"

    _record_score(case_name, {
        "text": case.text,
        "duration_s": round(duration_s, 2),
        "rms": round(rms, 4),
        **artifacts,
    })


def test_batch_no_degradation(speaker_audio_path):
    """Sequential generations from same speaker should not degrade significantly."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES
    from tests.audio_quality.scorers.custom_scorers import score_batch_degradation

    case = next(c for c in FULL_EVAL_CASES if c.name == "batch_degradation_5")

    clips = []
    for i in range(case.batch_count):
        result = _run_generate(case.text, speaker_audio_path, seed=42 + i, randomize_seed=False)
        audio, sr = _load_generated_audio(result[0])
        clips.append(audio)

    degradation = score_batch_degradation(clips, sr)

    assert degradation["rms_drift_db"] < 3.0, (
        f"RMS drift across {case.batch_count} generations: {degradation['rms_drift_db']:.1f}dB"
    )
    assert degradation["duration_drift_pct"] < 30.0, (
        f"Duration drift across {case.batch_count} generations: {degradation['duration_drift_pct']:.1f}%"
    )

    _record_score("batch_degradation", {
        "text": case.text,
        "num_clips": case.batch_count,
        **degradation,
    })
