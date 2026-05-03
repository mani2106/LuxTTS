"""Tier 2: Full evaluation tests (GPU required).

Generates fresh audio with LuxTTS, scores it with VERSA + custom metrics,
and asserts quality thresholds. Uses speaker samples from speakers/en/.

Run with:
    pytest tests/audio_quality/test_full_eval.py -v -m gpu
    pytest tests/audio_quality/test_full_eval.py -v -m gpu -k "basic_speech"
"""

import asyncio
from pathlib import Path

import librosa
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SPEAKERS_DIR = Path("speakers") / "en"
SAMPLE_RATE = 48000

# Quality thresholds — these represent minimum acceptable quality.
# Adjust if the model legitimately improves/drifts.
MIN_DNSMOS_SIG = 2.5
MIN_DNSMOS_OVRL = 2.5
MIN_DNSMOS_BAK = 2.0
MAX_SILENCE_RATIO = 0.5
MAX_TRAILING_SILENCE_MS = 800
MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0


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


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async generate_audio calls."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _run_generate(text, speaker_audio_path, enable_post_processing=True, save_raw=False):
    """Synchronous wrapper around async generate_audio."""
    from utilities.audio_generation_pipeline import generate_audio
    from utilities.app_config import AppConfig

    config = AppConfig()

    coro = generate_audio(
        text=text,
        speaker_audio=speaker_audio_path,
        config=config,
        enable_post_processing=enable_post_processing,
        seed=42,
        randomize_seed=False,
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
    """Generate audio and verify quality metrics are above thresholds."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    case = next(c for c in FULL_EVAL_CASES if c.name == case_name)

    result = _run_generate(
        case.text, speaker_audio_path,
        enable_post_processing=case.enable_post_processing,
    )
    output_path = result[0]
    audio, sr = _load_generated_audio(output_path)

    # --- Duration checks ---
    duration_s = len(audio) / sr
    assert duration_s >= MIN_DURATION_S, (
        f"[{case_name}] Audio too short: {duration_s:.2f}s < {MIN_DURATION_S}s"
    )
    assert duration_s <= MAX_DURATION_S, (
        f"[{case_name}] Audio too long: {duration_s:.2f}s > {MAX_DURATION_S}s"
    )

    # --- Silence/artifact checks ---
    from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts
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

    # --- DNSMOS quality ---
    from tests.audio_quality.scorers.versa_scorer import score_dnsmos
    scores = score_dnsmos(audio, sr)

    assert scores["dnsmos_sig"] >= MIN_DNSMOS_SIG, (
        f"[{case_name}] DNSMOS SIG too low: {scores['dnsmos_sig']:.2f} < {MIN_DNSMOS_SIG}"
    )
    assert scores["dnsmos_ovrl"] >= MIN_DNSMOS_OVRL, (
        f"[{case_name}] DNSMOS OVRL too low: {scores['dnsmos_ovrl']:.2f} < {MIN_DNSMOS_OVRL}"
    )
    assert scores["dnsmos_bak"] >= MIN_DNSMOS_BAK, (
        f"[{case_name}] DNSMOS BAK too low: {scores['dnsmos_bak']:.2f} < {MIN_DNSMOS_BAK}"
    )


def test_post_processing_improves_quality(speaker_audio_path):
    """Post-processed audio should not degrade DNSMOS scores."""
    from tests.audio_quality.scorers.custom_scorers import score_post_processing_delta

    text = "Hello, how are you doing today?"

    # Generate both raw and processed from same seed
    raw_result = _run_generate(text, speaker_audio_path, enable_post_processing=False, save_raw=False)
    proc_result = _run_generate(text, speaker_audio_path, enable_post_processing=True, save_raw=False)

    raw_audio, sr = _load_generated_audio(raw_result[0])
    proc_audio, _ = _load_generated_audio(proc_result[0])

    delta = score_post_processing_delta(raw_audio, proc_audio, sr)

    # Post-processing should not drastically change RMS (within +/-6dB)
    assert abs(delta["rms_change_db"]) < 6.0, (
        f"Post-processing changed RMS by {delta['rms_change_db']:+.1f}dB (limit: +/-6dB)"
    )

    # If DNSMOS ran, post-processing should not hurt signal quality by more than 0.5
    if delta["dnsmos_delta_sig"] is not None:
        assert delta["dnsmos_delta_sig"] > -0.5, (
            f"Post-processing hurt DNSMOS SIG by {delta['dnsmos_delta_sig']:+.2f}"
        )


def test_speaker_similarity(speaker_audio_path, speaker_audio_array):
    """Generated audio should have reasonable similarity to the reference speaker."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    case = next(c for c in FULL_EVAL_CASES if c.name == "basic_speech")

    result = _run_generate(case.text, speaker_audio_path)
    generated_audio, sr = _load_generated_audio(result[0])

    from tests.audio_quality.scorers.versa_scorer import score_speaker_similarity
    sim = score_speaker_similarity(generated_audio, speaker_audio_array[0], sr, use_gpu=True)

    assert sim["speaker_similarity"] > 0.3, (
        f"Speaker similarity too low: {sim['speaker_similarity']:.3f} — voice cloning may have failed"
    )


def test_intelligibility_wer(speaker_audio_path):
    """Generated speech should be mostly intelligible (low WER)."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    case = next(c for c in FULL_EVAL_CASES if c.name == "raw_tts_no_postproc")

    result = _run_generate(
        case.text, speaker_audio_path,
        enable_post_processing=False,
    )
    audio, sr = _load_generated_audio(result[0])

    from tests.audio_quality.scorers.versa_scorer import score_wer
    wer_result = score_wer(audio, sr, case.reference_text, use_gpu=True)

    assert wer_result["wer"] < 0.5, (
        f"WER too high: {wer_result['wer']:.1%} — "
        f"expected: '{wer_result['ref_text']}', "
        f"got: '{wer_result['hyp_text']}'"
    )


@pytest.mark.parametrize("case_name", [
    "vocalization_sighs",
    "vocalization_gasps",
    "vocalization_whispers",
    "vocalization_screams",
])
def test_vocalization_generation(case_name, speaker_audio_path):
    """Vocalization tags should produce non-empty audio without errors."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    case = next(c for c in FULL_EVAL_CASES if c.name == case_name)

    result = _run_generate(case.text, speaker_audio_path)
    output_path = result[0]
    audio, sr = _load_generated_audio(output_path)

    duration_s = len(audio) / sr
    assert duration_s >= 0.3, (
        f"[{case_name}] Vocalization audio too short: {duration_s:.2f}s"
    )

    # Should not be silence
    rms = float(np.sqrt(np.mean(audio ** 2)))
    assert rms > 0.001, (
        f"[{case_name}] Vocalization audio is near-silent: RMS={rms:.6f}"
    )

    # Basic quality check
    from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts
    artifacts = detect_silence_artifacts(audio, sr)
    assert artifacts["has_clipping"] is False, f"[{case_name}] Clipping in vocalization output"


def test_batch_no_degradation(speaker_audio_path):
    """Sequential generations from same speaker should not degrade significantly."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    case = next(c for c in FULL_EVAL_CASES if c.name == "batch_degradation_5")

    clips = []
    for i in range(case.batch_count):
        result = _run_generate(case.text, speaker_audio_path, seed=42 + i, randomize_seed=False)
        audio, sr = _load_generated_audio(result[0])
        clips.append(audio)

    from tests.audio_quality.scorers.custom_scorers import score_batch_degradation
    degradation = score_batch_degradation(clips, sr)

    # RMS drift between first and last should be small (<3dB)
    assert degradation["rms_drift_db"] < 3.0, (
        f"RMS drift across {case.batch_count} generations: {degradation['rms_drift_db']:.1f}dB"
    )

    # Duration should be consistent (<30% drift)
    assert degradation["duration_drift_pct"] < 30.0, (
        f"Duration drift across {case.batch_count} generations: {degradation['duration_drift_pct']:.1f}%"
    )
