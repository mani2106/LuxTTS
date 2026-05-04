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

import librosa
import numpy as np
import pytest

from tests.audio_quality.conftest import SPEAKERS

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# Subset of speakers for tests that don't need the full matrix.
# Most tests use ALL speakers via parametrize; these are for single-speaker tests.
DEFAULT_SPEAKER = "cicero"

# Gate test cases: name -> (enable_post_processing)
GATE_TEST_CASES = [
    ("basic_speech", True),
    ("raw_tts_no_postproc", False),
]

VOCALIZATION_CASES = [
    ("vocalization_sighs", "[sighs] I can't believe we made it."),
    ("vocalization_gasps", "[gasps] Who's there?"),
    ("vocalization_whispers", "[whispers] Don't make a sound."),
    ("vocalization_screams", "[screams] Get away from me!"),
    ("vocalization_moans", "[moans]"),
    ("vocalization_whimpers", "[whimpers] Please stop."),
    ("vocalization_struggling", "[struggling]"),
    ("vocalization_groans", "[groans]"),
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
    """Run gate checks on audio. Returns (gate_results, passed, errors)."""
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
