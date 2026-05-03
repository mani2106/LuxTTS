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
