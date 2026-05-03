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
