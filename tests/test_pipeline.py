"""Tests for audio generation pipeline."""

import pytest
from utilities.audio_generation_pipeline import _handle_ping, generate_audio
from utilities.app_config import AppConfig
from pathlib import Path


def test_ping_handler():
    """Test ping health check."""
    output_path, seed = _handle_ping()
    assert seed == 0
    assert Path(output_path).exists()
    assert "ping_silence" in output_path


@pytest.mark.parametrize("enable_post_processing", [True, False])
def test_generate_audio_with_post_processing_settings(enable_post_processing):
    """Test that generate_audio accepts post-processing parameters."""
    # This is a smoke test to verify the parameters are accepted
    # Actual generation would require a loaded model
    config = AppConfig()

    # Verify the function signature accepts these parameters
    import inspect
    sig = inspect.signature(generate_audio)
    assert 'enable_post_processing' in sig.parameters
    assert 'pitch_shift' in sig.parameters
    assert 'eq_intensity' in sig.parameters
    assert 'de_ess_intensity' in sig.parameters
    assert 'compressor_threshold_offset' in sig.parameters
    assert 'compressor_ratio' in sig.parameters
    assert 'target_loudness' in sig.parameters
