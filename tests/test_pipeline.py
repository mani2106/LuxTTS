"""Tests for audio generation pipeline."""

import pytest
from utilities.audio_generation_pipeline import _handle_ping
from pathlib import Path


def test_ping_handler():
    """Test ping health check."""
    output_path, seed = _handle_ping()
    assert seed == 0
    assert Path(output_path).exists()
    assert "ping_silence" in output_path
