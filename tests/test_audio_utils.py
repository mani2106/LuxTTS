"""Tests for audio utilities."""

import tempfile

import numpy as np
import pytest

from utilities.audio_utils import create_silence, load_wav_file, save_wav_file
from pathlib import Path


def test_create_silence():
    """Test silence generation."""
    silence = create_silence(0.5, sample_rate=48000)
    assert silence.shape == (24000,)
    assert silence.sum() == 0


def test_save_and_load_wav():
    """Test round-trip WAV save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4800))
        wav_path = Path(tmpdir) / "test.wav"

        # Save
        save_wav_file(test_audio, wav_path, sample_rate=48000)
        assert wav_path.exists()

        # Load
        loaded, sr = load_wav_file(wav_path)
        assert sr == 48000
        assert len(loaded) == 4800
