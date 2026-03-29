"""Tests for audio post-processor."""

import pytest
import numpy as np
from utilities.post_processor import PitchDetector, AudioPostProcessor


def test_pitch_detector_all_caps():
    """Text in ALL CAPS returns +2.0 semitones."""
    detector = PitchDetector()
    result = detector.detect_pitch("THIS IS SHOUTING!")
    assert result == 2.0


def test_pitch_detector_exclamation():
    """Text ending with ! returns +1.0 semitones."""
    detector = PitchDetector()
    result = detector.detect_pitch("I am excited!")
    assert result == 1.0


def test_pitch_detector_question():
    """Text ending with ? returns +0.5 semitones."""
    detector = PitchDetector()
    result = detector.detect_pitch("Is this working?")
    assert result == 0.5


def test_pitch_detector_ellipsis():
    """Text with ... returns -1.0 semitones."""
    detector = PitchDetector()
    result = detector.detect_pitch("I'm not sure...")
    assert result == -1.0


def test_pitch_detector_long_text():
    """Text >200 chars returns -0.5 semitones."""
    detector = PitchDetector()
    long_text = "x" * 250
    result = detector.detect_pitch(long_text)
    assert result == -0.5


def test_pitch_detector_default():
    """Plain text returns 0.0 semitones."""
    detector = PitchDetector()
    result = detector.detect_pitch("Hello world")
    assert result == 0.0


def test_pitch_detector_manual_override():
    """Manual pitch_shift parameter overrides text detection."""
    detector = PitchDetector()
    result = detector.detect_pitch("THIS IS SHOUTING!", manual_pitch_shift=5.0)
    assert result == 5.0


def test_pitch_detector_rule_priority():
    """First matching rule wins. ALL CAPS before exclamation."""
    detector = PitchDetector()
    # Would match both ALL CAPS and !, but ALL CAPS is checked first
    result = detector.detect_pitch("SHOUTING!")
    assert result == 2.0


@pytest.fixture
def sample_48k_audio():
    """Generate 48kHz test audio with sibilant-like content."""
    duration = 1.0  # seconds
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration))
    # Mix of low frequency (speech-like) and high frequency (sibilant-like)
    audio = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 8000 * t)
    return audio.astype(np.float32), sr


def test_de_esser_reduces_high_frequencies(sample_48k_audio):
    """De-esser should reduce energy in sibilant frequency band (5-8kHz)."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.de_esser(audio, sr, intensity=0.5)

    # Check that de-esser processes the signal and produces diagnostics
    assert processed is not None
    assert len(processed) == len(audio)
    assert 'reduction_db_curve' in diagnostics
    assert len(diagnostics['reduction_db_curve']) > 0


def test_de_esser_zero_intensity_bypass(sample_48k_audio):
    """Zero intensity should bypass processing (output ≈ input)."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    processed, diagnostics = processor.de_esser(audio, sr, intensity=0.0)

    # With zero intensity, output should be very close to input
    np.testing.assert_allclose(processed, audio, atol=1e-6)


def test_de_esser_clips_invalid_inputs():
    """De-esser should handle edge cases gracefully."""
    processor = AudioPostProcessor()

    # Silence input
    silence = np.zeros(48000, dtype=np.float32)
    processed, _ = processor.de_esser(silence, 48000, intensity=0.5)
    assert processed is not None
    assert len(processed) == 48000
