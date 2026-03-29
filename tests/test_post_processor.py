"""Tests for audio post-processor."""

import pytest
from utilities.post_processor import PitchDetector


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
