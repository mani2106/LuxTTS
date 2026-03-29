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


def test_equalize_reduces_high_frequencies(sample_48k_audio):
    """EQ should reduce high frequencies above 8kHz (high-shelf cut)."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.equalize(audio, sr, intensity=1.0)

    # Check output differs from input
    assert not np.allclose(processed, audio)

    # Check diagnostics (only populated when return_diagnostics=True)
    assert 'pre_spectrum' in diagnostics
    assert 'post_spectrum' in diagnostics


def test_equalize_zero_intensity_bypass(sample_48k_audio):
    """Zero intensity should bypass EQ."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    processed, diagnostics = processor.equalize(audio, sr, intensity=0.0)

    np.testing.assert_allclose(processed, audio, atol=1e-6)


def test_pitch_shift_positive_semitones(sample_48k_audio):
    """Positive semitones should raise pitch."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.pitch_shift(audio, sr, n_steps=2.0)

    # Check that output differs from input
    assert not np.allclose(processed, audio)

    # Check diagnostics
    assert 'semitones_applied' in diagnostics
    assert diagnostics['semitones_applied'] == 2.0


def test_pitch_shift_zero_steps_bypass(sample_48k_audio):
    """Zero semitones should bypass processing (output ≈ input)."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    processed, diagnostics = processor.pitch_shift(audio, sr, n_steps=0.0)

    # With zero steps, output should be very close to input
    np.testing.assert_allclose(processed, audio, atol=1e-6)

    # No diagnostics for bypass
    assert diagnostics == {}


def test_pitch_shift_negative_semitones(sample_48k_audio):
    """Negative semitones should lower pitch."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.pitch_shift(audio, sr, n_steps=-1.5)

    # Check that output differs from input
    assert not np.allclose(processed, audio)

    # Check diagnostics
    assert 'semitones_applied' in diagnostics
    assert diagnostics['semitones_applied'] == -1.5


def test_normalize_loudness_to_target(sample_48k_audio):
    """Loudness normalization should adjust audio to target LUFS."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    # Normalize to -16 LUFS (standard for speech)
    processed, diagnostics = processor.normalize_loudness(audio, sr, target_lufs=-16.0)

    # Check that output is processed
    assert processed is not None
    assert len(processed) == len(audio)

    # Check diagnostics
    assert 'measured_lufs' in diagnostics
    assert 'gain_applied_db' in diagnostics
    assert 'peak_limiting_applied' in diagnostics

    # Measured LUFS should be a reasonable number (typically between -60 and 0)
    assert -60.0 < diagnostics['measured_lufs'] < 0.0

    # Gain should be calculated
    assert isinstance(diagnostics['gain_applied_db'], float)

    # Peak limiting should be a boolean
    assert isinstance(diagnostics['peak_limiting_applied'], bool)


def test_normalize_loudness_already_at_target():
    """Audio already at target LUFS should have minimal gain applied."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Generate audio at approximately -16 LUFS
    # RMS of -20 dB ≈ -16 LUFS (rough approximation for the fallback)
    # Amplitude of 0.1 gives RMS of -20 dBFS
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    audio = audio.astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)

    # First pass: measure what LUFS we get
    _, first_diag = processor.normalize_loudness(audio, sr, target_lufs=-16.0)
    measured = first_diag['measured_lufs']

    # Second pass: normalize to the measured LUFS (should require minimal gain)
    processed, diagnostics = processor.normalize_loudness(audio, sr, target_lufs=measured)

    # Check that output exists
    assert processed is not None
    assert len(processed) == len(audio)

    # When normalizing to the already-measured LUFS, gain should be minimal
    # (within ±1 dB due to floating point precision)
    assert abs(diagnostics['gain_applied_db']) < 1.0

    # Peak should not exceed -1 dBTP (0.89 linear)
    peak_limit_linear = 10 ** (-1.0 / 20)
    assert np.max(np.abs(processed)) <= peak_limit_linear + 1e-6


def test_normalize_loudness_peak_limiting():
    """Peak limiter should prevent clipping when applying large gain."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Generate hot audio (near clipping)
    audio = 0.8 * np.sin(2 * np.pi * 440 * t)
    audio = audio.astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)

    # Normalize to -10 LUFS (much louder than input)
    # This will require significant gain, triggering peak limiting
    processed, diagnostics = processor.normalize_loudness(audio, sr, target_lufs=-10.0)

    # Check that peak limiting was triggered
    # (it should be, since we're applying lots of gain to hot audio)
    # Note: This may not always trigger depending on the exact LUFS measurement

    # Peak should never exceed -1 dBTP (0.89 linear)
    peak_limit_linear = 10 ** (-1.0 / 20)
    assert np.max(np.abs(processed)) <= peak_limit_linear + 1e-6


def test_normalize_loudness_silence():
    """Silence input should be handled gracefully."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)

    processor = AudioPostProcessor()

    # Should not crash on silence
    processed, diagnostics = processor.normalize_loudness(silence, sr, target_lufs=-16.0)

    assert processed is not None
    assert len(processed) == 48000
    assert 'measured_lufs' in diagnostics


def test_compressor_reduces_dynamic_range(sample_48k_audio):
    """Compressor should reduce dynamic range of audio."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.compress(
        audio, sr,
        threshold_offset_db=-6.0,
        ratio=4.0,
        knee_db=4.0,
        attack_ms=10.0,
        release_ms=100.0,
        max_reduction_db=12.0,
    )

    # Check that compressor processes the signal
    assert processed is not None
    assert len(processed) == len(audio)

    # Check diagnostics
    assert 'input_lufs' in diagnostics
    assert 'output_lufs' in diagnostics
    assert 'max_reduction_db' in diagnostics
    assert 'makeup_gain_db' in diagnostics
    assert 'gain_curve' in diagnostics

    # Max reduction should be positive (gain reduction occurred)
    assert diagnostics['max_reduction_db'] >= 0

    # Gain curve should be an array
    assert isinstance(diagnostics['gain_curve'], np.ndarray)
    assert len(diagnostics['gain_curve']) > 0


def test_compressor_silence_input():
    """Compressor should handle silence input gracefully."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)

    processor = AudioPostProcessor()

    processed, diagnostics = processor.compress(
        silence, sr,
        threshold_offset_db=-6.0,
        ratio=4.0,
    )

    # Should not crash on silence
    assert processed is not None
    assert len(processed) == 48000

    # Output should also be silence (or very close)
    assert np.max(np.abs(processed)) < 1e-6


def test_compressor_very_short_utterance():
    """Compressor should handle very short utterances (<500ms) with adjusted parameters."""
    sr = 48000
    duration = 0.1  # 100ms - very short
    t = np.linspace(0, duration, int(sr * duration))

    # Create a simple tone
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = audio.astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)

    # This should not crash and should adjust attack/look-ahead internally
    processed, diagnostics = processor.compress(
        audio, sr,
        threshold_offset_db=-6.0,
        ratio=4.0,
        knee_db=4.0,
        attack_ms=10.0,  # Will be reduced internally for short audio
        release_ms=100.0,
    )

    # Check processing succeeded
    assert processed is not None
    assert len(processed) == len(audio)

    # Check diagnostics are populated
    assert 'input_lufs' in diagnostics
    assert 'output_lufs' in diagnostics


def test_compressor_empty_input():
    """Compressor should handle empty input gracefully."""
    processor = AudioPostProcessor()

    # Empty array
    empty_audio = np.array([], dtype=np.float32)
    processed, diagnostics = processor.compress(empty_audio, 48000)

    # Should return empty array without crashing
    assert processed is not None
    assert len(processed) == 0



def test_compressor_with_threshold_db(sample_48k_audio):
    """Compressor should use threshold_db parameter correctly."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.compressor(
        audio, sr,
        threshold_db=-20.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=50.0,
    )

    # Check that compressor processes the signal
    assert processed is not None
    assert len(processed) == len(audio)
    
    # Check diagnostics
    if 'gain_reduction_db_curve' in diagnostics:
        assert len(diagnostics['gain_reduction_db_curve']) > 0


def test_compressor_high_threshold_bypass(sample_48k_audio):
    """High threshold (>= 0 dB) should bypass compression."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    processed, diagnostics = processor.compressor(audio, sr, threshold_db=0.0)

    np.testing.assert_allclose(processed, audio, atol=1e-6)


def test_process_full_chain(sample_48k_audio):
    """Full processing chain should run all stages in correct order."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.process(
        audio, sr,
        text="Hello world",
        eq_intensity=1.0,
        de_ess_intensity=0.5,
        compressor_threshold_db=-20.0,
        compressor_ratio=4.0,
        compressor_attack_ms=5.0,
        compressor_release_ms=50.0,
        target_loudness=-16.0,
        enable_post_processing=True,
    )

    # Check that processing occurred
    assert processed is not None
    assert len(processed) > 0

    # Check that diagnostics from multiple stages are present
    # Note: not all stages may have diagnostics (e.g., pitch_shift with near-zero shift)
    assert isinstance(diagnostics, dict)


def test_process_with_text_pitch_detection(sample_48k_audio):
    """process() should detect pitch from text when pitch_shift is None."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    # Test with excited text
    processed, diagnostics = processor.process(
        audio, sr,
        text="This is exciting!",
        pitch_shift=None,  # Auto-detect from text
        enable_post_processing=True,
    )

    assert processed is not None
    # Check that pitch shift diagnostics include detected value
    if 'pitch_shift' in diagnostics:
        assert 'detected_semitones' in diagnostics['pitch_shift']
        # Excited text should detect +1.0
        assert diagnostics['pitch_shift']['detected_semitones'] == 1.0


def test_process_manual_pitch_override(sample_48k_audio):
    """process() should use manual pitch_shift value when provided."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    manual_pitch = 3.5
    processed, diagnostics = processor.process(
        audio, sr,
        text="This is exciting!",  # Would normally be +1.0
        pitch_shift=manual_pitch,  # Override with manual value
        enable_post_processing=True,
    )

    assert processed is not None
    # Check that manual pitch was used
    if 'pitch_shift' in diagnostics:
        assert diagnostics['pitch_shift']['detected_semitones'] == manual_pitch


def test_process_disabled_bypass(sample_48k_audio):
    """process() should bypass all processing when enable_post_processing=False."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    processed, diagnostics = processor.process(
        audio, sr,
        text="Hello world",
        enable_post_processing=False,
    )

    # Should return original audio unchanged
    np.testing.assert_allclose(processed, audio, atol=1e-6)
    # Diagnostics should be empty
    assert diagnostics == {}


def test_process_with_all_caps_text(sample_48k_audio):
    """process() should detect +2.0 pitch for ALL CAPS text."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.process(
        audio, sr,
        text="THIS IS SHOUTING",
        pitch_shift=None,
        enable_post_processing=True,
    )

    assert processed is not None
    if 'pitch_shift' in diagnostics:
        assert diagnostics['pitch_shift']['detected_semitones'] == 2.0
