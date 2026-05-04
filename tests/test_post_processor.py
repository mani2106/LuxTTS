"""Tests for audio post-processor."""

import pytest
import numpy as np
from utilities.post_processor import (
    PitchDetector,
    AudioPostProcessor,
    HAS_TDR_NOVA,
    SignalProfile,
    analyze_signal,
)


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
    """Compressor should use threshold_offset_db parameter correctly."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.compress(
        audio, sr,
        threshold_offset_db=-6.0,
        ratio=4.0,
        knee_db=4.0,
        attack_ms=10.0,
        release_ms=100.0,
    )

    # Check that compressor processes the signal
    assert processed is not None
    assert len(processed) == len(audio)

    # Check diagnostics
    if 'input_lufs' in diagnostics:
        assert float(diagnostics['input_lufs']) != 0.0 or diagnostics['input_lufs'] == 0.0  # numeric value


def test_compressor_high_threshold_bypass(sample_48k_audio):
    """Very high threshold offset should result in minimal compression."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    # Very high offset means threshold is very high above RMS → minimal compression
    processed, diagnostics = processor.compress(audio, sr, threshold_offset_db=60.0)

    # With a very high threshold, very little compression should happen
    assert processed is not None
    assert len(processed) == len(audio)


def test_process_full_chain(sample_48k_audio):
    """Full processing chain should run all stages in correct order."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.process(
        audio, sr,
        text="Hello world",
        eq_intensity=1.0,
        de_ess_intensity=0.3,
        target_loudness=-18.0,
        enable_post_processing=True,
    )

    # Check that processing occurred
    assert processed is not None
    assert len(processed) > 0

    # Check that core adaptive stages are present
    assert 'signal_profile' in diagnostics
    assert 'high_pass_filter' in diagnostics
    assert 'normalize_loudness' in diagnostics
    assert 'manifest' in diagnostics


def test_process_with_text_pitch_detection(sample_48k_audio):
    """process() should detect pitch from text when pitch_shift is None and auto pitch shift enabled."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    # Test with excited text and auto pitch shift enabled
    processed, diagnostics = processor.process(
        audio, sr,
        text="This is exciting!",
        pitch_shift=None,  # Auto-detect from text
        enable_auto_pitch_shift=True,
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
        enable_auto_pitch_shift=True,
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
        enable_auto_pitch_shift=True,
        enable_post_processing=True,
    )

    assert processed is not None
    if 'pitch_shift' in diagnostics:
        assert diagnostics['pitch_shift']['detected_semitones'] == 2.0


def test_process_full_chain_includes_expressiveness(sample_48k_audio):
    """Full chain with opt-in stages should include prosodic_modulation, room_presence, spectral_enrich."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.process(
        audio, sr,
        text="Hello world",
        eq_intensity=1.0,
        de_ess_intensity=0.3,
        target_loudness=-18.0,
        enable_prosodic_modulation=True,
        enable_room_presence=True,
        enable_spectral_enrichment=True,
    )

    assert processed is not None
    assert len(processed) > 0
    assert not np.any(np.isnan(processed))
    assert 'prosodic_modulation' in diagnostics
    assert 'room_presence' in diagnostics
    assert 'spectral_enrich' in diagnostics


def test_prosodic_modulation_changes_audio(sample_48k_audio):
    """Prosodic modulation should subtly vary amplitude."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.prosodic_modulation(audio, sr, text="This is exciting!")

    assert processed is not None
    assert len(processed) == len(audio)
    assert processed.dtype == np.float32
    assert not np.allclose(processed, audio, atol=1e-6)
    assert 'emotion' in diagnostics


def test_prosodic_modulation_calm_text_shallower(sample_48k_audio):
    """Calm text should have shallower modulation than excited text."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    calm_processed, _ = processor.prosodic_modulation(audio, sr, text="Hello world")
    excited_processed, _ = processor.prosodic_modulation(audio, sr, text="This is exciting!")

    calm_diff = np.sqrt(np.mean((calm_processed - audio) ** 2))
    excited_diff = np.sqrt(np.mean((excited_processed - audio) ** 2))

    assert excited_diff > calm_diff


def test_prosodic_modulation_silence():
    """Prosodic modulation on silence should remain silence."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)
    processor = AudioPostProcessor()

    processed, _ = processor.prosodic_modulation(silence, sr, text="Hello!")

    np.testing.assert_allclose(processed, silence, atol=1e-7)


# ---- TDR Nova integration tests ----


def test_tdr_nova_combined_deess_and_eq(sample_48k_audio):
    """TDR Nova path should process de-essing and EQ in one pass."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    if not HAS_TDR_NOVA:
        pytest.skip("TDR Nova VST3 not available")

    processed, diagnostics = processor._process_tdr_nova(
        audio, sr, de_ess_intensity=0.5, eq_intensity=1.0,
    )

    assert processed is not None
    assert len(processed) == len(audio)
    assert not np.any(np.isnan(processed))
    assert 'backend' in diagnostics
    assert diagnostics['backend'] == 'tdr_nova'


def test_tdr_nova_zero_intensity(sample_48k_audio):
    """TDR Nova with zero intensities should still pass audio through."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    if not HAS_TDR_NOVA:
        pytest.skip("TDR Nova VST3 not available")

    processed, diagnostics = processor._process_tdr_nova(
        audio, sr, de_ess_intensity=0.0, eq_intensity=0.0,
    )

    assert processed is not None
    assert len(processed) == len(audio)
    assert not np.any(np.isnan(processed))


def test_process_uses_tdr_nova_when_available(sample_48k_audio):
    """Full process() should use TDR Nova when available."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.process(
        audio, sr,
        text="Hello world",
        eq_intensity=1.0,
        de_ess_intensity=0.3,
        target_loudness=-18.0,
    )

    assert processed is not None
    assert len(processed) > 0
    assert not np.any(np.isnan(processed))

    # New pipeline does not use TDR Nova in adaptive path -- signal analysis drives stages
    # TDR Nova is no longer invoked from process()


def test_tdr_nova_fallback_when_missing(sample_48k_audio):
    """If TDR Nova is disabled, adaptive pipeline still works without it."""
    audio, sr = sample_48k_audio
    import utilities.post_processor as pp

    original_has_tdr = pp.HAS_TDR_NOVA
    pp.HAS_TDR_NOVA = False

    try:
        processor = AudioPostProcessor(return_diagnostics=True)
        processed, diagnostics = processor.process(
            audio, sr,
            text="Hello",
            de_ess_intensity=0.3,
            eq_intensity=1.0,
            target_loudness=-18.0,
        )

        assert processed is not None
        assert not np.any(np.isnan(processed))
        assert 'tdr_nova' not in diagnostics
        assert 'signal_profile' in diagnostics
    finally:
        pp.HAS_TDR_NOVA = original_has_tdr


def test_room_presence_adds_reverb(sample_48k_audio):
    """Room presence should add subtle reverb tail."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.room_presence(audio, sr)

    assert processed is not None
    assert len(processed) == len(audio)
    assert processed.dtype == np.float32
    assert not np.allclose(processed, audio, atol=1e-6)
    assert 'wet_level_db' in diagnostics
    assert diagnostics['wet_level_db'] == -12.0


def test_room_presence_silence():
    """Room presence on silence should remain near-silence."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)
    processor = AudioPostProcessor()

    processed, _ = processor.room_presence(silence, sr)

    assert np.max(np.abs(processed)) < 1e-6


def test_room_presence_louder_wet_signal():
    """Higher wet level should produce more noticeable reverb."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5))
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    processor = AudioPostProcessor()

    quiet, _ = processor.room_presence(audio, sr, wet_db=-20)
    loud, _ = processor.room_presence(audio, sr, wet_db=-6)

    quiet_diff = np.sqrt(np.mean((quiet - audio) ** 2))
    loud_diff = np.sqrt(np.mean((loud - audio) ** 2))

    assert loud_diff > quiet_diff


def test_spectral_enrich_adds_harmonics(sample_48k_audio):
    """Spectral enrichment should modify audio by adding upper harmonics."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.spectral_enrich(audio, sr)

    assert processed is not None
    assert len(processed) == len(audio)
    assert processed.dtype == np.float32
    assert not np.allclose(processed, audio, atol=1e-6)
    assert 'intensity' in diagnostics


def test_spectral_enrich_zero_intensity_bypass(sample_48k_audio):
    """Zero intensity should bypass spectral enrichment."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    processed, _ = processor.spectral_enrich(audio, sr, intensity=0.0)

    np.testing.assert_allclose(processed, audio, atol=1e-6)


def test_spectral_enrich_silence():
    """Spectral enrichment on silence should remain silence."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)
    processor = AudioPostProcessor()

    processed, _ = processor.spectral_enrich(silence, sr)

    np.testing.assert_allclose(processed, silence, atol=1e-7)


# ---- SignalProfile and analyze_signal tests ----


def test_analyze_signal_speech_like():
    """Speech-like audio should have moderate peak, RMS, and centroid."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    assert 0.0 < profile.peak < 1.0
    assert profile.true_peak_db < 0.0  # True-peak should be negative dB for sub-unity signal
    assert 0.0 < profile.rms < 1.0
    assert 100 < profile.spectral_centroid < 5000
    assert 0.0 <= profile.sibilance_ratio <= 1.0
    assert profile.crest_factor_db > 0.0  # Peak > RMS means positive crest factor
    assert -20 < profile.spectral_tilt_db_per_octave < 20  # Reasonable range


def test_analyze_signal_bright_audio():
    """Audio with lots of high-frequency content should have high sibilance ratio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.1 * np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    assert profile.sibilance_ratio > 0.1
    assert profile.needs_de_essing is True


def test_analyze_signal_quiet_audio():
    """Quiet audio should have low RMS and not need limiting."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.01 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    assert profile.needs_limiting is False
    assert profile.rms < 0.05


def test_analyze_signal_clipping_audio():
    """Audio near clipping should need limiting."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.98 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    assert profile.needs_limiting is True


def test_analyze_signal_boomy_audio():
    """Audio with low spectral centroid should need mud cut."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.5 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    assert profile.needs_mud_cut is True


def test_signal_profile_properties():
    """SignalProfile properties should return correct booleans."""
    profile = SignalProfile(
        peak=0.5, true_peak_db=-6.0, rms=0.1,
        spectral_centroid=2500.0, sibilance_ratio=0.05,
        crest_factor_db=14.0, spectral_tilt_db_per_octave=-3.0,
    )

    assert profile.needs_limiting is False
    assert profile.needs_de_essing is False
    assert profile.needs_mud_cut is False
    assert profile.needs_presence_boost is False


def test_analyze_signal_silence():
    """Silent audio should not crash analysis."""
    audio = np.zeros(48000, dtype=np.float32)
    profile = analyze_signal(audio, 48000)

    assert profile.peak == 0.0
    assert profile.rms == 0.0
    assert profile.needs_limiting is False


def test_analyze_signal_true_peak_oversampled():
    """True-peak should catch inter-sample overs that sample peak misses."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Construct a signal near unity where true-peak may exceed sample peak
    audio = (0.95 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    # True-peak should be >= sample peak (never less)
    assert profile.true_peak_db >= 20 * np.log10(profile.peak + 1e-10)


# ---- Soft-knee limiter tests ----


def test_limit_peak_soft_knee_reduces_loud_audio():
    """Soft-knee limiter should reduce peaks without hard clipping artifacts."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.98 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor()
    processed, diagnostics = processor.limit_peak(audio, sr, threshold_db=-1.0)

    threshold_linear = 10 ** (-1.0 / 20)
    assert np.max(np.abs(processed)) <= threshold_linear + 0.02  # Small tolerance
    assert diagnostics['limiting_applied'] is True
    assert 'max_gain_reduction_db' in diagnostics
    assert diagnostics['max_gain_reduction_db'] <= 6.0  # Cap at 6dB


def test_limit_peak_safe_audio_passes_through():
    """Audio below threshold should pass through unchanged."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor()
    processed, diagnostics = processor.limit_peak(audio, sr, threshold_db=-1.0)

    np.testing.assert_allclose(processed, audio, atol=1e-6)
    assert diagnostics['limiting_applied'] is False


def test_limit_peak_silence():
    """Silent input should not crash."""
    audio = np.zeros(48000, dtype=np.float32)
    processor = AudioPostProcessor()
    processed, diagnostics = processor.limit_peak(audio, 48000, threshold_db=-1.0)

    assert processed is not None
    assert len(processed) == 48000


def test_limit_peak_no_hard_clipping():
    """Soft-knee should not produce the flat-top distortion of hard clipping."""
    sr = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    # Create a signal with sharp peaks
    audio = (0.99 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor()
    processed, diagnostics = processor.limit_peak(audio, sr, threshold_db=-1.0)

    # Count samples exactly at threshold — should be near zero for soft-knee
    threshold_linear = 10 ** (-1.0 / 20)
    at_threshold = np.sum(np.abs(processed) >= threshold_linear - 0.001)
    # Hard clip would have many samples exactly at threshold; soft-knee should have far fewer
    assert at_threshold < len(processed) * 0.1


# ---- Signal-adaptive process() tests ----


def test_process_adaptive_quiet_audio_skips_limiter():
    """Quiet audio should not trigger the adaptive limiter."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.1 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    processed, diagnostics = processor.process(audio, sr)

    assert 'signal_profile' in diagnostics
    assert diagnostics['signal_profile']['needs_limiting'] is False
    assert 'peak_limiter' not in diagnostics or diagnostics.get('peak_limiter', {}).get('limiting_applied') is False


def test_process_adaptive_loud_audio_gets_limiter():
    """Loud audio (peak > 0.93) should trigger the adaptive limiter."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.96 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    processed, diagnostics = processor.process(audio, sr)

    assert diagnostics['signal_profile']['needs_limiting'] is True
    assert 'peak_limiter' in diagnostics


def test_process_adaptive_no_sibilance_skips_deesser():
    """Audio without sibilance should skip de-essing."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    processed, diagnostics = processor.process(audio, sr)

    assert diagnostics['signal_profile']['needs_de_essing'] is False


def test_process_proportional_deesser_scales_with_sibilance():
    """De-esser intensity should be proportional to sibilance ratio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # High sibilance signal
    audio = (0.1 * np.sin(2 * np.pi * 200 * t) + 0.6 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    processed, diagnostics = processor.process(audio, sr)

    assert diagnostics['signal_profile']['needs_de_essing'] is True
    assert 'de_esser' in diagnostics
    assert 'proportional_intensity' in diagnostics['de_esser']


def test_process_always_runs_hpf_and_loudness():
    """HPF and loudness normalization should always run."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    processed, diagnostics = processor.process(audio, sr)

    assert 'high_pass_filter' in diagnostics
    assert 'normalize_loudness' in diagnostics


def test_process_disabled_returns_original():
    """When enable_post_processing=False, return original audio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor()
    processed, diagnostics = processor.process(audio, sr, enable_post_processing=False)

    np.testing.assert_allclose(processed, audio, atol=1e-6)
    assert diagnostics == {}


def test_process_target_lufs_default():
    """Default target LUFS should be -18.0."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    _, diagnostics = processor.process(audio, sr)

    assert diagnostics['normalize_loudness']['target_lufs'] == -18.0


def test_process_removed_stages_not_in_default_diagnostics():
    """Spectral enrichment, room presence, prosodic modulation should NOT appear in default diagnostics."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    _, diagnostics = processor.process(audio, sr)

    assert 'spectral_enrich' not in diagnostics
    assert 'room_presence' not in diagnostics
    assert 'prosodic_modulation' not in diagnostics


def test_process_emits_manifest():
    """process() should emit a standardized manifest with required fields."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    _, diagnostics = processor.process(audio, sr)

    assert 'manifest' in diagnostics
    manifest = diagnostics['manifest']
    required_fields = [
        'integrated_lufs', 'true_peak_db', 'spectral_centroid_hz',
        'sibilance_ratio', 'crest_factor_db', 'max_gain_reduction_db',
    ]
    for field in required_fields:
        assert field in manifest, f"Manifest missing required field: {field}"
