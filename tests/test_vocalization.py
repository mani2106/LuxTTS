"""Tests for vocalization audio tag processing."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from utilities.vocalization.tag_parser import parse_tags, Segment, SegmentType


def test_parse_no_tags():
    """Text without tags returns single speech segment."""
    segments = parse_tags("Hello world")
    assert len(segments) == 1
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "Hello world"


def test_parse_single_tag_at_start():
    """Tag at start splits correctly."""
    segments = parse_tags("[sighs] Hello there")
    assert len(segments) == 2
    assert segments[0].type == SegmentType.VOCALIZATION
    assert segments[0].tag == "sighs"
    assert segments[1].type == SegmentType.SPEECH
    assert segments[1].text == "Hello there"


def test_parse_tag_in_middle():
    """Tag in middle splits into three segments."""
    segments = parse_tags("Hello [gasps] how are you?")
    assert len(segments) == 3
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "Hello "
    assert segments[1].type == SegmentType.VOCALIZATION
    assert segments[1].tag == "gasps"
    assert segments[2].type == SegmentType.SPEECH
    assert segments[2].text == "how are you?"


def test_parse_tag_at_end():
    """Tag at end splits correctly."""
    segments = parse_tags("Goodbye [sighs]")
    assert len(segments) == 2
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "Goodbye "
    assert segments[1].type == SegmentType.VOCALIZATION
    assert segments[1].tag == "sighs"


def test_parse_multiple_consecutive_tags():
    """Multiple consecutive tags each become separate segments."""
    segments = parse_tags("[gasps] [screams] Help!")
    assert len(segments) == 3
    assert segments[0].tag == "gasps"
    assert segments[1].tag == "screams"
    assert segments[2].type == SegmentType.SPEECH
    assert segments[2].text == "Help!"


def test_parse_multi_word_tag():
    """Multi-word tags are supported."""
    segments = parse_tags("[breathes heavily] We must go.")
    assert len(segments) == 2
    assert segments[0].tag == "breathes heavily"
    assert segments[1].text == "We must go."


def test_parse_unknown_tag_treated_as_pause():
    """Unknown tags become vocalization segments with tag name preserved."""
    segments = parse_tags("[unknown_tag] Hello")
    assert len(segments) == 2
    assert segments[0].type == SegmentType.VOCALIZATION
    assert segments[0].tag == "unknown_tag"


def test_parse_malformed_tag_treated_as_text():
    """Malformed tags (missing bracket, nested) are treated as regular text."""
    # Missing closing bracket
    segments = parse_tags("[sighs Hello world")
    assert len(segments) == 1
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "[sighs Hello world"

    # Nested brackets
    segments = parse_tags("[sighs[gasps]] Hello")
    assert len(segments) == 1
    assert segments[0].type == SegmentType.SPEECH


def test_parse_empty_after_tags():
    """Text with only tags returns only vocalization segments."""
    segments = parse_tags("[sighs] [gasps]")
    assert len(segments) == 2
    assert all(s.type == SegmentType.VOCALIZATION for s in segments)


def test_has_vocalizations():
    """has_vocalizations() helper returns True when any vocalization segment present."""
    from utilities.vocalization.tag_parser import has_vocalizations

    assert has_vocalizations(parse_tags("[sighs] Hello"))
    assert has_vocalizations(parse_tags("Hello [gasps]"))
    assert not has_vocalizations(parse_tags("Hello world"))


# DSP Engine Tests
@pytest.fixture
def sample_48k_audio():
    """Generate 48kHz test audio."""
    duration = 1.0
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32), sr


def test_dsp_pitch_shift(sample_48k_audio):
    """Pitch shift changes audio pitch."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "pitch_shift", "semitones": 2.0})
    assert len(result) == len(audio)
    assert not np.allclose(result, audio)


def test_dsp_breath_noise(sample_48k_audio):
    """Breath noise adds noise layer."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "breath_noise", "amplitude": 0.1})
    assert len(result) == len(audio)


def test_dsp_low_pass_filter(sample_48k_audio):
    """Low pass filter removes high frequencies."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "low_pass_filter", "cutoff_hz": 1000})
    assert len(result) == len(audio)


def test_dsp_high_pass_filter(sample_48k_audio):
    """High pass filter removes low frequencies."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "high_pass_filter", "cutoff_hz": 1000})
    assert len(result) == len(audio)


def test_dsp_distortion(sample_48k_audio):
    """Distortion adds waveshaping."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "distortion", "intensity": 0.5})
    assert len(result) == len(audio)


def test_dsp_volume(sample_48k_audio):
    """Volume changes gain."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "volume", "factor": 0.5})
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) < np.max(np.abs(audio))


def test_dsp_fade_out(sample_48k_audio):
    """Fade out reduces amplitude at end."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "fade_out", "duration_s": 0.5})
    assert len(result) == len(audio)
    # End should be quieter than beginning
    assert np.max(np.abs(result[-100:])) < np.max(np.abs(result[:100]))


def test_dsp_fade_in(sample_48k_audio):
    """Fade in increases amplitude at start."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "fade_in", "duration_s": 0.5})
    assert len(result) == len(audio)
    # Start should be quieter than end
    assert np.max(np.abs(result[:100])) < np.max(np.abs(result[-100:]))


def test_dsp_chain_multiple_effects(sample_48k_audio):
    """Chain multiple effects in sequence."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    effects = [
        {"type": "pitch_shift", "semitones": -2.0},
        {"type": "low_pass_filter", "cutoff_hz": 2000},
        {"type": "breath_noise", "amplitude": 0.05},
        {"type": "fade_out", "duration_s": 0.3},
    ]
    result = engine.apply_chain(audio, sr, effects)
    assert len(result) == len(audio)


def test_dsp_unknown_effect_type(sample_48k_audio):
    """Unknown effect type raises ValueError."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.dsp_engine import DSPEngine
    engine = DSPEngine()
    with pytest.raises(ValueError, match="Unknown effect type"):
        engine.apply_effect(audio, sr, {"type": "unknown_effect"})
