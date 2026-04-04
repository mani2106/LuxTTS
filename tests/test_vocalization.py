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


# Recipe Loader Tests
def test_load_recipes():
    """Load all recipes from JSON."""
    from utilities.vocalization.recipes import load_recipes
    recipes = load_recipes()
    assert "sighs" in recipes
    assert "screams" in recipes
    assert "pause" in recipes


def test_get_recipe_existing():
    """Get recipe for existing tag."""
    from utilities.vocalization.recipes import get_recipe
    recipe = get_recipe("sighs")
    assert recipe["tts_text"] == "haaah"
    assert "effects" in recipe
    assert len(recipe["effects"]) > 0


def test_get_recipe_unknown():
    """Get recipe for unknown tag returns None."""
    from utilities.vocalization.recipes import get_recipe
    recipe = get_recipe("unknown_tag_xyz")
    assert recipe is None


def test_recipe_sighs_structure():
    """Sighs recipe has expected structure."""
    from utilities.vocalization.recipes import get_recipe
    recipe = get_recipe("sighs")
    assert "tts_text" in recipe
    assert "effects" in recipe
    assert recipe["tts_text"] == "haaah"

    # Check effects
    effects = recipe["effects"]
    assert any(e["type"] == "pitch_shift" for e in effects)
    assert any(e["type"] == "breath_noise" for e in effects)


def test_recipe_whisper_mode():
    """Whispers recipe has modify_speech mode."""
    from utilities.vocalization.recipes import get_recipe
    recipe = get_recipe("whispers")
    assert recipe.get("mode") == "modify_speech"
    assert recipe.get("tts_text") is None


def test_recipe_pause():
    """Pause recipe has duration but no tts_text."""
    from utilities.vocalization.recipes import get_recipe
    recipe = get_recipe("pause")
    assert recipe.get("tts_text") is None
    assert "duration_s" in recipe


# Stitcher Tests
def test_stitch_single_segment(sample_48k_audio):
    """Single segment returns unchanged."""
    audio, sr = sample_48k_audio
    from utilities.vocalization.stitcher import stitch_segments
    segments = [(audio, sr)]
    result, result_sr = stitch_segments(segments, crossfade_ms=50)
    assert np.allclose(result, audio)
    assert result_sr == sr


def test_stitch_two_segments_with_crossfade(sample_48k_audio):
    """Two segments are crossfaded."""
    audio1, sr = sample_48k_audio
    audio2 = np.roll(audio1, 1000)  # Different content
    from utilities.vocalization.stitcher import stitch_segments
    segments = [(audio1, sr), (audio2, sr)]
    result, result_sr = stitch_segments(segments, crossfade_ms=100)
    # Result should be shorter than sum (due to crossfade)
    expected_length = len(audio1) + len(audio2) - int(0.1 * sr)
    assert len(result) == expected_length
    assert result_sr == sr


def test_stitch_three_segments(sample_48k_audio):
    """Three segments stitched sequentially."""
    audio1, sr = sample_48k_audio
    audio2 = audio1 * 0.8
    audio3 = audio1 * 0.6
    from utilities.vocalization.stitcher import stitch_segments
    segments = [(audio1, sr), (audio2, sr), (audio3, sr)]
    result, result_sr = stitch_segments(segments, crossfade_ms=50)
    assert result_sr == sr
    assert len(result) < len(audio1) + len(audio2) + len(audio3)


def test_stitch_zero_crossfade(sample_48k_audio):
    """Zero crossfade concatenates directly."""
    audio1, sr = sample_48k_audio
    audio2 = audio1 * 0.5
    from utilities.vocalization.stitcher import stitch_segments
    segments = [(audio1, sr), (audio2, sr)]
    result, result_sr = stitch_segments(segments, crossfade_ms=0)
    expected_length = len(audio1) + len(audio2)
    assert len(result) == expected_length


def test_stitch_empty_segments():
    """Empty segment list returns empty audio."""
    from utilities.vocalization.stitcher import stitch_segments
    result, sr = stitch_segments([], crossfade_ms=50)
    assert len(result) == 0


# Vocalization Generator Tests
@pytest.fixture
def mock_model():
    """Mock LuxTTS model."""
    model = Mock()
    # Mock generate_speech to return simple audio
    dummy_audio = torch.randn(48000).numpy()  # 1 second of 48kHz
    model.generate_speech = MagicMock(return_value=torch.from_numpy(dummy_audio))
    return model


@pytest.fixture
def encode_dict():
    """Mock speaker encoding dict."""
    return {"speaker_embed": torch.randn(128)}


def test_vocalization_generator_sighs(mock_model, encode_dict):
    """Generate sighs vocalization."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.tag_parser import Segment, SegmentType
    segment = Segment(type=SegmentType.VOCALIZATION, tag="sighs")
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) > 0
    assert sr == 48000


def test_vocalization_generator_pause(mock_model, encode_dict):
    """Pause tag returns silence without TTS."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.tag_parser import Segment, SegmentType
    segment = Segment(type=SegmentType.VOCALIZATION, tag="pause")
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) == int(0.3 * 48000)  # 0.3s pause
    assert sr == 48000
    # Silence should be all zeros
    assert np.allclose(audio, 0)


def test_vocalization_generator_unknown_tag(mock_model, encode_dict):
    """Unknown tag returns 0.3s silence with warning."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.tag_parser import Segment, SegmentType
    segment = Segment(type=SegmentType.VOCALIZATION, tag="unknown_tag")
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) == int(0.3 * 48000)


def test_vocalization_generator_with_max_duration(mock_model, encode_dict):
    """Max duration truncates long TTS output."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.tag_parser import Segment, SegmentType
    # Return 2 seconds of audio
    long_audio = torch.randn(96000).numpy()
    mock_model.generate_speech = MagicMock(return_value=torch.from_numpy(long_audio))

    segment = Segment(type=SegmentType.VOCALIZATION, tag="gasps")  # max_duration_s: 0.4
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) <= int(0.5 * 48000)  # ~0.4s + tolerance


def test_vocalization_generator_custom_params(mock_model, encode_dict):
    """Custom TTS params are passed through."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.tag_parser import Segment, SegmentType
    segment = Segment(type=SegmentType.VOCALIZATION, tag="sighs")
    generator = VocalizationGenerator(
        mock_model,
        num_steps=4,
        guidance_scale=2.5,
        speed=0.7,
        t_shift=0.8,
    )
    audio, sr = generator.generate(segment, encode_dict)
    # Verify generate_speech was called with custom params
    call_args = mock_model.generate_speech.call_args
    assert call_args.kwargs['num_steps'] == 4
    assert call_args.kwargs['guidance_scale'] == 2.5


def test_vocalization_generator_is_modify_speech(mock_model):
    """is_modify_speech() identifies whisper mode."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    generator = VocalizationGenerator(mock_model)
    assert generator.is_modify_speech("whispers")
    assert not generator.is_modify_speech("sighs")


# Integration Tests - test component integration without full async pipeline
def test_integration_tag_parser_and_generator(mock_model, encode_dict):
    """Tag parser and vocalization generator work together."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.tag_parser import parse_tags, SegmentType

    text = "[sighs] Hello [gasps] world"
    segments = parse_tags(text)

    # Should parse into 4 segments
    assert len(segments) == 4
    assert segments[0].type == SegmentType.VOCALIZATION
    assert segments[1].type == SegmentType.SPEECH
    assert segments[2].type == SegmentType.VOCALIZATION
    assert segments[3].type == SegmentType.SPEECH

    # Generate audio for vocalization segments
    generator = VocalizationGenerator(mock_model)
    audio1, sr1 = generator.generate(segments[0], encode_dict)
    audio2, sr2 = generator.generate(segments[2], encode_dict)

    assert sr1 == 48000
    assert sr2 == 48000
    assert len(audio1) > 0
    assert len(audio2) > 0


def test_integration_stitch_with_generated_audio(mock_model, encode_dict):
    """Stitcher works with generated audio."""
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.tag_parser import parse_tags, has_vocalizations
    from utilities.vocalization.stitcher import stitch_segments
    import numpy as np

    text = "[sighs] Hello"
    segments = parse_tags(text)

    # Verify tags are detected
    assert has_vocalizations(segments)

    # Generate audio for each segment
    generator = VocalizationGenerator(mock_model)
    audio_segments = []

    for seg in segments:
        if seg.type == SegmentType.VOCALIZATION:
            audio, sr = generator.generate(seg, encode_dict)
            audio_segments.append((audio, sr))
        else:
            # Mock speech segment
            speech_audio = np.random.randn(24000).astype(np.float32) * 0.1
            audio_segments.append((speech_audio, 48000))

    # Stitch together
    result, sr = stitch_segments(audio_segments, crossfade_ms=50)
    assert sr == 48000
    assert len(result) > 0
    # Result should be shorter than sum due to crossfade
    assert len(result) < sum(len(a) for a, _ in audio_segments)


def test_integration_whisper_mode_detection():
    """Whisper mode is properly detected and affects speech."""
    from utilities.vocalization.tag_parser import parse_tags
    from utilities.vocalization.vocalization_generator import VocalizationGenerator
    from utilities.vocalization.recipes import get_recipe

    # Parse whisper tag
    segments = parse_tags("[whispers] quiet speech")
    assert segments[0].tag == "whispers"
    assert segments[1].text == "quiet speech"

    # Check generator detects whisper mode
    mock_model = Mock()
    generator = VocalizationGenerator(mock_model)
    assert generator.is_modify_speech("whispers")

    # Verify whisper recipe has modify_speech mode
    recipe = get_recipe("whispers")
    assert recipe["mode"] == "modify_speech"
    assert recipe["tts_text"] is None
    assert len(recipe["effects"]) > 0
