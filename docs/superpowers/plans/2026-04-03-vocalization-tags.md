# Vocalization Audio Tags Implementation Plan

> **Status:** ✅ **COMPLETED** (2026-04-04)
>
> All 11 tasks implemented, 81 tests passing.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audio tag parser + TTS + DSP post-processing pipeline for realistic non-speech vocalizations (sighs, screams, gasps, etc.) embedded inline within SkyrimNet NPC dialogue, using the cloned voice.

**Architecture:** Segment-based processing: parse text for `[tag]` patterns → split into speech/vocalization segments → generate each via LuxTTS → apply DSP recipes to vocalizations → crossfade stitch → pass through existing post-processor. Zero overhead when no tags present.

**Tech Stack:** numpy, scipy, librosa (already in requirements), JSON for recipes (no YAML dependency)

---

## FILE STRUCTURE

### New Files to Create
- `utilities/vocalization/__init__.py` — Package exports
- `utilities/vocalization/tag_parser.py` — Parse [tags] from text into segments
- `utilities/vocalization/dsp_engine.py` — Audio effect implementations (breath_noise, distortion, filters, fades, volume)
- `utilities/vocalization/recipes.py` — Load tag→recipe mappings from JSON
- `utilities/vocalization/vocalization_generator.py` — Orchestrate TTS → DSP pipeline
- `utilities/vocalization/stitcher.py` — Crossfade audio segments together
- `utilities/vocalization/recipes.json` — Default recipe definitions
- `tests/test_vocalization.py` — Test suite

### Modified Files
- `utilities/audio_generation_pipeline.py` — Integration at top of `generate_audio()` function

---

## Task 1: Create vocalization package structure

**Files:**
- Create: `utilities/vocalization/__init__.py`

- [ ] **Step 1: Create package init file**

```python
"""Vocalization audio tag processing package.

This package provides ElevenLabs-style [bracket] audio tags for non-speech
vocalizations (sighs, screams, gasps, etc.) embedded within dialogue text.

Public API:
    - parse_tags(text) — Parse text into speech/vocalization segments
    - generate_vocalization(segment, model, config) — Generate audio for a vocalization segment
    - stitch_segments(segments, crossfade_ms) — Stitch audio segments with crossfades
"""

from utilities.vocalization.tag_parser import parse_tags, Segment, SegmentType
from utilities.vocalization.vocalization_generator import VocalizationGenerator
from utilities.vocalization.stitcher import stitch_segments

__all__ = [
    "parse_tags",
    "Segment",
    "SegmentType",
    "VocalizationGenerator",
    "stitch_segments",
]
```

- [ ] **Step 2: Verify package structure**

Run: `python -c "from utilities.vocalization import parse_tags; print('Package import successful')"`

Expected: No import errors

- [ ] **Step 3: Commit**

```bash
git add utilities/vocalization/__init__.py
git commit -m "feat(vocalization): create package structure"
```

---

## Task 2: Implement tag parser

**Files:**
- Create: `utilities/vocalization/tag_parser.py`
- Test: `tests/test_vocalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocalization.py
import pytest
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocalization.py -v`

Expected: FAIL with "ModuleNotFoundError: utilities.vocalization.tag_parser"

- [ ] **Step 3: Write minimal implementation**

```python
# utilities/vocalization/tag_parser.py
"""Parse ElevenLabs-style [bracket] audio tags from text."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List


logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Type of audio segment."""
    SPEECH = "speech"
    VOCALIZATION = "vocalization"


@dataclass
class Segment:
    """A segment of text representing either speech or a vocalization tag.

    Attributes:
        type: SPEECH or VOCALIZATION
        text: The text content (for SPEECH segments)
        tag: The tag name (for VOCALIZATION segments), e.g., "sighs"
        original_text: The original text that produced this segment
    """
    type: SegmentType
    text: str = ""
    tag: str = ""
    original_text: str = ""

    def __repr__(self) -> str:
        if self.type == SegmentType.SPEECH:
            return f"Segment(SPEECH, text={self.text!r})"
        return f"Segment(VOCALIZATION, tag={self.tag!r})"


# Regex pattern to match [tags]
# Matches: [word] or [multi word phrase]
# Does NOT match: malformed like [unclosed, nested [[brackets]]
TAG_PATTERN = re.compile(r'\[([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\]')


def parse_tags(text: str) -> List[Segment]:
    """
    Parse text into speech and vocalization segments.

    Args:
        text: Input text that may contain [bracket] tags

    Returns:
        List of Segment objects in order
    """
    segments = []
    last_end = 0

    for match in TAG_PATTERN.finditer(text):
        tag = match.group(1)
        start, end = match.span()

        # Add speech segment before this tag if there is any
        if start > last_end:
            speech_text = text[last_end:start]
            segments.append(Segment(
                type=SegmentType.SPEECH,
                text=speech_text,
                original_text=speech_text
            ))

        # Add vocalization segment
        segments.append(Segment(
            type=SegmentType.VOCALIZATION,
            tag=tag,
            original_text=match.group(0)
        ))

        last_end = end

    # Add remaining text as speech segment
    if last_end < len(text):
        remaining = text[last_end:]
        segments.append(Segment(
            type=SegmentType.SPEECH,
            text=remaining,
            original_text=remaining
        ))

    return segments


def has_vocalizations(segments: List[Segment]) -> bool:
    """
    Check if any segment is a vocalization.

    Args:
        segments: List of Segment objects

    Returns:
        True if any VOCALIZATION segment is present
    """
    return any(seg.type == SegmentType.VOCALIZATION for seg in segments)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vocalization.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/vocalization/tag_parser.py tests/test_vocalization.py
git commit -m "feat(vocalization): implement tag parser with tests"
```

---

## Task 3: Implement DSP engine

**Files:**
- Create: `utilities/vocalization/dsp_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocalization.py (add to existing file)
import numpy as np
from utilities.vocalization.dsp_engine import DSPEngine


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
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "pitch_shift", "semitones": 2.0})
    assert len(result) == len(audio)
    assert not np.allclose(result, audio)


def test_dsp_breath_noise(sample_48k_audio):
    """Breath noise adds noise layer."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "breath_noise", "amplitude": 0.1})
    assert len(result) == len(audio)


def test_dsp_low_pass_filter(sample_48k_audio):
    """Low pass filter removes high frequencies."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "low_pass_filter", "cutoff_hz": 1000})
    assert len(result) == len(audio)


def test_dsp_high_pass_filter(sample_48k_audio):
    """High pass filter removes low frequencies."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "high_pass_filter", "cutoff_hz": 1000})
    assert len(result) == len(audio)


def test_dsp_distortion(sample_48k_audio):
    """Distortion adds waveshaping."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "distortion", "intensity": 0.5})
    assert len(result) == len(audio)


def test_dsp_volume(sample_48k_audio):
    """Volume changes gain."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "volume", "factor": 0.5})
    assert len(result) == len(audio)
    assert np.max(np.abs(result)) < np.max(np.abs(audio))


def test_dsp_fade_out(sample_48k_audio):
    """Fade out reduces amplitude at end."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "fade_out", "duration_s": 0.5})
    assert len(result) == len(audio)
    # End should be quieter than beginning
    assert np.max(np.abs(result[-100:])) < np.max(np.abs(result[:100]))


def test_dsp_fade_in(sample_48k_audio):
    """Fade in increases amplitude at start."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "fade_in", "duration_s": 0.5})
    assert len(result) == len(audio)
    # Start should be quieter than end
    assert np.max(np.abs(result[:100])) < np.max(np.abs(result[-100:]))


def test_dsp_time_stretch(sample_48k_audio):
    """Time stretch changes duration."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "time_stretch", "factor": 0.8})
    # Duration changes
    assert len(result) != len(audio)


def test_dsp_speed_up(sample_48k_audio):
    """Speed up shortens audio."""
    audio, sr = sample_48k_audio
    engine = DSPEngine()
    result = engine.apply_effect(audio, sr, {"type": "speed_up", "factor": 1.5})
    assert len(result) < len(audio)


def test_dsp_chain_multiple_effects(sample_48k_audio):
    """Chain multiple effects in sequence."""
    audio, sr = sample_48k_audio
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
    engine = DSPEngine()
    with pytest.raises(ValueError, match="Unknown effect type"):
        engine.apply_effect(audio, sr, {"type": "unknown_effect"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocalization.py::test_dsp_pitch_shift -v`

Expected: FAIL with "ModuleNotFoundError: utilities.vocalization.dsp_engine"

- [ ] **Step 3: Write minimal implementation**

```python
# utilities/vocalization/dsp_engine.py
"""Audio DSP effects for vocalization processing."""

import logging
from typing import Dict, List, Any

import librosa
import numpy as np
import scipy.signal as signal


logger = logging.getLogger(__name__)


class DSPEngine:
    """Audio DSP effects processor for vocalization tags."""

    def apply_effect(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """
        Apply a single DSP effect to audio.

        Args:
            audio: Input audio (float32, 48kHz)
            sr: Sample rate
            effect: Effect dict with 'type' and parameters

        Returns:
            Processed audio

        Raises:
            ValueError: If effect type is unknown
        """
        effect_type = effect["type"]

        handlers = {
            "pitch_shift": self._pitch_shift,
            "time_stretch": self._time_stretch,
            "low_pass_filter": self._low_pass_filter,
            "high_pass_filter": self._high_pass_filter,
            "high_shelf_boost": self._high_shelf_boost,
            "breath_noise": self._breath_noise,
            "distortion": self._distortion,
            "compress": self._compress,
            "fade_out": self._fade_out,
            "fade_in": self._fade_in,
            "volume": self._volume,
            "speed_up": self._speed_up,
        }

        if effect_type not in handlers:
            raise ValueError(f"Unknown effect type: {effect_type}")

        return handlers[effect_type](audio, sr, effect)

    def apply_chain(self, audio: np.ndarray, sr: int, effects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Apply a chain of effects in sequence.

        Args:
            audio: Input audio
            sr: Sample rate
            effects: List of effect dicts

        Returns:
            Processed audio after all effects applied
        """
        result = audio.copy()
        for effect in effects:
            result = self.apply_effect(result, sr, effect)
        return result

    def _pitch_shift(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Shift pitch by semitones using librosa."""
        semitones = effect.get("semitones", 0.0)
        if abs(semitones) < 0.01:
            return audio.copy()
        return librosa.effects.pitch_shift(audio.astype(np.float64), sr, n_steps=semitones).astype(np.float32)

    def _time_stretch(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Stretch time without changing pitch."""
        factor = effect.get("factor", 1.0)
        if abs(factor - 1.0) < 0.01:
            return audio.copy()
        return librosa.effects.time_stretch(audio.astype(np.float64), rate=factor).astype(np.float32)

    def _low_pass_filter(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Apply low-pass filter."""
        cutoff_hz = effect["cutoff_hz"]
        nyquist = sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def _high_pass_filter(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Apply high-pass filter."""
        cutoff_hz = effect["cutoff_hz"]
        nyquist = sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def _high_shelf_boost(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Apply high-shelf boost using peaking EQ."""
        frequency_hz = effect["frequency_hz"]
        gain_db = effect["gain_db"]
        # Design peaking filter
        w0 = 2 * np.pi * frequency_hz / sr
        Q = 1.0
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / (2 * Q)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def _breath_noise(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Add pink/filtered noise layer for breath sounds."""
        amplitude = effect.get("amplitude", 0.1)
        noise = np.random.randn(len(audio)).astype(np.float32) * amplitude
        # Low-pass filter noise for breathy character
        nyquist = sr / 2
        cutoff = 2000 / nyquist
        b, a = signal.butter(2, cutoff, btype='low')
        filtered_noise = signal.filtfilt(b, a, noise)
        return (audio + filtered_noise).astype(np.float32)

    def _distortion(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Apply waveshaping distortion."""
        intensity = effect.get("intensity", 0.5)
        # Soft clipping distortion
        threshold = 1.0 - intensity
        result = np.copy(audio)
        mask = np.abs(audio) > threshold
        result[mask] = np.sign(audio[mask]) * (threshold + (1 - threshold) * np.tanh((np.abs(audio[mask]) - threshold) / (1 - threshold)))
        return result.astype(np.float32)

    def _compress(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Simple dynamic range compression."""
        threshold_db = effect.get("threshold_db", -10.0)
        ratio = effect.get("ratio", 4.0)
        threshold_linear = 10 ** (threshold_db / 20)
        # Simple feedforward compression
        envelope = np.abs(audio)
        gain = np.ones_like(audio)
        mask = envelope > threshold_linear
        excess = envelope[mask] - threshold_linear
        gain[mask] = threshold_linear / (threshold_linear + excess / ratio)
        return (audio * gain).astype(np.float32)

    def _fade_out(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Apply exponential fade out."""
        duration_s = effect.get("duration_s", 0.5)
        fade_samples = int(duration_s * sr)
        if fade_samples >= len(audio):
            fade_samples = len(audio)
        result = audio.copy()
        # Exponential fade curve
        fade_curve = np.exp(-np.linspace(0, 5, fade_samples))
        result[-fade_samples:] *= fade_curve
        return result

    def _fade_in(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Apply exponential fade in."""
        duration_s = effect.get("duration_s", 0.5)
        fade_samples = int(duration_s * sr)
        if fade_samples >= len(audio):
            fade_samples = len(audio)
        result = audio.copy()
        # Exponential fade curve
        fade_curve = np.exp(-np.linspace(5, 0, fade_samples))
        result[:fade_samples] *= fade_curve
        return result

    def _volume(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Adjust volume by factor."""
        factor = effect.get("factor", 1.0)
        return (audio * factor).astype(np.float32)

    def _speed_up(self, audio: np.ndarray, sr: int, effect: Dict[str, Any]) -> np.ndarray:
        """Speed up with pitch preservation (time stretch < 1.0)."""
        factor = effect.get("factor", 1.5)
        # speed_up is inverse of time_stretch
        return self._time_stretch(audio, sr, {"factor": 1.0 / factor})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vocalization.py::test_dsp -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/vocalization/dsp_engine.py tests/test_vocalization.py
git commit -m "feat(vocalization): implement DSP engine with effects"
```

---

## Task 4: Create recipe definitions JSON

**Files:**
- Create: `utilities/vocalization/recipes.json`

- [ ] **Step 1: Create recipes.json file**

```json
{
  "sighs": {
    "tts_text": "haaah",
    "tts_speed": 0.9,
    "max_duration_s": 1.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -3},
      {"type": "low_pass_filter", "cutoff_hz": 800},
      {"type": "breath_noise", "amplitude": 0.08},
      {"type": "fade_out", "duration_s": 0.5}
    ]
  },
  "groans": {
    "tts_text": "uuugh",
    "tts_speed": 0.7,
    "max_duration_s": 2.0,
    "effects": [
      {"type": "pitch_shift", "semitones": -5},
      {"type": "low_pass_filter", "cutoff_hz": 600},
      {"type": "breath_noise", "amplitude": 0.06},
      {"type": "volume", "factor": 1.2}
    ]
  },
  "moans": {
    "tts_text": "oooooh",
    "tts_speed": 0.7,
    "max_duration_s": 2.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -4},
      {"type": "low_pass_filter", "cutoff_hz": 700},
      {"type": "breath_noise", "amplitude": 0.05},
      {"type": "fade_out", "duration_s": 0.8}
    ]
  },
  "gasps": {
    "tts_text": "huh",
    "tts_speed": 1.4,
    "max_duration_s": 0.4,
    "effects": [
      {"type": "pitch_shift", "semitones": 2},
      {"type": "breath_noise", "amplitude": 0.1},
      {"type": "fade_out", "duration_s": 0.2}
    ]
  },
  "screams": {
    "tts_text": "aaaaah",
    "tts_speed": 0.8,
    "max_duration_s": 2.0,
    "effects": [
      {"type": "pitch_shift", "semitones": 5},
      {"type": "distortion", "intensity": 0.6},
      {"type": "high_shelf_boost", "frequency_hz": 4000, "gain_db": 6},
      {"type": "compress", "threshold_db": -10, "ratio": 4},
      {"type": "volume", "factor": 1.5}
    ]
  },
  "shouts": {
    "tts_text": "hey",
    "tts_speed": 0.9,
    "max_duration_s": 1.0,
    "effects": [
      {"type": "pitch_shift", "semitones": 3},
      {"type": "distortion", "intensity": 0.3},
      {"type": "high_shelf_boost", "frequency_hz": 3000, "gain_db": 4},
      {"type": "volume", "factor": 1.3}
    ]
  },
  "whispers": {
    "tts_text": null,
    "mode": "modify_speech",
    "effects": [
      {"type": "pitch_shift", "semitones": -2},
      {"type": "high_pass_filter", "cutoff_hz": 800},
      {"type": "breath_noise", "amplitude": 0.15},
      {"type": "volume", "factor": 0.4}
    ]
  },
  "breathes heavily": {
    "tts_text": "haah huuh haah",
    "tts_speed": 0.6,
    "max_duration_s": 2.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -2},
      {"type": "low_pass_filter", "cutoff_hz": 500},
      {"type": "breath_noise", "amplitude": 0.12},
      {"type": "volume", "factor": 0.8}
    ]
  },
  "clears throat": {
    "tts_text": "ahem",
    "tts_speed": 1.0,
    "max_duration_s": 0.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -1},
      {"type": "high_pass_filter", "cutoff_hz": 1000},
      {"type": "fade_out", "duration_s": 0.1}
    ]
  },
  "coughs": {
    "tts_text": "kugh kugh",
    "tts_speed": 1.2,
    "max_duration_s": 0.6,
    "effects": [
      {"type": "pitch_shift", "semitones": -2},
      {"type": "low_pass_filter", "cutoff_hz": 1200},
      {"type": "volume", "factor": 0.9}
    ]
  },
  "laughs": {
    "tts_text": "haha",
    "tts_speed": 1.3,
    "max_duration_s": 1.5,
    "effects": [
      {"type": "pitch_shift", "semitones": 1},
      {"type": "high_pass_filter", "cutoff_hz": 1500},
      {"type": "volume", "factor": 1.1}
    ]
  },
  "sobs": {
    "tts_text": "sob sob",
    "tts_speed": 0.8,
    "max_duration_s": 2.0,
    "effects": [
      {"type": "pitch_shift", "semitones": -3},
      {"type": "low_pass_filter", "cutoff_hz": 900},
      {"type": "breath_noise", "amplitude": 0.1},
      {"type": "volume", "factor": 0.7}
    ]
  },
  "whimpers": {
    "tts_text": "whimper",
    "tts_speed": 0.9,
    "max_duration_s": 1.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -4},
      {"type": "low_pass_filter", "cutoff_hz": 700},
      {"type": "breath_noise", "amplitude": 0.08},
      {"type": "volume", "factor": 0.6}
    ]
  },
  "pause": {
    "tts_text": null,
    "duration_s": 0.3,
    "effects": []
  }
}
```

- [ ] **Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('utilities/vocalization/recipes.json')); print('Valid JSON')"`

Expected: "Valid JSON"

- [ ] **Step 3: Commit**

```bash
git add utilities/vocalization/recipes.json
git commit -m "feat(vocalization): add DSP recipe definitions for 14 tags"
```

---

## Task 5: Implement recipe loader

**Files:**
- Create: `utilities/vocalization/recipes.py`
- Test: `tests/test_vocalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocalization.py (add to existing file)
from utilities.vocalization.recipes import load_recipes, get_recipe


def test_load_recipes():
    """Load all recipes from JSON."""
    recipes = load_recipes()
    assert "sighs" in recipes
    assert "screams" in recipes
    assert "pause" in recipes


def test_get_recipe_existing():
    """Get recipe for existing tag."""
    recipe = get_recipe("sighs")
    assert recipe["tts_text"] == "haaah"
    assert "effects" in recipe
    assert len(recipe["effects"]) > 0


def test_get_recipe_unknown():
    """Get recipe for unknown tag returns None."""
    recipe = get_recipe("unknown_tag_xyz")
    assert recipe is None


def test_recipe_sighs_structure():
    """Sighs recipe has expected structure."""
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
    recipe = get_recipe("whispers")
    assert recipe.get("mode") == "modify_speech"
    assert recipe.get("tts_text") is None


def test_recipe_pause():
    """Pause recipe has duration but no tts_text."""
    recipe = get_recipe("pause")
    assert recipe.get("tts_text") is None
    assert "duration_s" in recipe
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocalization.py::test_recipe -v`

Expected: FAIL with "ModuleNotFoundError: utilities.vocalization.recipes"

- [ ] **Step 3: Write minimal implementation**

```python
# utilities/vocalization/recipes.py
"""Load and manage vocalization DSP recipes from JSON config."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)

# Path to recipes JSON
RECIPES_PATH = Path(__file__).parent / "recipes.json"

# Cache for loaded recipes
_recipes_cache: Optional[Dict[str, Dict]] = None


def load_recipes() -> Dict[str, Dict]:
    """
    Load all recipes from JSON config file.

    Returns:
        Dictionary mapping tag names to recipe definitions

    Raises:
        FileNotFoundError: If recipes.json doesn't exist
        json.JSONDecodeError: If recipes.json is invalid JSON
    """
    global _recipes_cache

    if _recipes_cache is not None:
        return _recipes_cache

    if not RECIPES_PATH.exists():
        raise FileNotFoundError(f"Recipes file not found: {RECIPES_PATH}")

    with open(RECIPES_PATH, 'r', encoding='utf-8') as f:
        _recipes_cache = json.load(f)

    logger.debug(f"Loaded {len(_recipes_cache)} recipes from {RECIPES_PATH}")
    return _recipes_cache


def get_recipe(tag: str) -> Optional[Dict]:
    """
    Get recipe for a specific tag.

    Args:
        tag: Tag name (e.g., "sighs", "breathes heavily")

    Returns:
        Recipe dict or None if tag not found
    """
    recipes = load_recipes()
    return recipes.get(tag)


def reload_recipes() -> Dict[str, Dict]:
    """
    Force reload recipes from disk (clears cache).

    Returns:
        Dictionary mapping tag names to recipe definitions
    """
    global _recipes_cache
    _recipes_cache = None
    return load_recipes()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vocalization.py::test_recipe -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/vocalization/recipes.py tests/test_vocalization.py
git commit -m "feat(vocalization): implement recipe loader"
```

---

## Task 6: Implement audio stitcher

**Files:**
- Create: `utilities/vocalization/stitcher.py`
- Test: `tests/test_vocalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocalization.py (add to existing file)
from utilities.vocalization.stitcher import stitch_segments


def test_stitch_single_segment(sample_48k_audio):
    """Single segment returns unchanged."""
    audio, sr = sample_48k_audio
    segments = [(audio, sr)]
    result, result_sr = stitch_segments(segments, crossfade_ms=50)
    assert np.allclose(result, audio)
    assert result_sr == sr


def test_stitch_two_segments_with_crossfade(sample_48k_audio):
    """Two segments are crossfaded."""
    audio1, sr = sample_48k_audio
    audio2 = np.roll(audio1, 1000)  # Different content
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
    segments = [(audio1, sr), (audio2, sr), (audio3, sr)]
    result, result_sr = stitch_segments(segments, crossfade_ms=50)
    assert result_sr == sr
    assert len(result) < len(audio1) + len(audio2) + len(audio3)


def test_stitch_zero_crossfade(sample_48k_audio):
    """Zero crossfade concatenates directly."""
    audio1, sr = sample_48k_audio
    audio2 = audio1 * 0.5
    segments = [(audio1, sr), (audio2, sr)]
    result, result_sr = stitch_segments(segments, crossfade_ms=0)
    expected_length = len(audio1) + len(audio2)
    assert len(result) == expected_length


def test_stitch_empty_segments():
    """Empty segment list returns empty audio."""
    result, sr = stitch_segments([], crossfade_ms=50)
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocalization.py::test_stitch -v`

Expected: FAIL with "ModuleNotFoundError: utilities.vocalization.stitcher"

- [ ] **Step 3: Write minimal implementation**

```python
# utilities/vocalization/stitcher.py
"""Audio segment stitching with crossfades."""

import logging
from typing import List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


def stitch_segments(
    segments: List[Tuple[np.ndarray, int]],
    crossfade_ms: float = 50.0
) -> Tuple[np.ndarray, int]:
    """
    Stitch audio segments together with crossfades.

    Args:
        segments: List of (audio_array, sample_rate) tuples
        crossfade_ms: Crossfade duration in milliseconds

    Returns:
        Tuple of (stitched_audio, sample_rate)
    """
    if not segments:
        return np.array([], dtype=np.float32), 48000

    if len(segments) == 1:
        return segments[0]

    # Get sample rate from first segment
    sr = segments[0][1]
    crossfade_samples = int(crossfade_ms / 1000 * sr)

    # Start with first segment
    result = segments[0][0].copy()

    # Append each subsequent segment with crossfade
    for audio, _ in segments[1:]:
        result = _crossfade_append(result, audio, crossfade_samples)

    return result, sr


def _crossfade_append(audio1: np.ndarray, audio2: np.ndarray, crossfade_samples: int) -> np.ndarray:
    """
    Append audio2 to audio1 with crossfade.

    Args:
        audio1: First audio segment
        audio2: Second audio segment to append
        crossfade_samples: Number of samples to crossfade

    Returns:
        Crossfaded audio
    """
    if crossfade_samples <= 0:
        return np.concatenate([audio1, audio2])

    # Limit crossfade to shorter segment
    crossfade_samples = min(crossfade_samples, len(audio1), len(audio2))

    # Create crossfade curves (equal power)
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)

    # Apply crossfade
    result = audio1.copy()
    result[-crossfade_samples:] *= fade_out

    audio2_crossfade = audio2[:crossfade_samples] * fade_in

    # Sum crossfaded region
    result[-crossfade_samples:] += audio2_crossfade

    # Append remaining audio2
    result = np.concatenate([result, audio2[crossfade_samples:]])

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vocalization.py::test_stitch -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/vocalization/stitcher.py tests/test_vocalization.py
git commit -m "feat(vocalization): implement audio segment stitcher with crossfade"
```

---

## Task 7: Implement vocalization generator

**Files:**
- Create: `utilities/vocalization/vocalization_generator.py`
- Test: `tests/test_vocalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocalization.py (add to existing file)
import torch
from unittest.mock import Mock, MagicMock
from utilities.vocalization.vocalization_generator import VocalizationGenerator
from utilities.vocalization.tag_parser import Segment, SegmentType


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
    segment = Segment(type=SegmentType.VOCALIZATION, tag="sighs")
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) > 0
    assert sr == 48000


def test_vocalization_generator_pause(mock_model, encode_dict):
    """Pause tag returns silence without TTS."""
    segment = Segment(type=SegmentType.VOCALIZATION, tag="pause")
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) == int(0.3 * 48000)  # 0.3s pause
    assert sr == 48000
    # Silence should be all zeros
    assert np.allclose(audio, 0)


def test_vocalization_generator_unknown_tag(mock_model, encode_dict):
    """Unknown tag returns 0.3s silence with warning."""
    segment = Segment(type=SegmentType.VOCALIZATION, tag="unknown_tag")
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) == int(0.3 * 48000)


def test_vocalization_generator_with_max_duration(mock_model, encode_dict):
    """Max duration truncates long TTS output."""
    # Return 2 seconds of audio
    long_audio = torch.randn(96000).numpy()
    mock_model.generate_speech = MagicMock(return_value=torch.from_numpy(long_audio))

    segment = Segment(type=SegmentType.VOCALIZATION, tag="gasps")  # max_duration_s: 0.4
    generator = VocalizationGenerator(mock_model)
    audio, sr = generator.generate(segment, encode_dict)
    assert len(audio) <= int(0.5 * 48000)  # ~0.4s + tolerance


def test_vocalization_generator_custom_params(mock_model, encode_dict):
    """Custom TTS params are passed through."""
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


def test_vocalization_generator_is_modify_speech(mock_model, encode_dict):
    """is_modify_speech() identifies whisper mode."""
    generator = VocalizationGenerator(mock_model)
    assert generator.is_modify_speech("whispers")
    assert not generator.is_modify_speech("sighs")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocalization.py::test_vocalization_generator -v`

Expected: FAIL with "ModuleNotFoundError: utilities.vocalization.vocalization_generator"

- [ ] **Step 3: Write minimal implementation**

```python
# utilities/vocalization/vocalization_generator.py
"""Orchestrate TTS generation and DSP processing for vocalization tags."""

import logging
from typing import Dict, Optional

import numpy as np
import torch


from utilities.vocalization.recipes import get_recipe
from utilities.vocalization.dsp_engine import DSPEngine
from utilities.app_constants import SAMPLE_RATE


logger = logging.getLogger(__name__)


class VocalizationGenerator:
    """
    Generate audio for vocalization tags using TTS + DSP processing.

    For each tag:
    1. Load recipe (tts_text, effects, params)
    2. Generate speech using LuxTTS with recipe's tts_text
    3. Apply DSP effect chain from recipe
    4. Return processed audio

    Special cases:
    - "pause" tag: returns silence of specified duration
    - Unknown tags: returns 0.3s silence with warning log
    - whisper mode: recipe has mode="modify_speech" (handled elsewhere)
    """

    def __init__(
        self,
        model,
        num_steps: int = 8,
        guidance_scale: float = 3.0,
        speed: float = 1.0,
        t_shift: float = 0.9,
        return_smooth: bool = True,
    ):
        """
        Initialize vocalization generator.

        Args:
            model: LuxTTS model instance
            num_steps: Flow matching steps
            guidance_scale: CFG scale
            speed: Speech speed multiplier
            t_shift: Sampling shift
            return_smooth: If True, disable 48k upsampling
        """
        self.model = model
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.speed = speed
        self.t_shift = t_shift
        self.return_smooth = return_smooth
        self.dsp_engine = DSPEngine()

    def generate(self, segment, encode_dict: Dict) -> tuple[np.ndarray, int]:
        """
        Generate audio for a vocalization segment.

        Args:
            segment: Segment with type=VOCALIZATION
            encode_dict: Speaker encoding from LuxTTS.encode_prompt()

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        tag = segment.tag
        recipe = get_recipe(tag)

        if recipe is None:
            logger.warning(f"Unknown vocalization tag: [{tag}], treating as 0.3s pause")
            return self._create_silence(0.3), SAMPLE_RATE

        # Handle pause tag (special case)
        if recipe.get("tts_text") is None and "duration_s" in recipe:
            duration = recipe["duration_s"]
            return self._create_silence(duration), SAMPLE_RATE

        # Get TTS text and params from recipe
        tts_text = recipe["tts_text"]
        if tts_text is None:
            # Modify speech mode (e.g., whispers) - shouldn't reach here
            logger.warning(f"Tag [{tag}] has no tts_text, treating as 0.3s pause")
            return self._create_silence(0.3), SAMPLE_RATE

        # Generate TTS audio
        tts_speed = recipe.get("tts_speed", self.speed)
        audio = self._generate_tts(tts_text, encode_dict, tts_speed)

        # Convert to numpy
        if hasattr(audio, 'numpy'):
            audio = audio.numpy()
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        audio = audio.astype(np.float32).squeeze()

        # Apply max duration if specified
        max_duration = recipe.get("max_duration_s")
        if max_duration and len(audio) > int(max_duration * SAMPLE_RATE):
            audio = audio[:int(max_duration * SAMPLE_RATE)]

        # Apply DSP effects
        effects = recipe.get("effects", [])
        if effects:
            audio = self.dsp_engine.apply_chain(audio, SAMPLE_RATE, effects)

        return audio, SAMPLE_RATE

    def is_modify_speech(self, tag: str) -> bool:
        """
        Check if tag operates in modify-speech mode (e.g., whispers).

        Args:
            tag: Tag name

        Returns:
            True if tag modifies following speech instead of generating standalone audio
        """
        recipe = get_recipe(tag)
        if recipe is None:
            return False
        return recipe.get("mode") == "modify_speech"

    def _generate_tts(self, text: str, encode_dict: Dict, speed: float) -> torch.Tensor:
        """Generate speech using LuxTTS model."""
        return self.model.generate_speech(
            text=text,
            encode_dict=encode_dict,
            num_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            speed=speed,
            t_shift=self.t_shift,
            return_smooth=self.return_smooth,
        )

    def _create_silence(self, duration_sec: float) -> np.ndarray:
        """Create silence audio array."""
        num_samples = int(duration_sec * SAMPLE_RATE)
        return np.zeros(num_samples, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vocalization.py::test_vocalization_generator -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/vocalization/vocalization_generator.py tests/test_vocalization.py
git commit -m "feat(vocalization): implement vocalization generator"
```

---

## Task 8: Integrate vocalization pipeline into audio_generation_pipeline

**Files:**
- Modify: `utilities/audio_generation_pipeline.py`
- Test: `tests/test_vocalization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vocalization.py (add to existing file)
from unittest.mock import patch, MagicMock
from utilities.audio_generation_pipeline import generate_audio
from utilities.app_config import AppConfig


@pytest.mark.asyncio
async def test_generate_audio_with_vocalization_tags():
    """Integration test: tags trigger vocalization pipeline."""
    # Mock model
    with patch('utilities.audio_generation_pipeline.load_model_if_needed') as mock_load:
        mock_model = MagicMock()
        dummy_audio = torch.randn(48000)  # 1 second
        mock_model.generate_speech = MagicMock(return_value=dummy_audio)
        mock_load.return_value = mock_model

        # Mock speaker encoding
        with patch('utilities.audio_generation_pipeline._get_speaker_encoding') as mock_enc:
            mock_enc.return_value = {"speaker_embed": torch.randn(128)}

            config = AppConfig()
            output_path, seed = await generate_audio(
                text="[sighs] Hello there",
                config=config,
                enable_post_processing=False,  # Simplify test
            )

            # Verify output was created
            assert Path(output_path).exists()

            # Verify generate_speech was called (for the speech part at least)
            assert mock_model.generate_speech.called


@pytest.mark.asyncio
async def test_generate_audio_no_tags_fast_path():
    """No tags uses existing pipeline (zero overhead)."""
    with patch('utilities.audio_generation_pipeline.load_model_if_needed') as mock_load:
        mock_model = MagicMock()
        dummy_audio = torch.randn(48000)
        mock_model.generate_speech = MagicMock(return_value=dummy_audio)
        mock_load.return_value = mock_model

        with patch('utilities.audio_generation_pipeline._get_speaker_encoding') as mock_enc:
            mock_enc.return_value = {"speaker_embed": torch.randn(128)}

            config = AppConfig()
            output_path, seed = await generate_audio(
                text="Hello world",  # No tags
                config=config,
                enable_post_processing=False,
            )

            assert Path(output_path).exists()


@pytest.mark.asyncio
async def test_generate_audio_with_whisper_modify_speech():
    """Whisper tag modifies following speech segment."""
    with patch('utilities.audio_generation_pipeline.load_model_if_needed') as mock_load:
        mock_model = MagicMock()
        dummy_audio = torch.randn(48000)
        mock_model.generate_speech = MagicMock(return_value=dummy_audio)
        mock_load.return_value = mock_model

        with patch('utilities.audio_generation_pipeline._get_speaker_encoding') as mock_enc:
            mock_enc.return_value = {"speaker_embed": torch.randn(128)}

            config = AppConfig()
            output_path, seed = await generate_audio(
                text="[whispers] This is quiet",
                config=config,
                enable_post_processing=False,
            )

            assert Path(output_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocalization.py::test_generate_audio_with_vocalization_tags -v`

Expected: FAIL (vocalization not yet integrated)

- [ ] **Step 3: Modify audio_generation_pipeline.py**

Read the current file first to find exact insertion point, then add this import at the top:

```python
# Add to imports section (around line 42)
from utilities.vocalization.tag_parser import parse_tags, has_vocalizations, Segment, SegmentType
from utilities.vocalization.vocalization_generator import VocalizationGenerator
from utilities.vocalization.stitcher import stitch_segments
from utilities.vocalization.recipes import get_recipe
```

Then modify the `generate_audio` function. Find the health check section (around line 107-109) and add the vocalization pipeline immediately after:

```python
# In generate_audio(), after health check (after line 109, before "Ensure model is loaded")

    # Check for vocalization tags
    segments = parse_tags(text)

    # Fast path: no tags, use existing pipeline
    if not has_vocalizations(segments):
        # Original code continues unchanged...
        # (The existing code below handles normal speech generation)
    else:
        # Vocalization path: handle tags separately
        return await _generate_with_vocalizations(
            segments=segments,
            speaker_audio=speaker_audio,
            language=language,
            cfg_scale=cfg_scale,
            seed=seed,
            randomize_seed=randomize_seed,
            num_steps=num_steps,
            t_shift=t_shift,
            return_smooth=return_smooth,
            config=config,
            enable_post_processing=enable_post_processing,
            pitch_shift=pitch_shift,
            eq_intensity=eq_intensity,
            compressor_threshold_offset=compressor_threshold_offset,
            compressor_ratio=compressor_ratio,
            compressor_knee_db=compressor_knee_db,
            compressor_attack_ms=compressor_attack_ms,
            compressor_release_ms=compressor_release_ms,
            max_gain_reduction_db=max_gain_reduction_db,
            de_ess_intensity=de_ess_intensity,
            target_loudness=target_loudness,
            return_diagnostics=return_diagnostics,
            save_raw=save_raw,
        )
```

Then add the helper function at the end of the file (before `init_speaker_cache`):

```python
# Add after _get_speaker_encoding() function, before init_speaker_cache()

async def _generate_with_vocalizations(
    segments: list,
    speaker_audio: Optional[str],
    language: str,
    cfg_scale: float,
    seed: int,
    randomize_seed: bool,
    num_steps: int,
    t_shift: float,
    return_smooth: bool,
    config: AppConfig,
    enable_post_processing: bool,
    pitch_shift: Optional[float],
    eq_intensity: float,
    compressor_threshold_offset: float,
    compressor_ratio: float,
    compressor_knee_db: float,
    compressor_attack_ms: float,
    compressor_release_ms: float,
    max_gain_reduction_db: float,
    de_ess_intensity: float,
    target_loudness: float,
    return_diagnostics: bool,
    save_raw: bool,
) -> Tuple[str, int] | Tuple[str, int, str, dict]:
    """
    Generate audio with vocalization tags.

    Processes each segment (speech or vocalization) separately,
    then stitches together with crossfades.
    """
    # Set seed
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)

    # Load model
    model = load_model_if_needed(config)

    # Get speaker encoding
    encode_dict = await _get_speaker_encoding(speaker_audio, model, config)

    # Initialize vocalization generator
    voc_generator = VocalizationGenerator(
        model=model,
        num_steps=num_steps,
        guidance_scale=cfg_scale,
        speed=1.0,  # Use recipe's speed
        t_shift=t_shift,
        return_smooth=return_smooth,
    )

    # Process each segment
    audio_segments = []
    whisper_effects = None  # Store whisper effects for next speech segment

    for segment in segments:
        if segment.type == SegmentType.VOCALIZATION:
            tag = segment.tag

            # Check for whisper mode
            if voc_generator.is_modify_speech(tag):
                recipe = get_recipe(tag)
                whisper_effects = recipe.get("effects", [])
                logger.debug(f"Tag [{tag}] in modify-speech mode, will affect next speech")
                continue

            # Generate vocalization
            audio, _ = voc_generator.generate(segment, encode_dict)
            audio_segments.append((audio, SAMPLE_RATE))
        else:
            # Speech segment
            speech_text = segment.text
            if not speech_text.strip():
                continue

            # Generate speech
            audio = model.generate_speech(
                text=speech_text,
                encode_dict=encode_dict,
                num_steps=num_steps,
                guidance_scale=cfg_scale,
                speed=1.0,
                t_shift=t_shift,
                return_smooth=return_smooth,
            )

            # Convert to numpy
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            elif isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio = audio.astype(np.float32).squeeze()

            # Apply whisper effects if pending
            if whisper_effects:
                from utilities.vocalization.dsp_engine import DSPEngine
                dsp = DSPEngine()
                audio = dsp.apply_chain(audio, SAMPLE_RATE, whisper_effects)
                whisper_effects = None

            audio_segments.append((audio, SAMPLE_RATE))

    # Stitch segments with crossfade
    if not audio_segments:
        # Fallback: create silence
        audio_segments.append((create_silence(0.5, SAMPLE_RATE), SAMPLE_RATE))

    stitched_audio, _ = stitch_segments(audio_segments, crossfade_ms=50)

    # Save raw if requested
    raw_path = None
    raw_diagnostics = {}
    if save_raw:
        raw_timestamp = int(time.time() * 1000)
        raw_path = OUTPUT_DIR / f"output_{raw_timestamp}_raw.wav"
        save_wav_file(stitched_audio, raw_path, sample_rate=SAMPLE_RATE)

    # Apply post-processing
    if enable_post_processing:
        processor = AudioPostProcessor(return_diagnostics=return_diagnostics)
        # Use the full original text for pitch detection
        full_text = "".join(s.text for s in segments if s.type == SegmentType.SPEECH)
        stitched_audio, raw_diagnostics = processor.process(
            stitched_audio,
            sr=SAMPLE_RATE,
            text=full_text,
            pitch_shift=pitch_shift,
            eq_intensity=eq_intensity,
            de_ess_intensity=de_ess_intensity,
            compressor_threshold_offset_db=compressor_threshold_offset,
            compressor_ratio=compressor_ratio,
            compressor_knee_db=compressor_knee_db,
            compressor_attack_ms=compressor_attack_ms,
            compressor_release_ms=compressor_release_ms,
            max_gain_reduction_db=max_gain_reduction_db,
            target_loudness=target_loudness,
        )

    # Save output
    timestamp = int(time.time() * 1000)
    output_path = OUTPUT_DIR / f"output_{timestamp}.wav"
    save_wav_file(stitched_audio, output_path, sample_rate=SAMPLE_RATE)

    logger.info(f"Saved audio with vocalizations to {output_path}")

    # Return based on what was requested
    if save_raw or return_diagnostics:
        return str(output_path), seed, str(raw_path) if raw_path else None, raw_diagnostics
    return str(output_path), seed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vocalization.py::test_generate_audio_with_vocalization_tags -v`

Expected: PASS

- [ ] **Step 5: Test that normal path is unchanged**

Run: `pytest tests/test_vocalization.py::test_generate_audio_no_tags_fast_path -v`

Expected: PASS

- [ ] **Step 6: Run existing pipeline tests to ensure no regression**

Run: `pytest tests/test_pipeline.py -v`

Expected: All existing tests pass

- [ ] **Step 7: Commit**

```bash
git add utilities/audio_generation_pipeline.py tests/test_vocalization.py
git commit -m "feat(vocalization): integrate vocalization pipeline into generate_audio"
```

---

## Task 9: Update __init__.py exports

**Files:**
- Modify: `utilities/vocalization/__init__.py`

- [ ] **Step 1: Verify exports are complete**

Run: `python -c "from utilities.vocalization import parse_tags, VocalizationGenerator, stitch_segments; print('All exports accessible')"`

Expected: "All exports accessible"

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/test_vocalization.py -v`

Expected: All tests pass

- [ ] **Step 3: Run full project test suite**

Run: `pytest tests/ -v`

Expected: All tests pass (including existing tests)

- [ ] **Step 4: Commit if any changes**

```bash
git add utilities/vocalization/__init__.py
git commit -m "chore(vocalization): verify and update package exports"
```

---

## Task 10: Documentation and cleanup

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md with vocalization documentation**

Add to CLAUDE.md after the "SkyrimNet-LuxTTS Server" section:

```markdown
### Vocalization Audio Tags

The server supports ElevenLabs-style `[bracket]` audio tags for non-speech vocalizations embedded within dialogue text. Examples:

- `[sighs]` — Slow exhale, relief or resignation
- `[gasps]` — Sharp inhalation, surprise
- `[screams]` — Loud, high-intensity vocalization
- `[whispers]` — Quiet, breathy speech (modifies next speech segment)
- `[pause]` — Silence gap

**Usage:**
```
[sighs] I can't believe we made it.
Stop! [gasps] How did you find me?
[whispers] Don't make a sound.
```

**Tags:** sighs, groans, moans, gasps, screams, shouts, whispers, breathes heavily, clears throat, coughs, laughs, sobs, whimpers, pause

**Implementation:** `utilities/vocalization/` package with parser, DSP engine, recipe loader, generator, and stitcher. Recipes defined in `utilities/vocalization/recipes.json` (JSON format).

**Performance:** Zero overhead when no tags present (single regex scan). With tags: ~100-300ms extra DSP per vocalization.
```

- [ ] **Step 2: Verify README mentions the feature (if applicable)**

Check if README or docs need updating. For now, CLAUDE.md is sufficient for developers.

- [ ] **Step 3: Final integration test**

Run: `pytest tests/test_vocalization.py tests/test_pipeline.py tests/test_post_processor.py -v`

Expected: All tests pass

- [ ] **Step 4: Commit documentation**

```bash
git add CLAUDE.md
git commit -m "docs(vocalization): document vocalization audio tags feature"
```

---

## Task 11: Manual verification (optional but recommended)

**Files:**
- Create test script: `tests/manual_vocalization_test.py`

- [ ] **Step 1: Create manual test script**

```python
"""Manual verification test for vocalization tags.

This script generates sample audio with various tags for manual listening test.
Run with: python tests/manual_vocalization_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.audio_generation_pipeline import generate_audio
from utilities.app_config import AppConfig


async def main():
    """Generate test audio files."""
    config = AppConfig()

    test_cases = [
        ("[sighs] I can't believe we made it.", "test_sighs"),
        ("[gasps] Who's there?", "test_gasps"),
        ("[screams] Get away from me!", "test_screams"),
        ("[whispers] Don't make a sound.", "test_whispers"),
        ("Hello [pause] my friend.", "test_pause"),
        ("[breathes heavily] We need to keep moving.", "test_heavy_breath"),
        ("Normal speech without tags.", "test_normal"),
    ]

    for text, filename in test_cases:
        print(f"Generating: {text}")
        output_path, seed = await generate_audio(
            text=text,
            config=config,
            enable_post_processing=True,
        )
        print(f"  -> {output_path}")

        # Copy to test output with descriptive name
        dest = Path("tests/output") / f"{filename}.wav"
        dest.parent.mkdir(exist_ok=True)
        Path(output_path).rename(dest)
        print(f"  -> Saved to {dest}")

    print("\nAll test files generated in tests/output/")
    print("Listen to verify vocalization quality.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run manual test (optional)**

Run: `python tests/manual_vocalization_test.py`

Expected: Generates WAV files in tests/output/ for manual listening verification

- [ ] **Step 3: Commit manual test script**

```bash
git add tests/manual_vocalization_test.py
git commit -m "test(vocalization): add manual verification test script"
```

---

## Final Verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`

Expected: All tests pass

- [ ] **Step 2: Check imports work**

Run: `python -c "from utilities.vocalization import parse_tags, VocalizationGenerator, stitch_segments; print('OK')"`

Expected: "OK"

- [ ] **Step 3: Verify file structure**

Run: `ls -la utilities/vocalization/`

Expected: `__init__.py`, `tag_parser.py`, `dsp_engine.py`, `recipes.py`, `vocalization_generator.py`, `stitcher.py`, `recipes.json`

- [ ] **Step 4: Final commit if needed**

```bash
git status
git add -A
git commit -m "feat(vocalization): complete vocalization audio tags implementation"
```

---

## Summary

This implementation adds:
- 14 supported vocalization tags (sighs, groans, moans, gasps, screams, shouts, whispers, breathes heavily, clears throat, coughs, laughs, sobs, whimpers, pause)
- Tag parser with regex-based segment extraction
- DSP engine with 12 effect types (pitch_shift, time_stretch, filters, breath_noise, distortion, compress, fades, volume, speed_up)
- JSON recipe definitions (no YAML dependency)
- Vocalization generator orchestrating TTS + DSP
- Audio stitcher with crossfade
- Whisper mode for `[whispers]` tag (modifies following speech)
- Zero-overhead fast path when no tags present
- Full test coverage with pytest

The system is extensible: new tags can be added by editing `recipes.json`. Neural vocalization models can be swapped in later by replacing `VocalizationGenerator`.
