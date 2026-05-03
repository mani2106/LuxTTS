# Vocalization Audio Tags Design Spec

**Date**: 2026-04-03
**Status**: Draft
**Context**: SkyrimNet game NPC dialogue needs realistic non-speech vocalizations (sighs, screams, gasps, etc.) embedded inline within speech, in the cloned voice.

## Problem

LuxTTS is a speech-only TTS model. When dialogue contains interjections like "Ahhh!", "Oh!", "Shhh!", the model spells them out as regular words rather than producing realistic vocalizations. Game NPCs need sounds like sighs, gasps, screams, whispers, groans — all in the cloned voice.

## Approach

**Layered architecture**:
- **Layer 1 (build now)**: Audio tag parser + TTS + DSP post-processing pipeline
- **Layer 2 (future)**: Swap in a neural vocalization model (e.g., NVSpeech) when available

The audio tag parser and pipeline infrastructure are the same regardless of the vocalization engine behind it. The DSP engine is a pluggable component.

## Audio Tag Syntax

ElevenLabs-style `[bracket]` tags inline in text. LLM-friendly — an LLM generating dialogue can naturally include these tags.

### Examples

```
[sighs] I can't believe we made it.
Stop right there! [gasps] How did you find me?
[groans] My head... where am I?
[whispers] Don't make a sound.
[breathes heavily] We need to keep moving.
[screams] Get away from me!
[gasps] [screams] Help!
```

### Parser Rules

- Tags are `[word]` or `[multi-word phrase]`
- Tags are standalone — they represent non-verbal sounds, not text to be spoken
- Tags can appear at start, middle, or end of dialogue
- Multiple consecutive tags are processed sequentially
- Unknown tags are treated as short pauses (0.3s silence) with a warning log
- Malformed or nested tags are skipped, treated as regular text
- Tags are stripped from text before TTS processing

### Supported Tags (Phase 1)

| Tag | Category | Description |
|-----|----------|-------------|
| `[sighs]` | Human reactions | Slow exhale, relief or resignation |
| `[groans]` | Human reactions | Low, prolonged vocalization |
| `[moans]` | Human reactions | Extended low vocalization |
| `[gasps]` | Emotional | Sharp inhalation, surprise |
| `[screams]` | Emotional | Loud, high-intensity vocalization |
| `[shouts]` | Emotional | Forceful, raised voice |
| `[whispers]` | Delivery | Quiet, breathy speech |
| `[breathes heavily]` | Breath/air | Labored breathing |
| `[clears throat]` | Breath/air | Throat clearing sound |
| `[coughs]` | Breath/air | Cough sound |
| `[laughs]` | Emotional | Laughter |
| `[sobs]` | Emotional | Crying/sobbing |
| `[whimpers]` | Emotional | Quiet crying |
| `[pause]` | Delivery | Silence gap |

## Processing Pipeline

### Flow

```
Input:  "[sighs] I can't do this anymore. [gasps] Who's there?"

Step 1 — Parse into segments:
  [Vocalization: "sighs"]
  [Speech: "I can't do this anymore."]
  [Vocalization: "gasps"]
  [Speech: "Who's there?"]

Step 2 — Generate each segment:
  [sighs] → TTS("haaah") → DSP(sigh_recipe) → sigh_audio
  [speech] → TTS("I can't do this anymore.") → speech_audio_1
  [gasps] → TTS("huh") → DSP(gasp_recipe) → gasp_audio
  [speech] → TTS("Who's there?") → speech_audio_2

Step 3 — Crossfade stitch:
  sigh_audio + crossfade + speech_audio_1 + crossfade + gasp_audio + crossfade + speech_audio_2

Output: Final audio with embedded vocalizations → existing post-processor
```

### DSP Recipe System

Each tag maps to a "recipe" — a chain of audio effects with tuned parameters. Recipes are defined in a config file (`utilities/vocalization/recipes.yaml`) for easy tuning without code changes.

Each recipe specifies:
- **tts_text**: The text to feed TTS for this vocalization (e.g., "haaah" for sighs)
- **effects**: Ordered list of DSP effects with parameters

Example recipe (YAML):

```yaml
sighs:
  tts_text: "haaah"
  effects:
    - type: pitch_shift
      semitones: -3
    - type: low_pass_filter
      cutoff_hz: 800
    - type: breath_noise
      amplitude: 0.08
    - type: fade_out
      duration_s: 0.5

screams:
  tts_text: "aaaaah"
  tts_speed: 0.8
  effects:
    - type: pitch_shift
      semitones: 5
    - type: distortion
      intensity: 0.6
    - type: high_shelf_boost
      frequency_hz: 4000
      gain_db: 6
    - type: compress
      threshold_db: -10
      ratio: 4

gasps:
  tts_text: "huh"
  tts_speed: 1.4
  max_duration_s: 0.4
  effects:
    - type: pitch_shift
      semitones: 2
    - type: breath_noise
      amplitude: 0.1

whispers:
  tts_text: null  # Uses the surrounding speech text instead
  mode: modify_speech
  effects:
    - type: pitch_shift
      semitones: -2
    - type: high_pass_filter
      cutoff_hz: 800
    - type: breath_noise
      amplitude: 0.15
    - type: volume
      factor: 0.4

pause:
  tts_text: null
  duration_s: 0.3
  effects: []
```

### DSP Effects Catalog

| Effect | Parameters | Library |
|--------|-----------|---------|
| `pitch_shift` | semitones (float) | librosa |
| `time_stretch` | factor (float) | librosa |
| `low_pass_filter` | cutoff_hz (float) | scipy.signal |
| `high_pass_filter` | cutoff_hz (float) | scipy.signal |
| `high_shelf_boost` | frequency_hz, gain_db | scipy.signal |
| `breath_noise` | amplitude (float, 0-1) | numpy |
| `distortion` | intensity (float, 0-1) | numpy (waveshaping) |
| `compress` | threshold_db, ratio | numpy |
| `fade_out` | duration_s (float) | numpy |
| `fade_in` | duration_s (float) | numpy |
| `volume` | factor (float) | numpy |
| `speed_up` | factor (float) | librosa |

### Whisper Mode

`[whispers]` is special — it doesn't replace speech, it *modifies* the next speech segment. The pipeline detects this and applies the whisper effects to the following speech audio instead of generating a separate vocalization.

## File Organization

```
utilities/
  vocalization/
    __init__.py
    tag_parser.py              # Parse [tags] from text, return segments
    dsp_engine.py              # Audio effect implementations
    recipes.py                 # Load recipes from YAML, resolve effects
    vocalization_generator.py  # Orchestrates: TTS text → generate → DSP
    stitcher.py                # Crossfade stitching of audio segments
    recipes.yaml               # Tag → DSP recipe mappings (user-editable)
```

### Modified Files

- `utilities/audio_generation_pipeline.py` — Add tag detection before TTS; route through vocalization pipeline if tags present

### Integration Point

The vocalization check happens at the **beginning** of `audio_generation_pipeline.py`'s generate function:

```
def generate(text, speaker, ...):
    segments = parse_tags(text)

    if all segments are speech:
        # Normal path — zero overhead
        return existing_pipeline(text, speaker, ...)

    # Vocalization path
    audio_parts = []
    for segment in segments:
        if segment.is_vocalization:
            audio = generate_vocalization(segment, speaker)
        else:
            audio = generate_speech(segment.text, speaker)
        audio_parts.append(audio)

    return stitch(audio_parts)
```

## Performance

- **No tags**: Single regex scan (~microseconds), then existing pipeline unchanged. Zero measurable overhead.
- **With tags**: ~100-300ms extra DSP processing per vocalization segment. Speech segments take the normal TTS path. Total overhead is small relative to TTS generation time (~1-3s).
- All DSP effects use numpy/scipy — no GPU required, no model loading.

## Error Handling

| Case | Behavior |
|------|----------|
| Unknown tag | Warning log, treat as 0.3s silence pause |
| Empty text after tags | Produce just the vocalization(s) |
| Tag at end of speech | Append vocalization after speech |
| Multiple consecutive tags | Process each, stitch sequentially |
| Malformed/nested tags | Skip, treat as regular text |
| Very short TTS output | Pad to minimum duration needed by effects |

## Future Upgrade Path

The `vocalization_generator.py` module is the abstraction boundary. Currently it uses TTS + DSP. In the future, it can be replaced with a neural vocalization model (NVSpeech, CosyVoice fine-tune, etc.) without changing anything upstream:

```
vocalization_generator.py
  ├── DSPVocalizationGenerator  (current — TTS + DSP)
  └── NeuralVocalizationGenerator  (future — swap in when ready)
```

The tag parser, stitcher, and pipeline integration remain identical.
