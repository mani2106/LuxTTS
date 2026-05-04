# Signal-Adaptive Post-Processing Pipeline

**Date:** 2026-05-04
**Status:** Proposed
**Supersedes:** 2026-05-03-audio-quality-refactor-design.md

## Problem

The current 8-stage post-processing pipeline degrades every speaker's audio quality by an average of 30.7% OVRL (DNSMOS), with some speakers losing up to 45%. Speaker identity (SIM) drops for multiple voices. The pipeline applies all stages uniformly regardless of signal characteristics — loud speakers that need limiting get the same heavy chain as quiet speakers that don't.

Additionally, the pipeline includes stages that professional audio engineers consider harmful for clean dialogue (spectral enrichment, prosodic modulation, baked room reverb).

## Context

### Audio Quality Evaluation Results (2026-05-04)

| Speaker | Raw OVRL | Processed OVRL | Drop | Raw SIM | Proc SIM |
|---------|----------|----------------|------|---------|----------|
| femaleoldgrumpy | 3.12 | 2.34 | -25% | 0.66 | 0.55 |
| malecommoner | 2.99 | 1.93 | -35% | 0.67 | 0.43 |
| maleargonian | 2.95 | 2.46 | -17% | 0.54 | 0.47 |
| cicero | 2.83 | 2.08 | -27% | 0.25 | 0.25 |
| femalenord | 2.81 | 1.54 | -45% | 0.58 | 0.64 |
| malekhajiit | 2.80 | 1.90 | -32% | 0.68 | 0.61 |
| femaleargonian | 2.34 | 1.82 | -22% | 0.56 | 0.39 |
| serana | 2.07 | 1.32 | -36% | 0.47 | 0.29 |
| aaaharleyvoicequest | 1.82 | 1.36 | -25% | 0.14 | 0.17 |

Three speakers (alduin, femalekhajiit, maleoldgrumpy) clip at peak=1.0 in raw TTS output — a model-level issue.

### SkyrimNet Integration

SkyrimNet's C++ DLL bypasses Skyrim's sound system entirely. It plays audio directly through its own pipeline (likely XAudio2), not through Skyrim's sound descriptors or reverb system. This means:

- No Skyrim reverb/ambience on generated speech
- No Skyrim distance attenuation
- No Skyrim sound category management

SkyrimNet has its own voice effects system (YAML-based) that applies character-specific DSP (pitch shift, reverb, chorus, distortion) after our TTS output.

**Responsibility split:**
- **LuxTTS server:** peak protection, loudness normalization, gentle de-essing, basic EQ, vocalization DSP
- **SkyrimNet voice effects:** character-specific reverb, creature pitch/formant shifts, supernatural effects

### Research Grounding

Key findings from professional audio engineering research:

1. **Spectral enrichment is harmful on clean dialogue** — Industry consensus: harmonic exciters add upper-midrange buildup and listening fatigue. DNSMOS penalizes waveshaping as degradation. (Sources: Gearspace forums, Waves engineering blog, Aphex product reviews)
2. **Reverb should be delivered dry for game dialogue** — Game engines handle reverb dynamically. Baking reverb into TTS output is counterproductive. (Sources: Audiokinetic Wwise docs, game audio practice)
3. **LUFS target should be -18 to -20 for PC/console RPG** — Console standards are -24 LUFS but observed game average is -18.5. Current -16 is too loud. (Sources: AES/EBU R128, game audio loudness studies)
4. **Compression only needed for loud signals** — 2:1 ratio is appropriate when compression is needed, but many voices don't need it at all. (Sources: GDC talks, game audio forums)
5. **Vocalizations need different processing per type** — Screams: heavier compression. Sighs: lighter. Gasps: fast-attack. (Sources: game audio dialogue processing, Soundgen research)

## Design

### Pipeline Architecture

```
Raw TTS Audio (48kHz, mono)
    |
    v
[0. Signal Analysis]  <- peak, RMS, spectral centroid, sibilance ratio
    |                   Produces SignalProfile
    v
[1. High-Pass Filter] <- ALWAYS RUNS
    |                   80Hz, 12dB/oct
    v
[2. Adaptive De-esser] <- RUNS IF sibilance ratio > threshold
    |                    Intensity 0.3, wider band (4-8kHz), gentler reduction
    v
[3. Adaptive EQ]       <- RUNS IF spectral centroid outside normal range
    |                    Mud cut at 300Hz (-2dB) if boomy
    |                    Presence at 3kHz (+1dB) if dull
    |                    Max +/-2dB per band
    v
[4. Adaptive Limiter]  <- RUNS IF peak > 0.93
    |                    Brick-wall at -1dBTP
    v
[5. Loudness Norm]     <- ALWAYS RUNS
    |                    Target: -18 LUFS, peak ceiling -1dBTP
    v
Output: clean, dry, normalized WAV
```

### Stage Details

#### Stage 0: Signal Analysis

Measures incoming audio characteristics to drive adaptive decisions.

```python
@dataclass
class SignalProfile:
    peak: float              # Max absolute amplitude
    rms: float               # Root mean square level
    spectral_centroid: float  # Hz - brightness measure
    sibilance_ratio: float    # Energy in 4-8kHz / total energy

    @property
    def needs_limiting(self) -> bool:
        return self.peak > 0.93

    @property
    def needs_de_essing(self) -> bool:
        return self.sibilance_ratio > 0.15

    @property
    def needs_mud_cut(self) -> bool:
        return self.spectral_centroid < 1500

    @property
    def needs_presence_boost(self) -> bool:
        return self.spectral_centroid > 3500 and self.rms < 0.15
```

Implementation: numpy operations only — peak/RMS are trivial, spectral centroid via `np.sum(freq * magnitude) / np.sum(magnitude)` on FFT, sibilance ratio via bandpass energy comparison. No model inference, no heavy computation.

#### Stage 1: High-Pass Filter (Always)

- 80Hz cutoff, 12dB/oct (2nd-order Butterworth)
- Standard dialogue practice: removes rumble, handling noise, plosive artifacts
- Runs unconditionally — no dialogue content exists below 80Hz

#### Stage 2: Adaptive De-esser

**Condition:** sibilance_ratio > 0.15

- Frequency range: 4-8kHz (wider than current 5-8kHz — TTS sibilance is more consistent than recorded)
- Max gain reduction: 4dB (was effectively unlimited — clipped at 0.3 = -10.5dB)
- Intensity: 0.3 (was 0.5)
- Gentle adaptive threshold based on signal RMS

**Why adaptive:** Not all TTS output needs de-essing. Quiet, breathy voices have low sibilance. Only apply when the signal actually has excessive high-frequency energy.

#### Stage 3: Adaptive EQ

**Condition:** spectral centroid outside 1500-3500Hz range

- If centroid < 1500Hz (boomy/muddy): -2dB peaking cut at 300Hz, Q=1.0
- If centroid > 3500Hz and low RMS (thin/bright): +1dB peaking boost at 3kHz, Q=1.0
- Max adjustment: +/-2dB per band (was -6dB high shelf — too aggressive)

**Why adaptive:** The current -6dB high shelf at 8kHz was smearing signal clarity (SIG drops of 10-25%). TTS output spectral balance varies by speaker — applying the same EQ to everyone is counterproductive.

**Removed:** -6dB high-shelf cut at 8kHz and +2dB presence boost. These are replaced by adaptive corrections that only engage when the signal actually needs them.

#### Stage 4: Adaptive Limiter

**Condition:** peak > 0.93

- Brick-wall limiter at -1dBTP
- Fast attack (1ms), moderate release (50ms)
- Makeup gain to compensate for limiting

**Replaces:** The current full compressor (adaptive threshold, 2:1 ratio, soft knee, look-ahead, makeup gain, hard limiter). Compression is overkill — most speakers don't need dynamics processing. Only the three speakers that clip (alduin, femalekhajiit, maleoldgrumpy) need peak protection.

**Why a limiter, not a compressor:** Compressors reduce dynamic range across the board. For game dialogue delivered through SkyrimNet's direct playback (which bypasses Skyrim's sound category management), we want to preserve the TTS model's natural dynamics. A limiter only catches peaks that would clip — it's invisible when the signal is below threshold.

#### Stage 5: Loudness Normalization (Always)

- Target: -18 LUFS (was -16)
- Peak ceiling: -1dBTP
- Uses pyloudnorm for ITU-R BS.1770-4 if available, RMS fallback otherwise

**Why -18 LUFS:** Console RPG standards target -24 LUFS but observed game average is -18.5. Since SkyrimNet bypasses Skyrim's sound category volume management, our output needs to be at a reasonable listening level. -18 LUFS provides headroom for SkyrimNet's voice effects to layer on top without clipping.

### Stages Removed from Speech Chain

| Stage | Reason | Grounding |
|-------|--------|-----------|
| Spectral enrichment | Introduces harmonics DNSMOS reads as noise; pros don't use exciters on clean dialogue | Gearspace consensus, Waves engineering blog |
| Room presence | SkyrimNet bypasses Skyrim's reverb; has its own voice effects for spatial processing | SkyrimNet GamePlugin architecture |
| Prosodic modulation | 3.5Hz AM is audible tremolo; DNSMOS penalizes as artifact | Signal processing principles, DNSMOS behavior research |
| Auto pitch shift | Heuristic text-to-pitch is unreliable; artifacts beyond +/-2 semitones | Professional pitch shifting guidance |
| Full compressor | Limiter sufficient for peak protection; compression kills dynamics unnecessarily | Game dialogue compression practices |

**Opt-in preservation:** Removed stages remain as callable methods on `AudioPostProcessor` but are not called by `process()` by default. They can be invoked via explicit parameters: `enable_spectral_enrichment=False`, `enable_room_presence=False`, `enable_prosodic_modulation=False`, `enable_auto_pitch_shift=False`. Default is `False` for all. This preserves backward compatibility without affecting the default audio quality.

### Vocalization Pipeline

Vocalizations go through their own DSP chains (defined in recipes.json), NOT through the main adaptive pipeline. This is the current architecture and remains unchanged.

#### Existing Vocalization Updates

**Screams:**
- Current recipe is reasonable (pitch +5, distortion 0.2, HF boost, comp 4:1)
- Consider formant-preserving pitch shift if librosa version supports it
- Keep current parameters

**Whispers:**
- Add band-pass approach: HPF at 600Hz AND LPF at 4kHz (currently HPF only)
- Preserves whisper-critical 1-4kHz range, removes unnecessary HF noise
- DNSMOS evaluation is invalid for whispers — use separate baseline

#### New NSFW Vocalization Tags

Grounded in Soundgen (Anikin, 2019) published parameters for non-speech vocalization synthesis.

**[moans]:**
- tts_text: "oooooh"
- tts_speed: 0.7
- effects: pitch_shift(-4), low_pass_filter(700Hz), breath_noise(0.08), fade_out(0.6s)
- Frequency range: 80-200Hz fundamental, energy below 1kHz

**[groans]:**
- tts_text: "uuugh"
- tts_speed: 0.7
- effects: pitch_shift(-3), low_pass_filter(600Hz), breath_noise(0.06), volume(1.1x)
- Frequency range: 80-150Hz fundamental, content below 800Hz

**[whimpers]:**
- tts_text: "ah"
- tts_speed: 0.9
- effects: pitch_shift(-3), low_pass_filter(900Hz), breath_noise(0.10), volume(0.6x)
- Frequency range: 200-400Hz fundamental, short rising-falling syllables

**[struggling]:**
- tts_text: "ngh"
- tts_speed: 0.85
- effects: pitch_shift(-4), low_pass_filter(800Hz), distortion(0.15), compress(threshold=-12dB, ratio=3:1)
- Frequency range: 80-180Hz fundamental, strain harmonics to 2kHz

### Files Modified

| File | Change |
|------|--------|
| `utilities/post_processor.py` | Rewrite `process()` to use signal-adaptive pipeline; add `SignalProfile` dataclass; remove spectral enrichment, prosodic modulation, room presence from default chain; make them opt-in |
| `utilities/app_constants.py` | Update defaults: LUFS -18 (was -16), de-ess 0.3 (was 0.5), add SIBILANCE_THRESHOLD, CENTROID_LOW, CENTROID_HIGH, PEAK_LIMIT_THRESHOLD |
| `utilities/vocalization/recipes.json` | Add moans, groans, whimpers, struggling tags; update whisper recipe to add LPF |
| `tests/test_post_processor.py` | Add tests for SignalProfile, adaptive stage gating, new LUFS target, new vocalization tags |
| `tests/audio_quality/test_full_eval.py` | Add new NSFW vocalization test cases |

### Testing Strategy

1. **DNSMOS regression guard:** Processed OVRL should be within 5% of raw OVRL (not 30%)
2. **Speaker similarity:** Processed SIM should be within 0.05 of raw SIM
3. **Peak compliance:** No output should exceed -1dBTP
4. **Adaptive gating:** Verify stages skip when conditions not met
5. **Whisper evaluation:** Separate baseline — DNSMOS invalid for unvoiced speech
6. **New baseline:** Save scores after implementation for future regression tracking

### Out of Scope

- Per-speaker DSP profiles (signal-adaptive handles this automatically)
- SkyrimNet voice effects changes (separate system)
- TTS model improvements (cicero SIM 0.25, aaaharleyvoicequest SIM 0.14 are model-level issues)
- Formant-preserving pitch shift (requires different algorithm than librosa provides — future work)
- WORLD vocoder whisper approach (higher quality but significant implementation effort — future work)

### Risks

1. **Signal analysis thresholds may need tuning.** The 0.93 peak threshold, 0.15 sibilance ratio, and 1500/3500Hz centroid boundaries are initial estimates that need validation against the full speaker set.
2. **DNSMOS may still penalize processed output** even with gentler processing, since it was trained on noise suppression data. The metric should be used as a regression guard, not a quality target.
3. **SkyrimNet voice effects interaction.** If SkyrimNet applies heavy effects on top of our output, the combined processing could introduce artifacts. This needs integration testing.
