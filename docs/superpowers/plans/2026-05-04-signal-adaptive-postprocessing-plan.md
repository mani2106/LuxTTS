# Signal-Adaptive Post-Processing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uniform 8-stage post-processing pipeline with a signal-adaptive pipeline that only applies processing when the audio actually needs it, reducing OVRL degradation from 30% to under 5%.

**Architecture:** A `SignalProfile` dataclass measures incoming audio (peak, RMS, spectral centroid, sibilance ratio) and drives adaptive decisions — which stages to run and how aggressively. Removed stages (spectral enrichment, prosodic modulation, room presence, auto pitch shift, full compressor) remain as opt-in parameters. LUFS target changes from -16 to -18.

**Tech Stack:** Python, numpy, scipy, pedalboard (optional), pyloudnorm (optional)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `utilities/post_processor.py` | Modify | Add `SignalProfile`, rewrite `process()` to be adaptive, add `_limit_peak()` method |
| `utilities/app_constants.py` | Modify | Update defaults for new LUFS target, adaptive thresholds, opt-in flags |
| `utilities/vocalization/recipes.json` | Modify | Add NSFW tags (moans, groans, whimpers, struggling), update whisper recipe |
| `tests/test_post_processor.py` | Modify | Add tests for SignalProfile, adaptive gating, limiter, new LUFS target |
| `tests/audio_quality/test_full_eval.py` | Modify | Add NSFW vocalization test cases |

---

### Task 1: Add SignalProfile dataclass and analysis

**Files:**
- Modify: `utilities/post_processor.py` (add after imports, before `PitchDetector` class ~line 26)
- Test: `tests/test_post_processor.py`

- [ ] **Step 1: Write the failing tests for SignalProfile**

Add to `tests/test_post_processor.py`:

```python
from utilities.post_processor import SignalProfile, analyze_signal


def test_analyze_signal_speech_like():
    """Speech-like audio should have moderate peak, RMS, and centroid."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Speech-like: fundamental + harmonics
    audio = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    assert 0.0 < profile.peak < 1.0
    assert 0.0 < profile.rms < 1.0
    assert 100 < profile.spectral_centroid < 5000
    assert 0.0 <= profile.sibilance_ratio <= 1.0


def test_analyze_signal_bright_audio():
    """Audio with lots of high-frequency content should have high sibilance ratio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Bright: mostly high-frequency
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
    # Boomy: mostly low-frequency content
    audio = (0.5 * np.sin(2 * np.pi * 100 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    profile = analyze_signal(audio, sr)

    assert profile.needs_mud_cut is True


def test_signal_profile_properties():
    """SignalProfile properties should return correct booleans."""
    profile = SignalProfile(peak=0.5, rms=0.1, spectral_centroid=2500.0, sibilance_ratio=0.05)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_analyze_signal_speech_like tests/test_post_processor.py::test_signal_profile_properties -v`
Expected: FAIL with `ImportError: cannot import name 'SignalProfile'`

- [ ] **Step 3: Implement SignalProfile and analyze_signal**

Add to `utilities/post_processor.py` after the imports (before `PitchDetector` class at ~line 26):

```python
@dataclass
class SignalProfile:
    """Analysis of audio signal characteristics for adaptive processing decisions."""
    peak: float
    rms: float
    spectral_centroid: float  # Hz
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


def analyze_signal(audio: np.ndarray, sr: int) -> SignalProfile:
    """Analyze audio to produce a SignalProfile for adaptive processing."""
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Spectral centroid via FFT
    n = len(audio)
    fft_magnitude = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total_energy = np.sum(fft_magnitude) + 1e-10
    spectral_centroid = float(np.sum(freqs * fft_magnitude) / total_energy)

    # Sibilance ratio: energy in 4-8kHz band / total energy
    sibilance_mask = (freqs >= 4000) & (freqs <= 8000)
    sibilance_energy = float(np.sum(fft_magnitude[sibilance_mask]))
    sibilance_ratio = sibilance_energy / total_energy

    return SignalProfile(
        peak=peak,
        rms=rms,
        spectral_centroid=spectral_centroid,
        sibilance_ratio=sibilance_ratio,
    )
```

Also add `dataclass` to imports at the top of the file:

```python
from dataclasses import dataclass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_analyze_signal tests/test_post_processor.py::test_signal_profile_properties -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: add SignalProfile dataclass and analyze_signal for adaptive pipeline"
```

---

### Task 2: Add adaptive limiter method

**Files:**
- Modify: `utilities/post_processor.py` (add method to `AudioPostProcessor` class, after `compress()` method ~line 560)
- Test: `tests/test_post_processor.py`

- [ ] **Step 1: Write the failing tests for adaptive limiter**

Add to `tests/test_post_processor.py`:

```python
def test_limit_peak_clipping_audio():
    """Limiter should reduce peaks that exceed threshold."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (0.98 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor()
    processed, diagnostics = processor.limit_peak(audio, sr, threshold_db=-1.0)

    peak_limit_linear = 10 ** (-1.0 / 20)
    assert np.max(np.abs(processed)) <= peak_limit_linear + 1e-6
    assert diagnostics['limiting_applied'] is True


def test_limit_peak_safe_audio():
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_limit_peak -v`
Expected: FAIL with `AttributeError: 'AudioPostProcessor' object has no attribute 'limit_peak'`

- [ ] **Step 3: Implement limit_peak method**

Add to `AudioPostProcessor` class in `utilities/post_processor.py` (after the `compress()` method ends, around line ~600):

```python
def limit_peak(
    self,
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -1.0,
) -> tuple[np.ndarray, dict]:
    """Brick-wall peak limiter. Only reduces samples exceeding threshold."""
    threshold_linear = 10 ** (threshold_db / 20)
    peak = np.max(np.abs(audio))

    if peak <= threshold_linear:
        return audio.copy(), {'limiting_applied': False, 'peak_before': float(peak)}

    # Simple brick-wall limiting with soft-knee smoothing
    # Apply gain reduction only where peaks exceed threshold
    processed = audio.copy()
    mask = np.abs(processed) > threshold_linear
    processed[mask] = np.sign(processed[mask]) * threshold_linear

    return processed.astype(np.float32), {
        'limiting_applied': True,
        'peak_before': float(peak),
        'peak_after': float(np.max(np.abs(processed))),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_limit_peak -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: add limit_peak brick-wall limiter method"
```

---

### Task 3: Update app_constants.py with new defaults

**Files:**
- Modify: `utilities/app_constants.py` (lines 21-32)

- [ ] **Step 1: Update constants**

In `utilities/app_constants.py`, replace the post-processing defaults section (lines 21-32) with:

```python
# Post-processing defaults
DEFAULT_POST_PROCESSING_ENABLED = True
DEFAULT_PITCH_SHIFT = None  # None = auto from text (only used when enable_auto_pitch_shift=True)
DEFAULT_EQ_INTENSITY = 1.0
DEFAULT_COMPRESSOR_THRESHOLD_OFFSET = -6.0  # dB offset from signal RMS
DEFAULT_COMPRESSOR_RATIO = 2.0
DEFAULT_COMPRESSOR_KNEE_DB = 8.0
DEFAULT_COMPRESSOR_ATTACK_MS = 10.0
DEFAULT_COMPRESSOR_RELEASE_MS = 100.0
DEFAULT_MAX_GAIN_REDUCTION_DB = 12.0
DEFAULT_DE_ESS_INTENSITY = 0.3  # Gentler for TTS (was 0.5)
DEFAULT_TARGET_LOUDNESS_LUFS = -18.0  # RPG dialogue standard (was -16.0)

# Signal-adaptive thresholds
SIBILANCE_RATIO_THRESHOLD = 0.15  # De-esser activates above this
CENTROID_LOW_THRESHOLD = 1500.0   # Hz - mud cut activates below this
CENTROID_HIGH_THRESHOLD = 3500.0  # Hz - presence boost activates above this
PEAK_LIMIT_THRESHOLD = 0.93      # Limiter activates above this
HPF_CUTOFF_HZ = 80.0             # High-pass filter cutoff (always on)
```

- [ ] **Step 2: Verify no import errors**

Run: `.venv/Scripts/python -c "from utilities.app_constants import DEFAULT_TARGET_LOUDNESS_LUFS, SIBILANCE_RATIO_THRESHOLD; print(f'LUFS={DEFAULT_TARGET_LOUDNESS_LUFS}, Sibilance={SIBILANCE_RATIO_THRESHOLD}')"`
Expected: `LUFS=-18.0, Sibilance=0.15`

- [ ] **Step 3: Commit**

```bash
git add utilities/app_constants.py
git commit -m "feat: update post-processing defaults for signal-adaptive pipeline"
```

---

### Task 4: Rewrite process() method with signal-adaptive pipeline

**Files:**
- Modify: `utilities/post_processor.py` (lines 867-1002, the `process()` method)
- Test: `tests/test_post_processor.py`

- [ ] **Step 1: Write failing tests for adaptive process()**

Add to `tests/test_post_processor.py`:

```python
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
    # Low-frequency only — no sibilance
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    processor = AudioPostProcessor(return_diagnostics=True)
    processed, diagnostics = processor.process(audio, sr)

    assert diagnostics['signal_profile']['needs_de_essing'] is False


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_process_adaptive -v`
Expected: FAIL — current `process()` doesn't produce `signal_profile` in diagnostics

- [ ] **Step 3: Rewrite the process() method**

Replace the `process()` method in `utilities/post_processor.py` (lines 867-1002) with:

```python
def process(
    self,
    audio: np.ndarray,
    sr: int,
    text: Optional[str] = None,
    pitch_shift: Optional[float] = None,
    eq_intensity: float = 1.0,
    de_ess_intensity: float = 0.3,
    compressor_threshold_offset_db: float = -6.0,
    compressor_ratio: float = 2.0,
    compressor_knee_db: float = 8.0,
    compressor_attack_ms: float = 10.0,
    compressor_release_ms: float = 100.0,
    max_gain_reduction_db: float = 12.0,
    target_loudness: float = -18.0,
    enable_post_processing: bool = True,
    enable_spectral_enrichment: bool = False,
    enable_room_presence: bool = False,
    enable_prosodic_modulation: bool = False,
    enable_auto_pitch_shift: bool = False,
    enable_compressor: bool = False,
) -> tuple[np.ndarray, dict]:
    """Process audio through the signal-adaptive post-processing chain.

    Adaptive pipeline (default):
    0. Signal analysis -> SignalProfile
    1. High-pass filter (always, 80Hz)
    2. Adaptive de-esser (if sibilance detected)
    3. Adaptive EQ (if spectral centroid outside normal range)
    4. Adaptive limiter (if peak > threshold)
    5. Loudness normalization (always, -18 LUFS)

    Opt-in stages (default off):
    - Compressor, pitch shift, prosodic modulation, room presence,
      spectral enrichment

    Args:
        audio: Input audio (float32, typically 48kHz)
        sr: Sample rate
        text: Dialogue text (used for auto pitch detection if enabled)
        pitch_shift: Manual pitch override in semitones
        eq_intensity: EQ intensity (0.0-1.0)
        de_ess_intensity: De-essing intensity (0.0-1.0)
        compressor_*: Compressor parameters (only used if enable_compressor=True)
        target_loudness: Target loudness in LUFS (default -18.0)
        enable_post_processing: If False, bypass all processing
        enable_spectral_enrichment: Enable harmonic exciter (opt-in)
        enable_room_presence: Enable room reverb (opt-in)
        enable_prosodic_modulation: Enable amplitude modulation (opt-in)
        enable_auto_pitch_shift: Enable heuristic pitch detection (opt-in)
        enable_compressor: Enable full compressor instead of adaptive limiter (opt-in)

    Returns:
        (processed_audio, diagnostics_dict)
    """
    if not enable_post_processing:
        return audio.copy(), {}

    all_diagnostics = {}

    # Stage 0: Signal analysis
    profile = analyze_signal(audio, sr)
    all_diagnostics['signal_profile'] = {
        'peak': profile.peak,
        'rms': profile.rms,
        'spectral_centroid': profile.spectral_centroid,
        'sibilance_ratio': profile.sibilance_ratio,
        'needs_limiting': profile.needs_limiting,
        'needs_de_essing': profile.needs_de_essing,
        'needs_mud_cut': profile.needs_mud_cut,
        'needs_presence_boost': profile.needs_presence_boost,
    }

    # Stage 1: High-pass filter (always)
    b, a = signal.butter(2, 80 / (sr / 2), btype='high')
    audio = signal.filtfilt(b, a, audio).astype(np.float32)
    all_diagnostics['high_pass_filter'] = {'cutoff_hz': 80}

    # Stage 2: Adaptive de-esser
    if profile.needs_de_essing and de_ess_intensity > 0:
        audio, de_ess_diag = self.de_esser(audio, sr, intensity=de_ess_intensity)
        if de_ess_diag:
            all_diagnostics['de_esser'] = de_ess_diag
    elif HAS_TDR_NOVA and self.tdr_nova_plugin is not None and de_ess_intensity > 0:
        # TDR Nova combines de-esser + EQ — only use if de-essing would be beneficial
        pass  # Skip TDR Nova when signal doesn't need de-essing

    # Stage 3: Adaptive EQ
    if profile.needs_mud_cut:
        b, a = self._design_peaking(300, 1.0, -2.0 * eq_intensity, sr)
        audio = signal.filtfilt(b, a, audio).astype(np.float32)
        all_diagnostics['adaptive_eq'] = {'action': 'mud_cut', 'frequency': 300, 'gain_db': -2.0 * eq_intensity}
    elif profile.needs_presence_boost:
        b, a = self._design_peaking(3000, 1.0, 1.0 * eq_intensity, sr)
        audio = signal.filtfilt(b, a, audio).astype(np.float32)
        all_diagnostics['adaptive_eq'] = {'action': 'presence_boost', 'frequency': 3000, 'gain_db': 1.0 * eq_intensity}

    # Stage 4: Adaptive limiter OR opt-in compressor
    if enable_compressor:
        audio, comp_diag = self.compress(
            audio, sr,
            threshold_offset_db=compressor_threshold_offset_db,
            ratio=compressor_ratio,
            knee_db=compressor_knee_db,
            attack_ms=compressor_attack_ms,
            release_ms=compressor_release_ms,
            max_reduction_db=max_gain_reduction_db,
        )
        if comp_diag:
            all_diagnostics['compressor'] = comp_diag
    elif profile.needs_limiting:
        audio, limiter_diag = self.limit_peak(audio, sr, threshold_db=-1.0)
        if limiter_diag:
            all_diagnostics['peak_limiter'] = limiter_diag

    # Opt-in: Auto pitch shift
    if enable_auto_pitch_shift:
        if pitch_shift is None and text:
            detector = PitchDetector()
            detected_pitch = detector.detect_pitch(text)
        else:
            detected_pitch = pitch_shift if pitch_shift is not None else 0.0
        audio, pitch_diag = self.pitch_shift(audio, sr, n_steps=detected_pitch)
        if pitch_diag:
            all_diagnostics['pitch_shift'] = pitch_diag

    # Opt-in: Prosodic modulation
    if enable_prosodic_modulation:
        audio, prosodic_diag = self.prosodic_modulation(audio, sr, text=text or "")
        if prosodic_diag:
            all_diagnostics['prosodic_modulation'] = prosodic_diag

    # Opt-in: Room presence
    if enable_room_presence:
        audio, room_diag = self.room_presence(audio, sr)
        if room_diag:
            all_diagnostics['room_presence'] = room_diag

    # Opt-in: Spectral enrichment
    if enable_spectral_enrichment:
        audio, enrich_diag = self.spectral_enrich(audio, sr)
        if enrich_diag:
            all_diagnostics['spectral_enrich'] = enrich_diag

    # Stage 5: Loudness normalization (always)
    audio, loudness_diag = self.normalize_loudness(audio, sr, target_lufs=target_loudness)
    if loudness_diag:
        loudness_diag['target_lufs'] = target_loudness
        all_diagnostics['normalize_loudness'] = loudness_diag

    return audio.astype(np.float32), all_diagnostics
```

- [ ] **Step 4: Run the new tests**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_process_adaptive tests/test_post_processor.py::test_process_disabled tests/test_post_processor.py::test_process_target_lufs tests/test_post_processor.py::test_process_removed_stages_not_in_default_diagnostics tests/test_post_processor.py::test_process_always_runs -v`
Expected: All PASS

- [ ] **Step 5: Run ALL existing tests to check for regressions**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py -v`
Expected: All PASS. Existing tests for individual methods (de_esser, equalize, compress, etc.) should still pass since those methods are unchanged.

- [ ] **Step 6: Commit**

```bash
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: rewrite process() with signal-adaptive pipeline

- Signal analysis drives which stages run
- De-esser, EQ, limiter are conditionally applied
- Spectral enrichment, room presence, prosodic modulation, auto pitch shift
  are opt-in (enable_* parameters, default False)
- Compressor is opt-in (enable_compressor, default False)
- LUFS default changed to -18.0"
```

---

### Task 5: Update audio_generation_pipeline.py parameter defaults

**Files:**
- Modify: `utilities/audio_generation_pipeline.py` (update default parameter values in `generate_audio()` signature)

- [ ] **Step 1: Update the default imports and parameter values**

In `utilities/audio_generation_pipeline.py`, the `generate_audio()` function imports defaults from `app_constants.py`. The constants already changed in Task 3, so the imports are fine. But verify the function signature passes the correct defaults through to `process()`.

Check that `de_ess_intensity` and `target_loudness` use the constants:
- `de_ess_intensity: float = DEFAULT_DE_ESS_INTENSITY` — should now resolve to 0.3
- `target_loudness: float = DEFAULT_TARGET_LOUDNESS_LUFS` — should now resolve to -18.0

If these are hardcoded (not using constants), update them to use the constants.

Run: `grep -n "de_ess_intensity\|target_loudness\|DEFAULT_DE_ESS\|DEFAULT_TARGET_LOUDNESS" utilities/audio_generation_pipeline.py`

If any are hardcoded to the old values (0.5 or -16.0), update them to use the constants.

- [ ] **Step 2: Verify import works**

Run: `.venv/Scripts/python -c "from utilities.audio_generation_pipeline import generate_audio; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add utilities/audio_generation_pipeline.py
git commit -m "fix: update pipeline defaults to use new constant values"
```

---

### Task 6: Update vocalization recipes — new NSFW tags and whisper fix

**Files:**
- Modify: `utilities/vocalization/recipes.json`

- [ ] **Step 1: Add NSFW tags and update whisper recipe**

The tags `moans`, `groans`, and `whimpers` already exist in recipes.json. Update `whimpers` with the spec parameters and add the new `struggling` tag. Update `whispers` to add LPF.

Update the `whispers` entry to add band-pass filtering:

```json
"whispers": {
    "tts_text": null,
    "mode": "modify_speech",
    "effects": [
      {"type": "pitch_shift", "semitones": -2},
      {"type": "high_pass_filter", "cutoff_hz": 600},
      {"type": "low_pass_filter", "cutoff_hz": 4000},
      {"type": "breath_noise", "amplitude": 0.15},
      {"type": "volume", "factor": 0.4}
    ]
}
```

Update the `whimpers` entry to match spec parameters (LPF 900Hz instead of 700Hz, breath 0.10 instead of 0.08):

```json
"whimpers": {
    "tts_text": "ah",
    "tts_speed": 0.9,
    "max_duration_s": 1.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -3},
      {"type": "low_pass_filter", "cutoff_hz": 900},
      {"type": "breath_noise", "amplitude": 0.10},
      {"type": "volume", "factor": 0.6}
    ]
}
```

Update `moans` entry to match spec (breath 0.08 instead of 0.05):

```json
"moans": {
    "tts_text": "oooooh",
    "tts_speed": 0.7,
    "max_duration_s": 2.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -4},
      {"type": "low_pass_filter", "cutoff_hz": 700},
      {"type": "breath_noise", "amplitude": 0.08},
      {"type": "fade_out", "duration_s": 0.6}
    ]
}
```

Add new `struggling` tag after `sobs`:

```json
"struggling": {
    "tts_text": "ngh",
    "tts_speed": 0.85,
    "max_duration_s": 1.5,
    "effects": [
      {"type": "pitch_shift", "semitones": -4},
      {"type": "low_pass_filter", "cutoff_hz": 800},
      {"type": "distortion", "intensity": 0.15},
      {"type": "compress", "threshold_db": -12, "ratio": 3},
      {"type": "volume", "factor": 1.0}
    ]
}
```

- [ ] **Step 2: Verify JSON is valid**

Run: `.venv/Scripts/python -c "import json; data=json.load(open('utilities/vocalization/recipes.json')); print(f'Tags: {list(data.keys())}')"`
Expected: Tags list includes `struggling` alongside existing tags

- [ ] **Step 3: Commit**

```bash
git add utilities/vocalization/recipes.json
git commit -m "feat: add struggling tag, update whisper/whimper/moan recipes per spec"
```

---

### Task 7: Add NSFW vocalization test cases to audio quality eval

**Files:**
- Modify: `tests/audio_quality/test_full_eval.py` (lines 37-42)

- [ ] **Step 1: Add new vocalization test cases**

In `tests/audio_quality/test_full_eval.py`, update `VOCALIZATION_CASES` to include the new tags:

```python
VOCALIZATION_CASES = [
    ("vocalization_sighs", "[sighs] I can't believe we made it."),
    ("vocalization_gasps", "[gasps] Who's there?"),
    ("vocalization_whispers", "[whispers] Don't make a sound."),
    ("vocalization_screams", "[screams] Get away from me!"),
    ("vocalization_moans", "[moans]"),
    ("vocalization_whimpers", "[whimpers] Please stop."),
    ("vocalization_struggling", "[struggling]"),
    ("vocalization_groans", "[groans]"),
]
```

- [ ] **Step 2: Verify tests are collected**

Run: `.venv/Scripts/python -m pytest tests/audio_quality/test_full_eval.py::test_vocalization_generation --collect-only`
Expected: Shows all 8 parametrized vocalization test cases collected

- [ ] **Step 3: Commit**

```bash
git add tests/audio_quality/test_full_eval.py
git commit -m "feat: add NSFW vocalization test cases to audio quality eval"
```

---

### Task 8: Run full test suite and lint

**Files:** None — verification only

- [ ] **Step 1: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -v --ignore=tests/audio_quality -x`
Expected: All tests PASS. The `--ignore=tests/audio_quality` skips GPU tests.

- [ ] **Step 2: Run lint**

Run: `.venv/Scripts/python -m ruff check utilities/post_processor.py utilities/app_constants.py utilities/vocalization/recipes.json tests/test_post_processor.py`
Expected: No errors. If any, fix them.

- [ ] **Step 3: Fix the GradScaler import bug (if not already fixed)**

Verify the fix from earlier in the session is still in place:

Run: `.venv/Scripts/python -c "from zipvoice.utils.common import GradScaler; print('GradScaler import OK')"`
Expected: `GradScaler import OK`

- [ ] **Step 4: Final commit if any lint fixes needed**

If lint fixes were needed:
```bash
git add -A
git commit -m "style: lint fixes for signal-adaptive pipeline"
```

---

### Task 9: Run audio quality evaluation (requires GPU)

**Files:** None — evaluation only

This task generates fresh audio with the new pipeline and scores it. Requires GPU.

- [ ] **Step 1: Run full eval with generation**

Run: `.venv/Scripts/python tests/run_audio_eval.py`
Expected: Generation passes for all speakers, scoring completes. The 3 clipping speakers (alduin, femalekhajiit, maleoldgrumpy) may still fail raw gate checks — that's expected (model-level issue).

- [ ] **Step 2: If generation has failures, run scoring only**

If 3 speakers fail gate checks (expected), run scoring on the generated samples:
Run: `.venv/Scripts/python tests/run_audio_eval.py --skip-generate`

- [ ] **Step 3: Review results**

Compare the new scores against the baseline from the design spec. Target: processed OVRL within 5% of raw OVRL (not 30% degradation).

If degradation is still above 5%, investigate which stages are still causing drops and tune thresholds.

- [ ] **Step 4: Save as new baseline**

Once results look good, update the baseline:
```bash
cp tests/audio_quality/baselines/latest_scores.json tests/audio_quality/baselines/master_raw_baseline.json
```

- [ ] **Step 5: Commit baseline**

```bash
git add tests/audio_quality/baselines/
git commit -m "feat: update audio quality baseline with signal-adaptive pipeline scores"
```

---

### Task 10: Update decision records

**Files:** None — repowise tool only

- [ ] **Step 1: Record the architectural decision**

Use repowise `update_decision_records` to create a record of this pipeline change.

Action: `create`
Title: `Signal-Adaptive Post-Processing Pipeline`
Decision: `Replaced uniform 8-stage post-processing with 6-stage signal-adaptive pipeline. Stages run conditionally based on signal analysis (peak, RMS, spectral centroid, sibilance ratio). Removed spectral enrichment, room presence, prosodic modulation, and auto pitch shift from default chain (opt-in via enable_* params). Replaced full compressor with conditional limiter. LUFS target changed from -16 to -18.`
Rationale: `Audio quality evaluation showed 30% average OVRL degradation from post-processing. Research confirmed spectral enrichment and room reverb are harmful for clean dialogue. SkyrimNet GamePlugin bypasses Skyrim's sound system, so room reverb must come from SkyrimNet's voice effects, not TTS output.`
Affected_files: `["utilities/post_processor.py", "utilities/app_constants.py", "utilities/vocalization/recipes.json"]`
Tags: `["audio", "post-processing", "pipeline"]`

- [ ] **Step 2: Update existing DSP pipeline decision status**

Use repowise `update_decision_records` with action `update_status`:
Decision ID: `ca5ca79d73c1498ea1e60335c7098fcb` (the existing "DSP Post-Processing Pipeline" decision)
Status: `superseded`
Superseded by: the new decision ID from step 1

---

## Self-Review

**1. Spec coverage check:**
- SignalProfile + analyze_signal: Task 1 ✓
- HPF stage: Task 4 (inside process()) ✓
- Adaptive de-esser: Task 4 ✓
- Adaptive EQ: Task 4 ✓
- Adaptive limiter: Task 2 (method) + Task 4 (wiring) ✓
- Loudness normalization -18 LUFS: Task 3 (constant) + Task 4 (process) ✓
- Removed stages opt-in: Task 4 (enable_* params) ✓
- NSFW vocalization tags: Task 6 ✓
- Whisper LPF update: Task 6 ✓
- SkyrimNet responsibility split: Addressed by removing room reverb from default ✓
- Testing strategy: Task 8 + Task 9 ✓
- Decision records: Task 10 ✓

**2. Placeholder scan:** No TBDs, TODOs, or "implement later" patterns found.

**3. Type consistency:**
- `SignalProfile` dataclass defined in Task 1, used consistently in Task 4
- `analyze_signal()` returns `SignalProfile` — used in both tests and process()
- `limit_peak()` returns `tuple[np.ndarray, dict]` — matches other methods
- Default parameter `de_ess_intensity=0.3` matches `DEFAULT_DE_ESS_INTENSITY`
- Default parameter `target_loudness=-18.0` matches `DEFAULT_TARGET_LOUDNESS_LUFS`
