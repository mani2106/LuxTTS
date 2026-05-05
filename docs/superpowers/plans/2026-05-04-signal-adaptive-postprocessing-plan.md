# Signal-Adaptive Post-Processing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uniform 8-stage post-processing pipeline with a signal-adaptive pipeline that only applies processing when the audio actually needs it, reducing OVRL degradation from 30% to under 5%.

**Architecture:** A `SignalProfile` dataclass measures incoming audio (peak, RMS, true-peak dBTP, spectral centroid, sibilance ratio, crest factor, spectral tilt) and drives adaptive decisions — which stages run and how aggressively. The limiter uses soft-knee limiting instead of brick-wall clipping. De-esser intensity scales proportionally with measured sibilance. A standardized manifest JSON is emitted per sample for regression tracing. Removed stages (spectral enrichment, prosodic modulation, room presence, auto pitch shift, full compressor) remain as opt-in parameters. LUFS target changes from -16 to -18.

**Tech Stack:** Python, numpy, scipy, pedalboard (optional), pyloudnorm (optional)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `utilities/post_processor.py` | Modify | Add `SignalProfile`, rewrite `process()` to be adaptive, add `_limit_peak()` soft-knee limiter, proportional de-esser, manifest output |
| `utilities/app_constants.py` | Modify | Update defaults for new LUFS target, adaptive thresholds, limiter params, opt-in flags |
| `utilities/vocalization/recipes.json` | Modify | Add struggling tag, update whisper/whimper/moan recipes |
| `utilities/vocalization/distinctness_check.py` | Create | Vocalization distinctness check (centroid + energy ratio vs speech baseline) |
| `tests/test_post_processor.py` | Modify | Add tests for SignalProfile, adaptive gating, soft-knee limiter, proportional de-esser, manifest output |
| `tests/test_distinctness_check.py` | Create | Tests for vocalization distinctness check |
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_analyze_signal_speech_like tests/test_post_processor.py::test_signal_profile_properties -v`
Expected: FAIL with `ImportError: cannot import name 'SignalProfile'`

- [ ] **Step 3: Implement SignalProfile and analyze_signal**

Add to `utilities/post_processor.py` after the imports (before `PitchDetector` class at ~line 26):

```python
from dataclasses import dataclass


@dataclass
class SignalProfile:
    """Analysis of audio signal characteristics for adaptive processing decisions."""
    peak: float                      # Max absolute sample amplitude
    true_peak_db: float              # True-peak in dBTP via 4x oversampling
    rms: float                       # Root mean square level
    spectral_centroid: float         # Hz - brightness measure
    sibilance_ratio: float           # Energy in 4-8kHz / total energy
    crest_factor_db: float           # 20*log10(peak/rms) - dynamics measure
    spectral_tilt_db_per_octave: float  # Slope of spectrum - body/brightness proxy

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


def _compute_true_peak_db(audio: np.ndarray) -> float:
    """Compute true-peak via 4x oversampling."""
    n = len(audio)
    # Zero-pad to 4x length
    padded = np.zeros(n * 4, dtype=np.float64)
    padded[::4] = audio.astype(np.float64)
    # Low-pass filter (sinc interpolation approximation)
    cutoff = 1.0 / 4.0
    b, a = signal.butter(8, cutoff, btype='low')
    oversampled = signal.filtfilt(b, a, padded)
    true_peak = np.max(np.abs(oversampled))
    return float(20 * np.log10(true_peak + 1e-10))


def analyze_signal(audio: np.ndarray, sr: int) -> SignalProfile:
    """Analyze audio to produce a SignalProfile for adaptive processing."""
    peak = float(np.max(np.abs(audio)))
    true_peak_db = _compute_true_peak_db(audio)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    crest_factor_db = float(20 * np.log10(peak / (rms + 1e-10)))

    # Spectral analysis via FFT
    n = len(audio)
    fft_magnitude = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total_energy = np.sum(fft_magnitude) + 1e-10
    spectral_centroid = float(np.sum(freqs * fft_magnitude) / total_energy)

    # Sibilance ratio: energy in 4-8kHz band / total energy
    sibilance_mask = (freqs >= 4000) & (freqs <= 8000)
    sibilance_energy = float(np.sum(fft_magnitude[sibilance_mask]))
    sibilance_ratio = sibilance_energy / total_energy

    # Spectral tilt: slope of log-spectrum in dB per octave
    # Fit linear regression to (log2(freq), magnitude_dB)
    valid = freqs > 0
    log_freqs = np.log2(freqs[valid])
    mag_db = 20 * np.log10(fft_magnitude[valid] + 1e-10)
    if len(log_freqs) > 1:
        coeffs = np.polyfit(log_freqs, mag_db, 1)
        spectral_tilt = float(coeffs[0])  # dB per octave
    else:
        spectral_tilt = 0.0

    return SignalProfile(
        peak=peak,
        true_peak_db=true_peak_db,
        rms=rms,
        spectral_centroid=spectral_centroid,
        sibilance_ratio=sibilance_ratio,
        crest_factor_db=crest_factor_db,
        spectral_tilt_db_per_octave=spectral_tilt,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_analyze_signal tests/test_post_processor.py::test_signal_profile_properties -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: add SignalProfile with true-peak, crest factor, spectral tilt"
```

---

### Task 2: Add soft-knee limiter method

**Files:**
- Modify: `utilities/post_processor.py` (add method to `AudioPostProcessor` class, after `compress()` method ~line 560)
- Test: `tests/test_post_processor.py`

- [ ] **Step 1: Write the failing tests for soft-knee limiter**

Add to `tests/test_post_processor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_limit_peak -v`
Expected: FAIL with `AttributeError: 'AudioPostProcessor' object has no attribute 'limit_peak'`

- [ ] **Step 3: Implement soft-knee limiter**

Add to `AudioPostProcessor` class in `utilities/post_processor.py` (after the `compress()` method, around line ~600):

```python
def limit_peak(
    self,
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -1.0,
    attack_ms: float = 1.0,
    release_ms: float = 80.0,
    max_reduction_db: float = 6.0,
) -> tuple[np.ndarray, dict]:
    """Soft-knee peak limiter with envelope follower.

    Uses an envelope follower (attack/release smoothing) to apply
    gain reduction smoothly, avoiding the flat-top distortion of
    hard brick-wall clipping.
    """
    threshold_linear = 10 ** (threshold_db / 20)
    peak = np.max(np.abs(audio))

    if peak <= threshold_linear:
        return audio.copy(), {
            'limiting_applied': False,
            'peak_before': float(peak),
            'max_gain_reduction_db': 0.0,
        }

    # Envelope follower in sample domain
    attack_coeff = 1.0 - np.exp(-1.0 / (sr * attack_ms / 1000.0))
    release_coeff = 1.0 - np.exp(-1.0 / (sr * release_ms / 1000.0))

    envelope = np.zeros(len(audio), dtype=np.float64)
    envelope[0] = np.abs(audio[0])
    for i in range(1, len(audio)):
        abs_sample = abs(audio[i])
        coeff = attack_coeff if abs_sample > envelope[i - 1] else release_coeff
        envelope[i] = envelope[i - 1] + coeff * (abs_sample - envelope[i - 1])

    # Gain reduction: soft transition around threshold
    # gain = threshold / envelope when envelope > threshold
    gain_reduction = np.ones(len(audio), dtype=np.float64)
    over_mask = envelope > threshold_linear
    # Compute required gain, capped at max_reduction_db
    min_gain = 10 ** (-max_reduction_db / 20.0)
    gain_reduction[over_mask] = np.maximum(
        threshold_linear / envelope[over_mask], min_gain
    )

    processed = (audio.astype(np.float64) * gain_reduction).astype(np.float32)
    max_gr = float(-20 * np.log10(np.min(gain_reduction) + 1e-10))

    return processed, {
        'limiting_applied': True,
        'peak_before': float(peak),
        'peak_after': float(np.max(np.abs(processed))),
        'max_gain_reduction_db': round(max_gr, 2),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_limit_peak -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: add soft-knee limiter with envelope follower and 6dB cap"
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

# Soft-knee limiter defaults
LIMITER_THRESHOLD_DB = -1.0      # dBTP
LIMITER_ATTACK_MS = 1.0          # 0-2ms range
LIMITER_RELEASE_MS = 80.0        # 50-150ms range for speech
LIMITER_MAX_REDUCTION_DB = 6.0   # Cap to avoid aggressive pumping

# De-esser proportional scaling
DE_ESS_SIBILANCE_FLOOR = 0.15    # Below this ratio, no de-essing
DE_ESS_SIBILANCE_SCALE = 4.0     # Multiplier for proportional intensity
DE_ESS_MAX_REDUCTION_DB = -6.0   # Max gain reduction in sibilance band
```

- [ ] **Step 2: Verify no import errors**

Run: `.venv/Scripts/python -c "from utilities.app_constants import DEFAULT_TARGET_LOUDNESS_LUFS, LIMITER_MAX_REDUCTION_DB; print(f'LUFS={DEFAULT_TARGET_LOUDNESS_LUFS}, LimiterCap={LIMITER_MAX_REDUCTION_DB}')"`
Expected: `LUFS=-18.0, LimiterCap=6.0`

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
    # Proportional intensity should be reported
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
    2. Adaptive de-esser (proportional to sibilance ratio)
    3. Adaptive EQ (mud cut or presence boost, high-shelf for presence)
    4. Adaptive limiter (soft-knee, if peak > threshold)
    5. Loudness normalization (always, -18 LUFS)

    Emits a standardized manifest for regression tracing.

    Opt-in stages (default off):
    - Compressor, pitch shift, prosodic modulation, room presence,
      spectral enrichment
    """
    if not enable_post_processing:
        return audio.copy(), {}

    all_diagnostics = {}
    max_gain_reduction_db_seen = 0.0

    # Stage 0: Signal analysis
    profile = analyze_signal(audio, sr)
    all_diagnostics['signal_profile'] = {
        'peak': profile.peak,
        'true_peak_db': profile.true_peak_db,
        'rms': profile.rms,
        'spectral_centroid': profile.spectral_centroid,
        'sibilance_ratio': profile.sibilance_ratio,
        'crest_factor_db': profile.crest_factor_db,
        'spectral_tilt_db_per_octave': profile.spectral_tilt_db_per_octave,
        'needs_limiting': profile.needs_limiting,
        'needs_de_essing': profile.needs_de_essing,
        'needs_mud_cut': profile.needs_mud_cut,
        'needs_presence_boost': profile.needs_presence_boost,
    }

    # Stage 1: High-pass filter (always)
    b, a = signal.butter(2, 80 / (sr / 2), btype='high')
    audio = signal.filtfilt(b, a, audio).astype(np.float32)
    all_diagnostics['high_pass_filter'] = {'cutoff_hz': 80}

    # Stage 2: Adaptive de-esser (proportional to sibilance)
    if profile.needs_de_essing and de_ess_intensity > 0:
        # Proportional scaling: intensity = clamp((ratio - floor) * scale, 0, 1)
        from utilities.app_constants import DE_ESS_SIBILANCE_FLOOR, DE_ESS_SIBILANCE_SCALE
        proportional = min(max(
            (profile.sibilance_ratio - DE_ESS_SIBILANCE_FLOOR) * DE_ESS_SIBILANCE_SCALE,
            0.0
        ), 1.0)
        scaled_intensity = de_ess_intensity * proportional

        audio, de_ess_diag = self.de_esser(audio, sr, intensity=scaled_intensity)
        if de_ess_diag:
            de_ess_diag['proportional_intensity'] = round(proportional, 3)
            de_ess_diag['scaled_intensity'] = round(scaled_intensity, 3)
            all_diagnostics['de_esser'] = de_ess_diag

    # Stage 3: Adaptive EQ
    if profile.needs_mud_cut:
        b, a = self._design_peaking(300, 1.0, -2.0 * eq_intensity, sr)
        audio = signal.filtfilt(b, a, audio).astype(np.float32)
        all_diagnostics['adaptive_eq'] = {
            'action': 'mud_cut', 'frequency': 300, 'gain_db': -2.0 * eq_intensity,
        }
    elif profile.needs_presence_boost:
        # Use high-shelf for presence (gentler than peaking)
        b, a = signal.butter(2, 3500 / (sr / 2), btype='high')
        shelf_gain = 0.8 * eq_intensity  # +0.8 dB max
        audio = audio + shelf_gain * (signal.filtfilt(b, a, audio) * 0.1).astype(np.float32)
        all_diagnostics['adaptive_eq'] = {
            'action': 'presence_boost', 'type': 'high_shelf',
            'frequency': 3500, 'gain_db': 0.8 * eq_intensity,
        }

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
            max_gain_reduction_db_seen = max(
                max_gain_reduction_db_seen,
                comp_diag.get('max_gain_reduction_db', 0.0),
            )
            all_diagnostics['compressor'] = comp_diag
    elif profile.needs_limiting:
        audio, limiter_diag = self.limit_peak(audio, sr, threshold_db=-1.0)
        if limiter_diag:
            max_gain_reduction_db_seen = max(
                max_gain_reduction_db_seen,
                limiter_diag.get('max_gain_reduction_db', 0.0),
            )
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

    # Emit standardized manifest
    post_profile = analyze_signal(audio, sr)
    all_diagnostics['manifest'] = {
        'integrated_lufs': loudness_diag.get('loudness_lufs', 0.0) if loudness_diag else 0.0,
        'true_peak_db': post_profile.true_peak_db,
        'spectral_centroid_hz': round(post_profile.spectral_centroid, 1),
        'sibilance_ratio': round(post_profile.sibilance_ratio, 4),
        'crest_factor_db': round(post_profile.crest_factor_db, 1),
        'max_gain_reduction_db': round(max_gain_reduction_db_seen, 2),
    }

    return audio.astype(np.float32), all_diagnostics
```

- [ ] **Step 4: Run the new tests**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py::test_process_adaptive tests/test_post_processor.py::test_process_disabled tests/test_post_processor.py::test_process_target_lufs tests/test_post_processor.py::test_process_removed_stages_not_in_default_diagnostics tests/test_post_processor.py::test_process_always_runs tests/test_post_processor.py::test_process_proportional tests/test_post_processor.py::test_process_emits_manifest -v`
Expected: All PASS

- [ ] **Step 5: Run ALL existing tests to check for regressions**

Run: `.venv/Scripts/python -m pytest tests/test_post_processor.py -v`
Expected: All PASS. Existing tests for individual methods (de_esser, equalize, compress, etc.) should still pass since those methods are unchanged.

- [ ] **Step 6: Commit**

```bash
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: rewrite process() with signal-adaptive pipeline

- Signal analysis drives which stages run
- Soft-knee limiter replaces brick-wall clipping
- De-esser intensity scales proportionally with sibilance ratio
- Presence boost uses high-shelf instead of peaking EQ
- Standardized manifest emitted per sample for regression tracing
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

### Task 6: Update vocalization recipes — new tags and whisper fix

**Files:**
- Modify: `utilities/vocalization/recipes.json`

- [ ] **Step 1: Update whisper, whimper, moan recipes and add struggling tag**

Replace the `whispers` entry with band-pass filtering (add LPF):

```json
"whispers": {
    "tts_text": null,
    "mode": "modify_speech",
    "effects": [
      {"type": "pitch_shift", "semitones": -2},
      {"type": "high_pass_filter", "cutoff_hz": 600},
      {"type": "low_pass_filter", "cutoff_hz": 4000},
      {"type": "breath_noise", "amplitude": 0.15},
      {"type": "volume", "factor": 0.35}
    ]
}
```

Replace `whimpers` entry with spec parameters:

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

Replace `moans` entry with spec parameters:

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

### Task 7: Add vocalization distinctness check

**Files:**
- Create: `utilities/vocalization/distinctness_check.py`
- Create: `tests/test_distinctness_check.py`

Vocalizations must be audibly distinct from speech. This check compares a vocalization's spectral centroid and energy against a speech baseline and fails if they're too similar (meaning DSP didn't produce a meaningfully different sound).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distinctness_check.py`:

```python
import numpy as np
import pytest

from utilities.vocalization.distinctness_check import check_vocalization_distinctness


def _make_speech_like(sr=48000, duration=1.0):
    """Generate speech-like audio: fundamental + harmonics around 200-600Hz."""
    t = np.linspace(0, duration, int(sr * duration))
    return (0.3 * np.sin(2 * np.pi * 200 * t) + 0.15 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)


def _make_whisper_like(sr=48000, duration=1.0):
    """Generate whisper-like audio: band-limited noise in 600-4000Hz."""
    n = int(sr * duration)
    noise = np.random.randn(n).astype(np.float32) * 0.1
    b, a = __import__('scipy').signal.butter(4, [600 / (sr / 2), 4000 / (sr / 2)], btype='band')
    return __import__('scipy').signal.filtfilt(b, a, noise).astype(np.float32)


def test_distinct_vocalization_passes():
    """A whisper should be distinct enough from speech baseline."""
    speech = _make_speech_like()
    vocalization = _make_whisper_like()

    result = check_vocalization_distinctness(vocalization, speech, sr=48000)

    assert result['is_distinct'] is True


def test_identical_to_speech_fails():
    """Audio identical to speech should fail distinctness check."""
    speech = _make_speech_like()

    result = check_vocalization_distinctness(speech, speech, sr=48000)

    assert result['is_distinct'] is False


def test_similar_to_speech_fails():
    """Audio very similar to speech should fail."""
    speech = _make_speech_like()
    # Add tiny difference
    similar = speech + 0.001 * np.random.randn(len(speech)).astype(np.float32)

    result = check_vocalization_distinctness(similar, speech, sr=48000)

    assert result['is_distinct'] is False


def test_silence_handled():
    """Silent vocalization should not crash."""
    speech = _make_speech_like()
    silence = np.zeros(48000, dtype=np.float32)

    result = check_vocalization_distinctness(silence, speech, sr=48000)

    # Silence is distinct (very different energy) but edge case
    assert 'is_distinct' in result
    assert 'centroid_diff_hz' in result
    assert 'energy_ratio_db' in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_distinctness_check.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utilities.vocalization.distinctness_check'`

- [ ] **Step 3: Implement distinctness check**

Create `utilities/vocalization/distinctness_check.py`:

```python
"""Vocalization distinctness check — ensures vocalizations are audibly different from speech."""

import numpy as np
import scipy.signal as signal


def _spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid in Hz."""
    n = len(audio)
    fft_mag = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = np.sum(fft_mag) + 1e-10
    return float(np.sum(freqs * fft_mag) / total)


def _rms_energy_db(audio: np.ndarray) -> float:
    """Compute RMS energy in dB."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20 * np.log10(rms + 1e-10)


def check_vocalization_distinctness(
    vocalization: np.ndarray,
    speech_baseline: np.ndarray,
    sr: int = 48000,
    centroid_threshold_hz: float = 200.0,
    energy_threshold_db: float = 3.0,
) -> dict:
    """Check that a vocalization is spectrally distinct from speech.

    Fails if both centroid difference < threshold AND energy ratio within threshold,
    meaning the vocalization sounds too similar to normal speech.

    Args:
        vocalization: Processed vocalization audio
        speech_baseline: Reference speech audio for comparison
        sr: Sample rate
        centroid_threshold_hz: Min centroid difference to be distinct (Hz)
        energy_threshold_db: Max energy difference to be considered "similar" (dB)

    Returns:
        Dict with is_distinct bool, centroid_diff_hz, energy_ratio_db
    """
    voc_centroid = _spectral_centroid(vocalization, sr)
    speech_centroid = _spectral_centroid(speech_baseline, sr)
    centroid_diff = abs(voc_centroid - speech_centroid)

    voc_energy = _rms_energy_db(vocalization)
    speech_energy = _rms_energy_db(speech_baseline)
    energy_ratio_db = voc_energy - speech_energy

    # Not distinct if centroid is very close AND energy is very similar
    is_distinct = not (centroid_diff < centroid_threshold_hz and abs(energy_ratio_db) < energy_threshold_db)

    return {
        'is_distinct': is_distinct,
        'centroid_diff_hz': round(centroid_diff, 1),
        'energy_ratio_db': round(energy_ratio_db, 1),
        'voc_centroid_hz': round(voc_centroid, 1),
        'speech_centroid_hz': round(speech_centroid, 1),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_distinctness_check.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add utilities/vocalization/distinctness_check.py tests/test_distinctness_check.py
git commit -m "feat: add vocalization distinctness check (centroid + energy ratio)"
```

---

### Task 8: Add per-speaker baseline generation script

**Files:**
- Create: `utilities/generate_speaker_baselines.py`

Auto-generates per-speaker baseline JSONs from eval data. Each baseline stores 3 numbers: mean LUFS, spectral centroid, typical peak. Used by the adaptive pipeline to nudge thresholds per voice archetype.

- [ ] **Step 1: Implement baseline generator**

Create `utilities/generate_speaker_baselines.py`:

```python
"""Generate per-speaker baseline JSONs from audio samples.

Usage:
    python -m utilities.generate_speaker_baselines --speakers-dir speakers/en --output-dir baselines
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utilities.audio_utils import load_audio
from utilities.post_processor import analyze_signal


def generate_baseline(audio_path: Path) -> dict:
    """Analyze a speaker reference file and produce baseline stats."""
    audio, sr = load_audio(str(audio_path))
    profile = analyze_signal(audio, sr)
    return {
        'speaker': audio_path.stem,
        'mean_lufs': round(-20 * np.log10(profile.rms + 1e-10) - 4.0, 1),  # RMS-to-LUFS approx
        'spectral_centroid_hz': round(profile.spectral_centroid, 1),
        'peak': round(profile.peak, 4),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate per-speaker baseline stats')
    parser.add_argument('--speakers-dir', type=Path, default=Path('speakers/en'),
                        help='Directory containing speaker reference WAV files')
    parser.add_argument('--output-dir', type=Path, default=Path('baselines'),
                        help='Output directory for baseline JSON files')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for wav_file in sorted(args.speakers_dir.glob('*.wav')):
        baseline = generate_baseline(wav_file)
        out_path = args.output_dir / f"{wav_file.stem}_baseline.json"
        out_path.write_text(json.dumps(baseline, indent=2))
        print(f"  {wav_file.stem}: LUFS={baseline['mean_lufs']}, "
              f"centroid={baseline['spectral_centroid_hz']}Hz, peak={baseline['peak']}")

    print(f"\nGenerated {len(list(args.output_dir.glob('*.json')))} baselines in {args.output_dir}/")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify the script loads**

Run: `.venv/Scripts/python -c "from utilities.generate_speaker_baselines import generate_baseline; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add utilities/generate_speaker_baselines.py
git commit -m "feat: add per-speaker baseline JSON generator for adaptive thresholds"
```

---

### Task 9: Add NSFW vocalization test cases to audio quality eval

**Files:**
- Modify: `tests/audio_quality/test_full_eval.py` (the `VOCALIZATION_CASES` list)

- [ ] **Step 1: Add new vocalization test cases**

Update `VOCALIZATION_CASES` in `tests/audio_quality/test_full_eval.py`:

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

### Task 10: Run full test suite and lint

**Files:** None — verification only

- [ ] **Step 1: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -v --ignore=tests/audio_quality -x`
Expected: All tests PASS. The `--ignore=tests/audio_quality` skips GPU tests.

- [ ] **Step 2: Run lint**

Run: `.venv/Scripts/python -m ruff check utilities/post_processor.py utilities/app_constants.py utilities/vocalization/recipes.json utilities/vocalization/distinctness_check.py utilities/generate_speaker_baselines.py tests/test_post_processor.py tests/test_distinctness_check.py`
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

### Task 11: Run audio quality evaluation (requires GPU)

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

- [ ] **Step 4: Generate per-speaker baselines**

Run: `.venv/Scripts/python -m utilities.generate_speaker_baselines`
Expected: JSON files generated for each speaker in `baselines/`

- [ ] **Step 5: Save as new baseline**

```bash
cp tests/audio_quality/baselines/latest_scores.json tests/audio_quality/baselines/master_raw_baseline.json
```

- [ ] **Step 6: Commit baseline**

```bash
git add tests/audio_quality/baselines/ baselines/
git commit -m "feat: update audio quality baseline with signal-adaptive pipeline scores"
```

---

### Task 12: Update decision records

**Files:** None — repowise tool only

- [ ] **Step 1: Record the architectural decision**

Use repowise `update_decision_records` to create a record of this pipeline change.

Action: `create`
Title: `Signal-Adaptive Post-Processing Pipeline`
Decision: `Replaced uniform 8-stage post-processing with 6-stage signal-adaptive pipeline. SignalProfile measures peak, RMS, true-peak dBTP, spectral centroid, sibilance ratio, crest factor, spectral tilt to drive adaptive decisions. Soft-knee limiter replaces brick-wall clipping (0-2ms attack, 50-150ms release, 6dB max reduction). De-esser intensity scales proportionally with sibilance ratio. Presence boost uses high-shelf EQ instead of peaking. Standardized manifest JSON emitted per sample. Per-speaker baseline JSONs auto-generated from eval data. Vocalization distinctness check ensures vocalizations are audibly different from speech. Removed spectral enrichment, room presence, prosodic modulation, and auto pitch shift from default chain (opt-in via enable_* params). LUFS target changed from -16 to -18.`
Rationale: `Audio quality evaluation showed 30% average OVRL degradation from post-processing. External audio engineer review confirmed soft-knee limiting, proportional de-essing, true-peak measurement, crest factor monitoring, per-speaker baselines, manifest standardization, and vocalization distinctness checks as necessary additions. Spectral enrichment and room reverb are harmful for clean dialogue. SkyrimNet GamePlugin bypasses Skyrim's sound system, so room reverb must come from SkyrimNet's voice effects, not TTS output.`
Affected_files: `["utilities/post_processor.py", "utilities/app_constants.py", "utilities/vocalization/recipes.json", "utilities/vocalization/distinctness_check.py", "utilities/generate_speaker_baselines.py"]`
Tags: `["audio", "post-processing", "pipeline"]`

- [ ] **Step 2: Update existing DSP pipeline decision status**

Use repowise `update_decision_records` with action `update_status`:
Decision ID: `ca5ca79d73c1498ea1e60335c7098fcb` (the existing "DSP Post-Processing Pipeline" decision)
Status: `superseded`
Superseded by: the new decision ID from step 1

---

## Self-Review

**1. Spec coverage check:**
- SignalProfile (peak, RMS, true-peak, centroid, sibilance, crest factor, tilt): Task 1 ✓
- HPF stage: Task 4 (inside process()) ✓
- Proportional de-esser: Task 4 ✓
- High-shelf presence boost: Task 4 ✓
- Soft-knee limiter: Task 2 ✓
- Loudness normalization -18 LUFS: Task 3 (constant) + Task 4 (process) ✓
- Removed stages opt-in: Task 4 (enable_* params) ✓
- Standardized manifest: Task 4 ✓
- Per-speaker baselines: Task 8 ✓
- Vocalization distinctness check: Task 7 ✓
- NSFW vocalization tags: Task 6 ✓
- Whisper LPF update: Task 6 ✓
- SkyrimNet responsibility split: Addressed by removing room reverb from default ✓
- Testing strategy: Task 10 + Task 11 ✓
- Decision records: Task 12 ✓

**2. Placeholder scan:** No TBDs, TODOs, or "implement later" patterns found.

**3. Type consistency:**
- `SignalProfile` dataclass defined in Task 1, used consistently in Tasks 2, 4, 8
- `analyze_signal()` returns `SignalProfile` — used in both tests and process()
- `limit_peak()` returns `tuple[np.ndarray, dict]` — matches other methods
- `check_vocalization_distinctness()` returns `dict` — used in Task 7 tests
- `generate_baseline()` returns `dict` — used in Task 8
- Default parameter `de_ess_intensity=0.3` matches `DEFAULT_DE_ESS_INTENSITY`
- Default parameter `target_loudness=-18.0` matches `DEFAULT_TARGET_LOUDNESS_LUFS`
