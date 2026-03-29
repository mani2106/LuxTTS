"""Audio post-processing pipeline for LuxTTS output quality improvement."""

import logging
import re
from typing import Optional

import librosa
import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)

# Optional dependency detection
try:
    import pedalboard
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

try:
    import pyloudnorm as pyln
    HAS_PLOUDNORM = True
except ImportError:
    HAS_PLOUDNORM = False


class PitchDetector:
    """
    Detects pitch offset from dialogue text using heuristic pattern matching.

    Rules evaluated in order (first match wins):
    1. Manual override provided → use provided value
    2. Text ALL CAPS (majority) → +2.0 (shouting)
    3. Text ends with '!' → +1.0 (excited)
    4. Text ends with '?' → +0.5 (questioning)
    5. Text contains '...' → -1.0 (hesitant/sad)
    6. Text length > 200 chars → -0.5 (narrative/calm)
    7. Default → 0.0 (neutral)
    """

    ALL_CAPS_THRESHOLD = 0.7  # 70% of characters must be uppercase
    LONG_TEXT_THRESHOLD = 200  # characters

    def detect_pitch(self, text: str, manual_pitch_shift: Optional[float] = None) -> float:
        """
        Detect pitch offset from dialogue text.

        Args:
            text: Dialogue text to analyze
            manual_pitch_shift: Manual override in semitones. If provided,
                all text analysis is skipped.

        Returns:
            Pitch offset in semitones (positive = higher, negative = lower)
        """
        # Manual override takes precedence
        if manual_pitch_shift is not None:
            return float(manual_pitch_shift)

        if not text:
            return 0.0

        # Rule: ALL CAPS majority → shouting
        if self._is_all_caps(text):
            return 2.0

        # Rule: Ends with exclamation → excited
        if text.rstrip().endswith('!'):
            return 1.0

        # Rule: Ends with question → questioning
        if text.rstrip().endswith('?'):
            return 0.5

        # Rule: Contains ellipsis → hesitant
        if '...' in text:
            return -1.0

        # Rule: Long text → narrative/calm
        if len(text) > self.LONG_TEXT_THRESHOLD:
            return -0.5

        # Default: neutral
        return 0.0

    def _is_all_caps(self, text: str) -> bool:
        """Check if majority of alphanumeric characters are uppercase."""
        # Remove whitespace and punctuation for analysis
        alnum_chars = [c for c in text if c.isalnum()]
        if not alnum_chars:
            return False

        uppercase_count = sum(1 for c in alnum_chars if c.isupper())
        ratio = uppercase_count / len(alnum_chars)
        return ratio >= self.ALL_CAPS_THRESHOLD


class AudioPostProcessor:
    """
    Composable audio post-processing pipeline.

    Each stage returns (processed_audio, diagnostics_dict).
    Diagnostics are empty dict when return_diagnostics=False.
    """

    def __init__(self, return_diagnostics: bool = False):
        """
        Initialize the post-processor.

        Args:
            return_diagnostics: If True, return diagnostic metrics from each stage.
        """
        self.return_diagnostics = return_diagnostics

    def de_esser(
        self,
        audio: np.ndarray,
        sr: int,
        intensity: float = 0.5,
    ) -> tuple[np.ndarray, dict]:
        """
        Reduce sibilance (harsh 's' and 'sh' sounds).

        Uses band-pass filtered envelope detection in 5-8kHz range with
        adaptive threshold relative to signal RMS.

        Args:
            audio: Input audio (float32, 48kHz)
            sr: Sample rate (should be 48000)
            intensity: De-essing strength (0.0 = bypass, 1.0 = full)

        Returns:
            (processed_audio, diagnostics_dict)
        """
        # Bypass if intensity is zero, regardless of available backends
        if intensity <= 0.0:
            return audio.copy(), {}

        # Use pedalboard if available, otherwise native scipy
        if HAS_PEDALBOARD:
            return self._de_esser_pedalboard(audio, sr, intensity)

        return self._de_esser_native(audio, sr, intensity)

    def _de_esser_native(
        self,
        audio: np.ndarray,
        sr: int,
        intensity: float,
    ) -> tuple[np.ndarray, dict]:
        """Native scipy implementation of de-essing."""
        # Design band-pass filter for sibilant range (5-8kHz)
        low = 5000 / (sr / 2)
        high = 8000 / (sr / 2)
        b, a = signal.butter(4, [low, high], btype='band')

        # Extract sibilant band
        sibilant = signal.filtfilt(b, a, audio)

        # Compute adaptive threshold from signal RMS
        signal_rms = np.sqrt(np.mean(audio ** 2))
        threshold = signal_rms * (0.3 + intensity * 0.4)

        # Envelope follower (simple rectification + low-pass)
        envelope = np.abs(sibilant)
        # Smooth envelope with attack/release (simplified as moving average)
        window_size = int(0.01 * sr)  # 10ms
        if window_size < 1:
            window_size = 1
        envelope_padded = np.pad(envelope, (window_size // 2, window_size // 2), mode='edge')
        kernel = np.ones(window_size) / window_size
        smooth_envelope = np.convolve(envelope_padded, kernel, mode='same')[:len(envelope)]

        # Compute gain reduction where envelope exceeds threshold
        gain_reduction = np.ones_like(audio)
        mask = smooth_envelope > threshold
        # Reduce gain proportionally to how much we exceed threshold
        excess = smooth_envelope[mask] - threshold
        reduction = 1.0 - (excess / (threshold + excess)) * intensity * 0.5
        gain_reduction[mask] = np.clip(reduction, 0.3, 1.0)

        # Apply gain reduction
        processed = audio * gain_reduction

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['reduction_db_curve'] = 20 * np.log10(gain_reduction + 1e-10)

        return processed.astype(np.float32), diagnostics

    def _de_esser_pedalboard(
        self,
        audio: np.ndarray,
        sr: int,
        intensity: float,
    ) -> tuple[np.ndarray, dict]:
        """Pedalboard implementation of de-essing."""
        # Map intensity (0-1) to pedalboard parameters
        # pedalboard.DeEsser uses frequency (Hz) and threshold (dB)
        de_esser = pedalboard.DeEsser()

        processed = de_esser(audio, sr)

        diagnostics = {}
        if self.return_diagnostics:
            # Pedalboard doesn't expose gain curve, use placeholder
            diagnostics['reduction_db_curve'] = np.zeros(len(audio) // 100)  # Downsampled

        return processed.astype(np.float32), diagnostics

    def equalize(
        self,
        audio: np.ndarray,
        sr: int,
        intensity: float = 1.0,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply EQ to tame metallic harshness and add vocal warmth.

        - High-shelf cut: -6dB at 8kHz (Q=0.7)
        - Presence peak: +2dB at 3kHz (Q=1.0)
        Both scaled by intensity.

        Returns:
            (processed_audio, diagnostics_dict)
        """
        if intensity <= 0.0:
            return audio.copy(), {}

        if HAS_PEDALBOARD:
            return self._equalize_pedalboard(audio, sr, intensity)

        return self._equalize_native(audio, sr, intensity)

    def _equalize_native(
        self,
        audio: np.ndarray,
        sr: int,
        intensity: float,
    ) -> tuple[np.ndarray, dict]:
        """Native scipy implementation of EQ."""
        diagnostics = {}
        if self.return_diagnostics:
            freqs, pre_spec = signal.welch(audio, sr, nperseg=2048)
            diagnostics['pre_spectrum'] = (freqs, pre_spec)

        # High-shelf low-pass at 8kHz to tame metallic artifacts
        fc = 8000
        Q = 0.7
        gain_db = -6.0 * intensity
        b1, a1 = self._design_peaking(fc, Q, gain_db, sr)

        # Presence peak at 3kHz for warmth
        fc = 3000
        Q = 1.0
        gain_db = 2.0 * intensity
        b2, a2 = self._design_peaking(fc, Q, gain_db, sr)

        # Apply filters in series
        processed = signal.filtfilt(b1, a1, audio)
        processed = signal.filtfilt(b2, a2, processed)

        if self.return_diagnostics:
            freqs, post_spec = signal.welch(processed, sr, nperseg=2048)
            diagnostics['post_spectrum'] = (freqs, post_spec)

        return processed.astype(np.float32), diagnostics

    def _equalize_pedalboard(
        self,
        audio: np.ndarray,
        sr: int,
        intensity: float,
    ) -> tuple[np.ndarray, dict]:
        """Pedalboard implementation of EQ."""
        high_shelf = pedalboard.HighShelfFilter(
            cutoff_frequency_hz=8000,
            gain_db=-6.0 * intensity,
            q=0.7,
        )
        peak = pedalboard.PeakFilter(
            cutoff_frequency_hz=3000,
            gain_db=2.0 * intensity,
            q=1.0,
        )

        chain = pedalboard.Pedalboard([high_shelf, peak])
        processed = chain(audio, sr)

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['pre_spectrum'] = (np.array([0]), np.array([0]))
            diagnostics['post_spectrum'] = (np.array([0]), np.array([0]))

        return processed.astype(np.float32), diagnostics

    def pitch_shift(
        self,
        audio: np.ndarray,
        sr: int,
        n_steps: float,
    ) -> tuple[np.ndarray, dict]:
        """
        Shift pitch by n_steps semitones.

        Uses librosa.effects.pitch_shift() which time-stretches the audio
        to preserve duration while changing pitch.

        Args:
            audio: Input audio (float32, any sample rate)
            sr: Sample rate
            n_steps: Pitch shift in semitones (positive = higher, negative = lower)

        Returns:
            (processed_audio, diagnostics_dict) where diagnostics contains
            'semitones_applied': float value of semitones shifted
        """
        # Bypass for near-zero pitch shifts to avoid unnecessary processing
        if abs(n_steps) < 0.01:
            return audio.copy(), {}

        # Apply pitch shift using librosa
        processed = librosa.effects.pitch_shift(
            y=audio.astype(np.float64),
            sr=sr,
            n_steps=n_steps,
        )

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['semitones_applied'] = float(n_steps)

        return processed.astype(np.float32), diagnostics

    @staticmethod
    def _design_peaking(fc: float, Q: float, gain_db: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Design a peaking EQ filter using scipy.signal.iirfilter."""
        # Convert gain from dB to linear
        A = 10 ** (gain_db / 40)

        # Angular frequency
        w0 = 2 * np.pi * fc / sr

        # Compute filter coefficients (biquad peaking EQ)
        alpha = np.sin(w0) / (2 * Q)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        # Normalize by a0
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        return b, a

    def compress(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_offset_db: float = -6.0,
        ratio: float = 4.0,
        knee_db: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        max_reduction_db: float = 12.0,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply dynamic range compression to even out volume inconsistencies.

        Features:
        - Soft-knee curve for smooth gain reduction transitions
        - Adaptive threshold from short-term RMS (200ms window)
        - Look-ahead (3ms) on envelope detector
        - Detector on low-pass filtered signal (3-6kHz bandpass)
        - Makeup gain from loudness difference
        - Hard limiter at -1 dBTP

        Args:
            audio: Input audio (float32, 48kHz)
            sr: Sample rate (should be 48000)
            threshold_offset_db: Threshold offset from signal RMS in dB
            ratio: Compression ratio (e.g., 4.0 = 4:1)
            knee_db: Soft-knee width in dB (3-6dB typical)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            max_reduction_db: Maximum gain reduction per frame in dB

        Returns:
            (processed_audio, diagnostics_dict)
        """
        if len(audio) == 0:
            return audio.copy(), {}

        # Use pedalboard if available, otherwise native scipy
        if HAS_PEDALBOARD:
            return self._compress_pedalboard(
                audio, sr, threshold_offset_db, ratio, knee_db,
                attack_ms, release_ms, max_reduction_db
            )

        return self._compress_native(
            audio, sr, threshold_offset_db, ratio, knee_db,
            attack_ms, release_ms, max_reduction_db
        )

    def _compress_native(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_offset_db: float,
        ratio: float,
        knee_db: float,
        attack_ms: float,
        release_ms: float,
        max_reduction_db: float,
    ) -> tuple[np.ndarray, dict]:
        """Native scipy implementation of compression with soft-knee and adaptive threshold."""
        # Edge case: very short utterance - reduce attack/look-ahead
        duration_ms = len(audio) / sr * 1000
        if duration_ms < 500:
            attack_ms = min(attack_ms, duration_ms / 10)
            look_ahead_ms = min(3.0, duration_ms / 20)
        else:
            look_ahead_ms = 3.0

        # Step 1: Low-pass filter for detector (3-6kHz bandpass)
        # This helps detector respond to perceived loudness rather than HF artifacts
        low = 3000 / (sr / 2)
        high = 6000 / (sr / 2)
        b, a = signal.butter(2, [low, high], btype='band')
        detector_signal = signal.filtfilt(b, a, audio)

        # Step 2: Compute adaptive threshold from short-term RMS (200ms window)
        window_samples = int(0.2 * sr)
        if window_samples < 1:
            window_samples = 1

        # Vectorized RMS computation using convolution
        squared = detector_signal ** 2
        window = np.ones(window_samples) / window_samples
        # Pad to handle edges
        squared_padded = np.pad(squared, (window_samples // 2, window_samples // 2), mode='edge')
        rms_squared = np.convolve(squared_padded, window, mode='same')[:len(squared)]
        rms = np.sqrt(np.maximum(rms_squared, 1e-10))

        # Convert RMS to dB
        rms_db = 20 * np.log10(rms + 1e-10)

        # Adaptive threshold with offset
        threshold_db = rms_db + threshold_offset_db

        # Clamp threshold for very quiet utterances to avoid over-compression
        min_threshold_db = -40.0  # Don't go below -40dB
        threshold_db = np.maximum(threshold_db, min_threshold_db)

        # Step 3: Look-ahead on envelope detector
        look_samples = int(look_ahead_ms * sr / 1000)
        if look_samples > 0:
            # Shift RMS backward to create look-ahead
            rms_db = np.roll(rms_db, look_samples)
            # Pad beginning with original values
            rms_db[:look_samples] = rms_db[look_samples]

        # Step 4: Compute gain reduction with soft-knee curve
        # Soft-knee transition around threshold
        knee_half = knee_db / 2

        # Input level above threshold
        input_db = rms_db
        excess_db = input_db - threshold_db

        # Soft-knee gain computation (vectorized)
        # Below threshold - knee/2: no reduction
        # In knee region: gradual reduction
        # Above threshold + knee/2: linear reduction with ratio

        gain_reduction_db = np.zeros_like(excess_db)

        # Below knee region: no reduction
        below_knee = excess_db <= -knee_half
        gain_reduction_db[below_knee] = 0.0

        # In knee region: smooth transition
        in_knee = (excess_db > -knee_half) & (excess_db <= knee_half)
        if np.any(in_knee):
            # Parabolic smooth curve in knee region
            x = excess_db[in_knee] + knee_half
            # Gain reduction follows smooth curve from 0 to knee_half * (1 - 1/ratio)
            knee_range = knee_half * (1 - 1 / ratio)
            gain_reduction_db[in_knee] = (knee_range / (2 * knee_half ** 2)) * x ** 2

        # Above knee region: linear reduction
        above_knee = excess_db > knee_half
        if np.any(above_knee):
            # Standard compression formula: (excess - knee/2) * (1 - 1/ratio) + knee_reduction
            knee_reduction = knee_half * (1 - 1 / ratio)
            gain_reduction_db[above_knee] = knee_reduction + (excess_db[above_knee] - knee_half) * (1 - 1 / ratio)

        # Apply max gain reduction limit
        gain_reduction_db = np.minimum(gain_reduction_db, max_reduction_db)

        # Step 5: Convert dB gain reduction to linear gain
        gain_linear = 10 ** (-gain_reduction_db / 20)

        # Step 6: Apply attack/release smoothing to gain
        # Convert attack/release from ms to coefficient
        attack_coeff = np.exp(-1.0 / (attack_ms * sr / 1000))
        release_coeff = np.exp(-1.0 / (release_ms * sr / 1000))

        # Vectorized envelope following with attack/release
        smoothed_gain = np.ones_like(gain_linear)
        prev_gain = 1.0

        for i in range(len(gain_linear)):
            target_gain = gain_linear[i]
            if target_gain < prev_gain:
                # Attack phase (gain decreasing)
                coeff = attack_coeff
            else:
                # Release phase (gain increasing)
                coeff = release_coeff

            smoothed_gain[i] = coeff * prev_gain + (1 - coeff) * target_gain
            prev_gain = smoothed_gain[i]

        # Step 7: Apply gain reduction to audio
        processed = audio * smoothed_gain

        # Step 8: Compute makeup gain from loudness difference
        input_lufs = self._compute_lufs(audio, sr)
        output_lufs = self._compute_lufs(processed, sr)
        makeup_gain_db = input_lufs - output_lufs

        # Apply makeup gain (capped to avoid excessive boosting)
        makeup_gain_db = min(makeup_gain_db, 6.0)  # Max 6dB makeup
        makeup_gain_linear = 10 ** (makeup_gain_db / 20)
        processed = processed * makeup_gain_linear

        # Step 9: Hard limiter at -1 dBTP
        peak_limit = 10 ** (-1.0 / 20)  # -1 dB
        processed = np.clip(processed, -peak_limit, peak_limit)

        # Ensure output is float32
        processed = processed.astype(np.float32)

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['input_lufs'] = input_lufs
            diagnostics['output_lufs'] = output_lufs
            diagnostics['max_reduction_db'] = float(np.max(gain_reduction_db))
            diagnostics['makeup_gain_db'] = makeup_gain_db
            # Downsample gain curve for storage
            downsample_factor = max(1, len(smoothed_gain) // 1000)
            diagnostics['gain_curve'] = 20 * np.log10(smoothed_gain[::downsample_factor] + 1e-10)

        return processed, diagnostics

    def _compress_pedalboard(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_offset_db: float,
        ratio: float,
        knee_db: float,
        attack_ms: float,
        release_ms: float,
        max_reduction_db: float,
    ) -> tuple[np.ndarray, dict]:
        """Pedalboard implementation of compression."""
        # Convert threshold_offset_db to absolute threshold
        # Compute RMS to get reference level
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        threshold_db = rms_db + threshold_offset_db

        # Pedalboard.Compressor uses threshold_db parameter
        compressor = pedalboard.Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )

        processed = compressor(audio, sr)

        # Apply hard limiter at -1 dBTP
        peak_limit = 10 ** (-1.0 / 20)
        processed = np.clip(processed, -peak_limit, peak_limit)

        diagnostics = {}
        if self.return_diagnostics:
            input_lufs = self._compute_lufs(audio, sr)
            output_lufs = self._compute_lufs(processed, sr)
            diagnostics['input_lufs'] = input_lufs
            diagnostics['output_lufs'] = output_lufs
            diagnostics['max_reduction_db'] = 0.0  # Pedalboard doesn't expose this
            diagnostics['makeup_gain_db'] = 0.0
            diagnostics['gain_curve'] = np.zeros(1000)  # Placeholder

        return processed.astype(np.float32), diagnostics

    @staticmethod
    def _compute_lufs(audio: np.ndarray, sr: int) -> float:
        """
        Compute integrated loudness in LUFS.

        Uses pyloudnorm if available for ITU-R BS.1770 compliance,
        otherwise falls back to RMS-based approximation.

        Args:
            audio: Input audio (float32)
            sr: Sample rate

        Returns:
            Loudness in LUFS (negative values, e.g., -16.0)
        """
        if HAS_PLOUDNORM and len(audio) > 0:
            try:
                # Use pyloudnorm for proper LUFS measurement
                meter = pyln.Meter(sr)  # Create BS.1770 meter
                lufs = meter.integrated_loudness(audio)
                return float(lufs)
            except Exception:
                # Fall back to RMS if pyloudnorm fails
                pass

        # RMS-based approximation (not true LUFS, but reasonable fallback)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return -100.0  # Silence

        # Approximate LUFS from RMS (calibration factor based on typical speech)
        rms_db = 20 * np.log10(rms)
        # This is a rough approximation; true LUFS uses K-weighting
        return rms_db - 3.0  # Typical offset for speech material

    def normalize_loudness(
        self,
        audio: np.ndarray,
        sr: int,
        target_lufs: float = -16.0,
    ) -> tuple[np.ndarray, dict]:
        """
        Normalize audio to target LUFS with hard peak limiting.

        Measures current loudness using pyloudnorm (ITU-R BS.1770-4) if available,
        otherwise falls back to RMS-based estimation. Applies gain to reach target
        LUFS, then enforces -1 dBTP peak limit to prevent clipping.

        Args:
            audio: Input audio (float32, any sample rate)
            sr: Sample rate
            target_lufs: Target loudness in LUFS (default: -16.0 for speech)

        Returns:
            (processed_audio, diagnostics_dict)
            Diagnostics contain:
                - measured_lufs: Measured loudness before processing (LUFS)
                - gain_applied_db: Gain applied to reach target (dB)
                - peak_limiting_applied: Whether peak limiter was triggered (bool)
        """
        if HAS_PLOUDNORM:
            measured_lufs = self._measure_loudness_pyloudnorm(audio, sr)
        else:
            measured_lufs = self._measure_loudness_rms_fallback(audio)

        # Calculate gain needed to reach target
        gain_db = target_lufs - measured_lufs
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain
        processed = audio * gain_linear

        # Hard peak limiter at -1 dBTP (0.89 linear)
        peak_limit_linear = 10 ** (-1.0 / 20)
        peak_limiting_applied = False

        if np.max(np.abs(processed)) > peak_limit_linear:
            peak_limiting_applied = True
            # Soft clipping with hard limit at threshold
            processed = np.clip(processed, -peak_limit_linear, peak_limit_linear)

        diagnostics = {
            'measured_lufs': float(measured_lufs),
            'gain_applied_db': float(gain_db),
            'peak_limiting_applied': peak_limiting_applied,
        }

        return processed.astype(np.float32), diagnostics

    def _measure_loudness_pyloudnorm(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure loudness using pyloudnorm (ITU-R BS.1770-4).

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate

        Returns:
            Loudness in LUFS
        """
        import pyloudnorm as pyln

        # Ensure 2D array for stereo compatibility (samples, channels)
        if audio.ndim == 1:
            audio_2d = audio.reshape(-1, 1)
        else:
            audio_2d = audio.T  # Convert to (samples, channels)

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_2d)

        return float(loudness)

    @staticmethod
    def _measure_loudness_rms_fallback(audio: np.ndarray) -> float:
        """
        Estimate loudness using RMS level (fallback when pyloudnorm unavailable).

        Note: This is a rough approximation. True LUFS measurement requires
        frequency-weighted filtering (K-weighting) and gating as specified in
        ITU-R BS.1770-4. RMS correlates roughly with LUFS but is not equivalent.

        Args:
            audio: Input audio (mono or stereo)

        Returns:
            Estimated loudness in LUFS-equivalent units
        """
        # Calculate RMS level
        if audio.ndim == 1:
            rms = np.sqrt(np.mean(audio ** 2))
        else:
            # Multi-channel: average RMS across channels
            rms_per_channel = np.sqrt(np.mean(audio ** 2, axis=0))
            rms = np.mean(rms_per_channel)

        # Convert RMS to dB (reference: 1.0)
        rms_db = 20 * np.log10(rms + 1e-10)

        # Rough conversion from RMS dB to LUFS-equivalent
        # This is an approximation; typical speech at -16 LUFS has RMS around -20 dB
        # We shift by ~4 dB to get a reasonable LUFS estimate
        estimated_lufs = rms_db - 4.0

        return float(estimated_lufs)

    def compressor(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
    ) -> tuple[np.ndarray, dict]:
        """
        Apply dynamic range compression to reduce volume peaks.

        Uses smooth gain following with configurable attack/release times.

        Args:
            audio: Input audio (float32, 48kHz)
            sr: Sample rate (should be 48000)
            threshold_db: Level above which compression activates (dBFS)
            ratio: Compression ratio (e.g., 4.0 = 4:1, input needs 4dB for 1dB output)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds

        Returns:
            (processed_audio, diagnostics_dict)
        """
        # If threshold is very high (e.g., 0 dB or above), bypass compression
        if threshold_db >= 0.0:
            return audio.copy(), {}

        if HAS_PEDALBOARD:
            return self._compressor_pedalboard(audio, sr, threshold_db, ratio, attack_ms, release_ms)

        return self._compressor_native(audio, sr, threshold_db, ratio, attack_ms, release_ms)

    def _compressor_native(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> tuple[np.ndarray, dict]:
        """Native scipy implementation of compression."""
        # Convert attack/release from ms to samples
        attack_samples = int(attack_ms * sr / 1000)
        release_samples = int(release_ms * sr / 1000)

        # Compute input signal level (RMS with smoothing)
        window_size = max(1, int(0.01 * sr))  # 10ms window
        kernel = np.ones(window_size) / window_size

        # Compute squared signal for RMS
        squared = audio ** 2
        squared_padded = np.pad(squared, (window_size // 2, window_size // 2), mode='edge')
        rms_squared = np.convolve(squared_padded, kernel, mode='same')[:len(squared)]
        rms = np.sqrt(np.maximum(rms_squared, 1e-10))

        # Convert to dB
        signal_db = 20 * np.log10(rms + 1e-10)

        # Compute gain reduction
        # Above threshold: gain = threshold + (signal - threshold) / ratio
        # Below threshold: gain = signal (no reduction)
        gain_db = np.copy(signal_db)
        above_threshold = signal_db > threshold_db
        gain_db[above_threshold] = threshold_db + (signal_db[above_threshold] - threshold_db) / ratio

        # Compute required gain reduction in dB
        gain_reduction_db = signal_db - gain_db

        # Smooth gain reduction with attack/release
        smoothed_gain_reduction = np.zeros_like(gain_reduction_db)
        current_gain = 0.0

        for i in range(len(gain_reduction_db)):
            target_gain = gain_reduction_db[i]

            if target_gain > current_gain:
                # Attack: gain increasing (more compression)
                coef = np.exp(-1.0 / attack_samples)
            else:
                # Release: gain decreasing (less compression)
                coef = np.exp(-1.0 / release_samples)

            current_gain = coef * current_gain + (1 - coef) * target_gain
            smoothed_gain_reduction[i] = current_gain

        # Apply gain reduction
        gain_linear = 10 ** (-smoothed_gain_reduction / 20)
        processed = audio * gain_linear

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['gain_reduction_db_curve'] = smoothed_gain_reduction

        return processed.astype(np.float32), diagnostics

    def _compressor_pedalboard(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> tuple[np.ndarray, dict]:
        """Pedalboard implementation of compression."""
        compressor = pedalboard.Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )

        processed = compressor(audio, sr)

        diagnostics = {}
        if self.return_diagnostics:
            # Pedalboard doesn't expose gain curve, use placeholder
            diagnostics['gain_reduction_db_curve'] = np.zeros(len(audio) // 100)

        return processed.astype(np.float32), diagnostics

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        text: Optional[str] = None,
        pitch_shift: Optional[float] = None,
        eq_intensity: float = 1.0,
        de_ess_intensity: float = 0.5,
        compressor_threshold_db: float = -20.0,
        compressor_ratio: float = 4.0,
        compressor_attack_ms: float = 5.0,
        compressor_release_ms: float = 50.0,
        target_loudness: float = -16.0,
        enable_post_processing: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Process audio through the full post-processing chain.

        Processing order:
        1. De-esser (reduce sibilance)
        2. EQ (tame harshness, add warmth)
        3. Compressor (reduce dynamic range)
        4. Pitch shift (adjust pitch)
        5. Normalize loudness (EBU R128)

        Args:
            audio: Input audio (float32, typically 48kHz)
            sr: Sample rate
            text: Dialogue text for automatic pitch detection (used if pitch_shift is None)
            pitch_shift: Manual pitch override in semitones. If None, auto-detected from text.
            eq_intensity: EQ processing intensity (0.0 = bypass, 1.0 = full)
            de_ess_intensity: De-essing intensity (0.0 = bypass, 1.0 = full)
            compressor_threshold_db: Compression threshold (dBFS, negative values)
            compressor_ratio: Compression ratio (e.g., 4.0 = 4:1)
            compressor_attack_ms: Attack time in milliseconds
            compressor_release_ms: Release time in milliseconds
            target_loudness: Target loudness in LUFS
            enable_post_processing: If False, bypass all processing and return original audio

        Returns:
            (processed_audio, diagnostics_dict)
            - processed_audio: The final processed audio
            - diagnostics_dict: Nested dict with per-stage diagnostics
                {
                    'de_esser': {...},
                    'equalize': {...},
                    'compressor': {...},
                    'pitch_shift': {...},
                    'normalize_loudness': {...}
                }
        """
        # Bypass if disabled
        if not enable_post_processing:
            return audio.copy(), {}

        # Initialize diagnostics
        all_diagnostics = {}

        # Stage 1: De-esser
        audio, de_ess_diagnostics = self.de_esser(audio, sr, intensity=de_ess_intensity)
        if de_ess_diagnostics:
            all_diagnostics['de_esser'] = de_ess_diagnostics

        # Stage 2: EQ
        audio, eq_diagnostics = self.equalize(audio, sr, intensity=eq_intensity)
        if eq_diagnostics:
            all_diagnostics['equalize'] = eq_diagnostics

        # Stage 3: Compressor
        audio, compressor_diagnostics = self.compressor(
            audio, sr,
            threshold_db=compressor_threshold_db,
            ratio=compressor_ratio,
            attack_ms=compressor_attack_ms,
            release_ms=compressor_release_ms,
        )
        if compressor_diagnostics:
            all_diagnostics['compressor'] = compressor_diagnostics

        # Stage 4: Pitch shift
        # Detect pitch from text if not provided
        if pitch_shift is None and text:
            detector = PitchDetector()
            detected_pitch = detector.detect_pitch(text)
        else:
            detected_pitch = pitch_shift if pitch_shift is not None else 0.0

        audio, pitch_diagnostics = self.pitch_shift(audio, sr, n_steps=detected_pitch)
        if pitch_diagnostics:
            all_diagnostics['pitch_shift'] = pitch_diagnostics
        # Add detected pitch value to diagnostics even if empty
        if 'pitch_shift' not in all_diagnostics:
            all_diagnostics['pitch_shift'] = {'detected_semitones': detected_pitch}
        else:
            all_diagnostics['pitch_shift']['detected_semitones'] = detected_pitch

        # Stage 5: Normalize loudness
        audio, loudness_diagnostics = self.normalize_loudness(audio, sr, target_lufs=target_loudness)
        if loudness_diagnostics:
            all_diagnostics['normalize_loudness'] = loudness_diagnostics

        return audio.astype(np.float32), all_diagnostics
