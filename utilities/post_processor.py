"""Audio post-processing pipeline for LuxTTS output quality improvement."""

import logging
import re
from typing import Optional

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
