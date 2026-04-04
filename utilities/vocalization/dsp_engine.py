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
        # librosa 0.10+ uses different API - pass sr as keyword
        try:
            return librosa.effects.pitch_shift(y=audio.astype(np.float64), sr=sr, n_steps=semitones).astype(np.float32)
        except TypeError:
            # Fallback for older librosa versions
            return librosa.effects.pitch_shift(audio.astype(np.float64), sr, semitones).astype(np.float32)

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
        excess = np.abs(audio[mask]) - threshold
        result[mask] = np.sign(audio[mask]) * (threshold + (1 - threshold) * np.tanh(excess / (1 - threshold)))
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
