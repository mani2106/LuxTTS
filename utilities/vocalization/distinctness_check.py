"""Vocalization distinctness check — ensures vocalizations are audibly different from speech."""

import numpy as np


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
