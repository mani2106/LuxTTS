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

    # Create equal-power crossfade curves (cosine/sine)
    fade_out = np.cos(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2
    fade_in = np.sin(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2

    # Apply crossfade
    result = audio1.copy()
    result[-crossfade_samples:] *= fade_out

    audio2_crossfade = audio2[:crossfade_samples] * fade_in

    # Sum crossfaded region
    result[-crossfade_samples:] += audio2_crossfade

    # Append remaining audio2
    result = np.concatenate([result, audio2[crossfade_samples:]])

    return result
