"""Custom LuxTTS-specific audio quality scorers.

These scorers handle domain-specific checks that VERSA doesn't cover:
- Silence/artifact detection (pure signal analysis, no ML)
- Post-processing chain delta analysis
- Batch degradation detection
- Vocalization tag quality
- Chunking/crossfade quality
"""

import logging
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def detect_silence_artifacts(
    audio: np.ndarray,
    sr: int,
    trailing_threshold_ms: float = 500.0,
    clipping_threshold: float = 0.99,
    silence_db: float = -60.0,
) -> dict:
    """Detect silence issues and artifacts in audio.

    Pure signal analysis — no ML, runs on CPU, suitable for fast CI.

    Args:
        audio: Float32 audio array
        sr: Sample rate
        trailing_threshold_ms: Flag if trailing silence exceeds this (ms)
        clipping_threshold: Flag if any sample exceeds this amplitude
        silence_db: Threshold for silence detection (dB)

    Returns:
        {
            "trailing_silence_ms": float,
            "has_trailing_artifact": bool,
            "has_clipping": bool,
            "silence_ratio": float,
            "zero_crossing_rate": float,
            "peak_amplitude": float,
            "duration_s": float,
        }
    """
    duration_s = len(audio) / sr
    peak_amplitude = float(np.max(np.abs(audio)))

    # Clipping detection - check for flat regions at peak amplitude
    # Audio is clipped if consecutive samples hit the same max value
    abs_audio = np.abs(audio)
    max_val = np.max(abs_audio)
    # Count samples at peak amplitude (with tolerance for float precision)
    at_peak = np.sum(abs_audio >= max_val * 0.999)
    # If more than 0.1% of samples are at peak, likely clipped
    has_clipping = bool((at_peak / len(audio) > 0.001) or (max_val >= clipping_threshold))

    # Trailing silence detection
    silence_linear = 10 ** (silence_db / 20.0)
    abs_audio = np.abs(audio)
    non_silent_from_end = np.argmax(abs_audio[::-1] > silence_linear)
    trailing_silence_ms = float(non_silent_from_end / sr * 1000)
    has_trailing_artifact = bool(trailing_silence_ms > trailing_threshold_ms)

    # Overall silence ratio
    silent_samples = np.sum(abs_audio < silence_linear)
    silence_ratio = float(silent_samples / len(audio))

    # Zero crossing rate (indicator of noisiness/buzziness)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
    zcr = float(zero_crossings / len(audio))

    return {
        "trailing_silence_ms": trailing_silence_ms,
        "has_trailing_artifact": has_trailing_artifact,
        "has_clipping": has_clipping,
        "silence_ratio": silence_ratio,
        "zero_crossing_rate": zcr,
        "peak_amplitude": peak_amplitude,
        "duration_s": duration_s,
    }


def score_post_processing_delta(
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
) -> dict:
    """Measure quality impact of the post-processing chain.

    Args:
        raw_audio: Raw TTS output (before DSP)
        processed_audio: Post-processed audio (after DSP)
        sr: Sample rate

    Returns:
        {
            "rms_change_db": float,
            "peak_change_db": float,
            "spectral_centroid_change_hz": float,
            "dnsmos_delta_sig": float or None,
            "dnsmos_delta_ovrl": float or None,
        }
    """
    # RMS change
    raw_rms = float(np.sqrt(np.mean(raw_audio ** 2))) + 1e-10
    proc_rms = float(np.sqrt(np.mean(processed_audio ** 2))) + 1e-10
    rms_change_db = 20.0 * np.log10(proc_rms / raw_rms)

    # Peak change
    raw_peak = float(np.max(np.abs(raw_audio))) + 1e-10
    proc_peak = float(np.max(np.abs(processed_audio))) + 1e-10
    peak_change_db = 20.0 * np.log10(proc_peak / raw_peak)

    # Spectral centroid change
    raw_centroid = _spectral_centroid(raw_audio, sr)
    proc_centroid = _spectral_centroid(processed_audio, sr)
    centroid_change = proc_centroid - raw_centroid

    # Optional DNSMOS delta (CPU-safe)
    dnsmos_delta_sig = None
    dnsmos_delta_ovrl = None
    try:
        from tests.audio_quality.scorers.versa_scorer import score_dnsmos
        raw_scores = score_dnsmos(raw_audio, sr)
        proc_scores = score_dnsmos(processed_audio, sr)
        dnsmos_delta_sig = proc_scores["dnsmos_sig"] - raw_scores["dnsmos_sig"]
        dnsmos_delta_ovrl = proc_scores["dnsmos_ovrl"] - raw_scores["dnsmos_ovrl"]
    except Exception:
        logger.debug("DNSMOS not available for post-processing delta, skipping")

    return {
        "rms_change_db": rms_change_db,
        "peak_change_db": peak_change_db,
        "spectral_centroid_change_hz": centroid_change,
        "dnsmos_delta_sig": dnsmos_delta_sig,
        "dnsmos_delta_ovrl": dnsmos_delta_ovrl,
    }


def score_batch_degradation(
    audio_clips: list,
    sr: int,
    use_gpu: bool = False,
) -> dict:
    """Detect voice quality drift across sequential same-speaker generations.

    Args:
        audio_clips: List of audio arrays from sequential generations
        sr: Sample rate
        use_gpu: Whether to use GPU for ML-based metrics

    Returns:
        {
            "num_clips": int,
            "first_last_speaker_similarity": float or None,
            "dnsmos_drift": float or None,
            "rms_drift_db": float,
            "duration_drift_pct": float,
        }
    """
    if len(audio_clips) < 2:
        return {"num_clips": len(audio_clips), "error": "Need at least 2 clips"}

    first = audio_clips[0]
    last = audio_clips[-1]

    # RMS drift
    first_rms = float(np.sqrt(np.mean(first ** 2))) + 1e-10
    last_rms = float(np.sqrt(np.mean(last ** 2))) + 1e-10
    rms_drift_db = abs(20.0 * np.log10(last_rms / first_rms))

    # Duration drift
    duration_drift_pct = abs(len(last) - len(first)) / max(len(first), 1) * 100

    # DNSMOS drift
    dnsmos_drift = None
    try:
        from tests.audio_quality.scorers.versa_scorer import score_dnsmos
        first_scores = score_dnsmos(first, sr)
        last_scores = score_dnsmos(last, sr)
        dnsmos_drift = last_scores["dnsmos_ovrl"] - first_scores["dnsmos_ovrl"]
    except Exception:
        logger.debug("DNSMOS not available for batch degradation scoring")

    # Speaker similarity drift (GPU recommended)
    speaker_sim = None
    try:
        from tests.audio_quality.scorers.versa_scorer import score_speaker_similarity
        result = score_speaker_similarity(last, first, sr, use_gpu=use_gpu)
        speaker_sim = result["speaker_similarity"]
    except Exception:
        logger.debug("Speaker similarity not available for batch degradation")

    return {
        "num_clips": len(audio_clips),
        "first_last_speaker_similarity": speaker_sim,
        "dnsmos_drift": dnsmos_drift,
        "rms_drift_db": rms_drift_db,
        "duration_drift_pct": duration_drift_pct,
    }


def score_vocalization_quality(
    vocalization_audio: np.ndarray,
    speech_audio: np.ndarray,
    sr: int,
    expected_min_duration_s: float = 0.2,
    expected_max_duration_s: float = 3.0,
) -> dict:
    """Assess vocalization tag quality — should sound distinct from speech.

    Args:
        vocalization_audio: Audio from a vocalization tag (e.g., [sighs])
        speech_audio: Normal speech audio for comparison
        sr: Sample rate
        expected_min_duration_s: Minimum expected duration
        expected_max_duration_s: Maximum expected duration

    Returns:
        {
            "duration_s": float,
            "duration_in_range": bool,
            "spectral_centroid_dist_hz": float,
            "energy_ratio_vs_speech": float,
            "is_distinct_from_speech": bool,
        }
    """
    voc_duration = len(vocalization_audio) / sr

    speech_centroid = _spectral_centroid(speech_audio, sr)
    voc_centroid = _spectral_centroid(vocalization_audio, sr)
    centroid_dist = abs(voc_centroid - speech_centroid)

    speech_rms = float(np.sqrt(np.mean(speech_audio ** 2))) + 1e-10
    voc_rms = float(np.sqrt(np.mean(vocalization_audio ** 2))) + 1e-10
    energy_ratio = voc_rms / speech_rms

    is_distinct = bool(centroid_dist > 200 or abs(20 * np.log10(energy_ratio)) > 3)

    return {
        "duration_s": voc_duration,
        "duration_in_range": bool(expected_min_duration_s <= voc_duration <= expected_max_duration_s),
        "spectral_centroid_dist_hz": centroid_dist,
        "energy_ratio_vs_speech": energy_ratio,
        "is_distinct_from_speech": is_distinct,
    }


def score_chunking_quality(
    audio: np.ndarray,
    sr: int,
    estimated_chunk_duration_s: float = 3.0,
    energy_dip_threshold_db: float = -6.0,
) -> dict:
    """Detect crossfade artifacts at chunk boundaries.

    Args:
        audio: Full concatenated audio
        sr: Sample rate
        estimated_chunk_duration_s: Expected duration per chunk
        energy_dip_threshold_db: Flag if energy dips below this at boundaries

    Returns:
        {
            "num_estimated_boundaries": int,
            "boundary_dips_db": list[float],
            "has_audible_artifacts": bool,
            "max_dip_db": float,
        }
    """
    total_duration = len(audio) / sr
    num_boundaries = max(0, int(total_duration / estimated_chunk_duration_s) - 1)

    if num_boundaries == 0:
        return {
            "num_estimated_boundaries": 0,
            "boundary_dips_db": [],
            "has_audible_artifacts": False,
            "max_dip_db": 0.0,
        }

    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    overall_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-10

    dips = []
    for i in range(1, num_boundaries + 1):
        center = int(i * estimated_chunk_duration_s * sr)
        start = max(0, center - window_samples // 2)
        end = min(len(audio), center + window_samples // 2)
        boundary_rms = float(np.sqrt(np.mean(audio[start:end] ** 2))) + 1e-10
        dip_db = 20.0 * np.log10(boundary_rms / overall_rms)
        dips.append(dip_db)

    max_dip = min(dips) if dips else 0.0
    has_artifacts = bool(max_dip < energy_dip_threshold_db)

    return {
        "num_estimated_boundaries": num_boundaries,
        "boundary_dips_db": dips,
        "has_audible_artifacts": has_artifacts,
        "max_dip_db": max_dip,
    }


def _spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid in Hz."""
    magnitudes = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    centroid = float(np.sum(magnitudes * freqs) / (np.sum(magnitudes) + 1e-10))
    return centroid
