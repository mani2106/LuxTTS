"""VERSA toolkit integration for standard audio quality metrics.

Provides thin wrappers around VERSA's Python API for:
- DNSMOS (signal quality, CPU-compatible via ONNX)
- UTMOS (naturalness, GPU recommended)
- Speaker similarity (voice cloning quality, GPU recommended)
- WER via Whisper (intelligibility, GPU recommended)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _resample(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample audio to target sample rate."""
    if sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)


def score_dnsmos_available() -> bool:
    """Check if DNSMOS is available without importing heavy deps."""
    try:
        import speechmos  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def score_dnsmos(audio: np.ndarray, sr: int) -> dict:
    """Score audio with DNSMOS (signal quality, background noise, overall).

    Runs on CPU via ONNX. Audio is resampled to 16kHz internally.

    Args:
        audio: Float32 audio array
        sr: Sample rate

    Returns:
        {"dnsmos_sig": float, "dnsmos_bak": float, "dnsmos_ovrl": float}
    """
    from speechmos import dnsmos

    audio_16k = _resample(audio, sr, 16000).astype(np.float64)

    result = dnsmos.run(audio_16k, 16000)

    key_map = {"sig_mos": "dnsmos_sig", "bak_mos": "dnsmos_bak", "ovrl_mos": "dnsmos_ovrl"}
    scores = {}
    for src_key, dst_key in key_map.items():
        if src_key in result:
            scores[dst_key] = float(result[src_key])

    if not scores:
        raise ValueError(f"DNSMOS returned unexpected format: {result}")

    return scores


def score_utmos(audio: np.ndarray, sr: int, use_gpu: bool = False) -> dict:
    """Score audio with UTMOS v1 (perceived naturalness).

    Args:
        audio: Float32 audio array
        sr: Sample rate
        use_gpu: Whether to use GPU

    Returns:
        {"utmos": float} — score in 1-5 range
    """
    from versa import pseudo_mos_setup, pseudo_mos_metric

    predictor_dict, predictor_fs = pseudo_mos_setup(
        predictor_types=["utmos"],
        predictor_args={"utmos": {"fs": 16000}},
        use_gpu=use_gpu,
    )

    audio_16k = _resample(audio, sr, 16000)
    scores = pseudo_mos_metric(audio_16k, 16000, predictor_dict, predictor_fs, use_gpu=use_gpu)

    return {"utmos": float(scores.get("utmos", 0.0))}


def score_speaker_similarity(
    generated_audio: np.ndarray,
    reference_audio: np.ndarray,
    sr: int,
    use_gpu: bool = False,
) -> dict:
    """Score speaker similarity between generated and reference audio.

    Args:
        generated_audio: TTS-generated audio
        reference_audio: Original reference/prompt audio
        sr: Sample rate (both must be same)
        use_gpu: Whether to use GPU

    Returns:
        {"speaker_similarity": float} — cosine similarity, higher is better (>0.8 = same speaker)
    """
    from versa import speaker_model_setup, speaker_metric

    model = speaker_model_setup(model_tag="default", use_gpu=use_gpu)

    gen_16k = _resample(generated_audio, sr, 16000)
    ref_16k = _resample(reference_audio, sr, 16000)

    result = speaker_metric(model=model, pred_x=gen_16k, gt_x=ref_16k, fs=16000)

    return {"speaker_similarity": float(result.get("spk_similarity", 0.0))}


def score_wer(
    audio: np.ndarray,
    sr: int,
    reference_text: str,
    whisper_model: str = "base",
    use_gpu: bool = False,
) -> dict:
    """Score word error rate using Whisper ASR.

    Args:
        audio: Audio to transcribe
        sr: Sample rate
        reference_text: Expected text
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        use_gpu: Whether to use GPU

    Returns:
        {"wer": float, "cer": float, "hyp_text": str, "ref_text": str}
    """
    from versa import whisper_wer_setup, whisper_levenshtein_metric

    wer_utils = whisper_wer_setup(
        model_tag=whisper_model,
        beam_size=5,
        text_cleaner="whisper_basic",
        use_gpu=use_gpu,
    )

    audio_16k = _resample(audio, sr, 16000)
    result = whisper_levenshtein_metric(
        wer_utils=wer_utils,
        pred_x=audio_16k,
        ref_text=reference_text,
        fs=16000,
    )

    # Calculate WER percentage
    total_words = (
        result["whisper_wer_delete"]
        + result["whisper_wer_replace"]
        + result["whisper_wer_equal"]
    )
    wer_score = (
        (result["whisper_wer_delete"] + result["whisper_wer_insert"] + result["whisper_wer_replace"])
        / max(total_words, 1)
    )

    total_chars = (
        result["whisper_cer_delete"]
        + result["whisper_cer_replace"]
        + result["whisper_cer_equal"]
    )
    cer_score = (
        (result["whisper_cer_delete"] + result["whisper_cer_insert"] + result["whisper_cer_replace"])
        / max(total_chars, 1)
    )

    return {
        "wer": float(wer_score),
        "cer": float(cer_score),
        "hyp_text": result.get("whisper_hyp_text", ""),
        "ref_text": result.get("ref_text", reference_text),
    }
