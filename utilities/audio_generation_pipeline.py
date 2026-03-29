"""Audio generation pipeline orchestration."""

import logging
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from utilities.app_constants import (
    EMBEDS_CACHE_DIR,
    SPEAKERS_DIR,
    DEFAULT_SPEAKER,
    OUTPUT_DIR,
    PING_TEXT,
    PING_DURATION_SEC,
    SAMPLE_RATE,
    DEFAULT_RMS,
    DEFAULT_T_SHIFT,
    DEFAULT_RETURN_SMOOTH,
    DEFAULT_REF_DURATION,
    DEFAULT_SPEED,
    DEFAULT_NUM_STEPS,
    DEFAULT_POST_PROCESSING_ENABLED,
    DEFAULT_PITCH_SHIFT,
    DEFAULT_EQ_INTENSITY,
    DEFAULT_COMPRESSOR_THRESHOLD_OFFSET,
    DEFAULT_COMPRESSOR_RATIO,
    DEFAULT_COMPRESSOR_KNEE_DB,
    DEFAULT_COMPRESSOR_ATTACK_MS,
    DEFAULT_COMPRESSOR_RELEASE_MS,
    DEFAULT_MAX_GAIN_REDUCTION_DB,
    DEFAULT_DE_ESS_INTENSITY,
    DEFAULT_TARGET_LOUDNESS_LUFS,
)
from utilities.model_utils import load_model_if_needed
from utilities.cache_utils import get_audio_file_hash, get_cached_embedding, cache_embedding
from utilities.audio_utils import save_wav_file, create_silence
from utilities.app_config import AppConfig
from utilities.post_processor import AudioPostProcessor


logger = logging.getLogger(__name__)


async def generate_audio(
    text: str,
    speaker_audio: Optional[str] = None,
    language: str = "en-us",
    cfg_scale: float = 3.0,
    seed: int = 420,
    randomize_seed: bool = True,
    speed: float = DEFAULT_SPEED,
    num_steps: int = DEFAULT_NUM_STEPS,
    t_shift: float = DEFAULT_T_SHIFT,
    return_smooth: bool = DEFAULT_RETURN_SMOOTH,
    config: AppConfig = None,
    # Post-processing parameters
    enable_post_processing: bool = DEFAULT_POST_PROCESSING_ENABLED,
    pitch_shift: Optional[float] = DEFAULT_PITCH_SHIFT,
    eq_intensity: float = DEFAULT_EQ_INTENSITY,
    compressor_threshold_offset: float = DEFAULT_COMPRESSOR_THRESHOLD_OFFSET,
    compressor_ratio: float = DEFAULT_COMPRESSOR_RATIO,
    compressor_knee_db: float = DEFAULT_COMPRESSOR_KNEE_DB,
    compressor_attack_ms: float = DEFAULT_COMPRESSOR_ATTACK_MS,
    compressor_release_ms: float = DEFAULT_COMPRESSOR_RELEASE_MS,
    max_gain_reduction_db: float = DEFAULT_MAX_GAIN_REDUCTION_DB,
    de_ess_intensity: float = DEFAULT_DE_ESS_INTENSITY,
    target_loudness: float = DEFAULT_TARGET_LOUDNESS_LUFS,
    return_diagnostics: bool = False,
    save_raw: bool = False,
) -> Tuple[str, int] | Tuple[str, int, str, dict]:
    """
    Generate audio from text using LuxTTS.

    Args:
        text: Text to synthesize (use "ping" for health check)
        speaker_audio: Path to reference audio for voice cloning
        language: Language code
        cfg_scale: Classifier-free guidance scale
        seed: Random seed
        randomize_seed: Whether to randomize seed before generation
        speed: Speech speed multiplier
        num_steps: Flow matching steps (4 for distilled model)
        t_shift: Sampling shift, higher can sound better but worse WER
        return_smooth: If True, disables 48k upsampling for smoother output
        config: App configuration
        enable_post_processing: If True, apply DSP post-processing chain
        pitch_shift: Manual pitch shift in semitones, or None for auto-detection
        eq_intensity: EQ aggressiveness (0.0-1.0)
        compressor_threshold_offset: Compressor threshold offset from signal RMS (dB)
        compressor_ratio: Compression ratio
        compressor_knee_db: Soft-knee width (dB)
        compressor_attack_ms: Attack time (ms)
        compressor_release_ms: Release time (ms)
        max_gain_reduction_db: Maximum gain reduction per frame (dB)
        de_ess_intensity: De-essing strength (0.0-1.0)
        target_loudness: Target LUFS for final normalization
        return_diagnostics: If True, return diagnostic metrics
        save_raw: If True, also save unprocessed audio (returns tuple of 4)

    Returns:
        Tuple of (output_wav_path, seed) or (output_wav_path, seed, raw_path, diagnostics)
    """
    # Health check
    if text == PING_TEXT:
        return _handle_ping()

    # Ensure model is loaded
    model = load_model_if_needed(config)

    # Set seed
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)

    # Get speaker encoding
    encode_dict = await _get_speaker_encoding(speaker_audio, model, config)

    # Generate speech
    logger.info(f"Generating speech for text: {text[:50]}...")
    audio = model.generate_speech(
        text=text,
        encode_dict=encode_dict,
        num_steps=num_steps,
        guidance_scale=cfg_scale,
        speed=speed,
        t_shift=t_shift,
        return_smooth=return_smooth,
    )

    # Convert to numpy for post-processing
    if hasattr(audio, 'numpy'):
        audio = audio.numpy()
    elif isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    audio_array = audio.astype(np.float32).squeeze()

    # Save raw output if requested (for A/B preview)
    raw_path = None
    raw_diagnostics = {}
    if save_raw:
        raw_timestamp = int(time.time() * 1000)
        raw_path = OUTPUT_DIR / f"output_{raw_timestamp}_raw.wav"
        save_wav_file(audio_array, raw_path, sample_rate=SAMPLE_RATE)
        logger.debug(f"Saved raw audio to {raw_path}")

    # Apply post-processing
    if enable_post_processing:
        processor = AudioPostProcessor(return_diagnostics=return_diagnostics)
        audio_array, raw_diagnostics = processor.process(
            audio_array,
            sr=SAMPLE_RATE,
            text=text,
            pitch_shift=pitch_shift,
            eq_intensity=eq_intensity,
            de_ess_intensity=de_ess_intensity,
            compressor_threshold_offset_db=compressor_threshold_offset,
            compressor_ratio=compressor_ratio,
            compressor_knee_db=compressor_knee_db,
            compressor_attack_ms=compressor_attack_ms,
            compressor_release_ms=compressor_release_ms,
            max_gain_reduction_db=max_gain_reduction_db,
            target_loudness=target_loudness,
        )

    # Save processed output
    timestamp = int(time.time() * 1000)
    output_path = OUTPUT_DIR / f"output_{timestamp}.wav"
    save_wav_file(audio_array, output_path, sample_rate=SAMPLE_RATE)

    logger.info(f"Saved audio to {output_path}")

    # Return based on what was requested
    if save_raw or return_diagnostics:
        return str(output_path), seed, str(raw_path) if raw_path else None, raw_diagnostics
    return str(output_path), seed


def _handle_ping() -> Tuple[str, int]:
    """Handle health check ping request."""
    silence = create_silence(PING_DURATION_SEC, SAMPLE_RATE)
    output_path = OUTPUT_DIR / "ping_silence.wav"
    save_wav_file(silence, output_path, sample_rate=SAMPLE_RATE)
    return str(output_path), 0


async def _get_speaker_encoding(
    speaker_audio: Optional[str],
    model,
    config: AppConfig,
) -> dict:
    """
    Get speaker encoding from audio file, using cache when available.

    Falls back to default speaker if audio loading fails.

    Args:
        speaker_audio: Path to reference audio, or None for default
        model: LuxTTS model instance
        config: App configuration

    Returns:
        Encoding dictionary from LuxTTS.encode_prompt()
    """
    # Determine which audio to use
    if speaker_audio is None:
        audio_path = SPEAKERS_DIR / f"{DEFAULT_SPEAKER}.wav"
    else:
        audio_path = Path(speaker_audio)

    # Check cache
    cache_key = get_audio_file_hash(audio_path)
    cached = get_cached_embedding(cache_key, EMBEDS_CACHE_DIR)
    if cached is not None:
        return cached

    # Encode with fallback
    try:
        logger.info(f"Encoding speaker from {audio_path}...")
        encode_dict = model.encode_prompt(
            prompt_audio=str(audio_path),
            duration=DEFAULT_REF_DURATION,
            rms=DEFAULT_RMS,
        )
        # Cache the result
        cache_embedding(cache_key, encode_dict, EMBEDS_CACHE_DIR)
        return encode_dict

    except Exception as e:
        logger.error(f"Failed to encode speaker audio {audio_path}: {e}")
        logger.info(f"Falling back to default speaker: {DEFAULT_SPEAKER}")

        # Load default preset
        fallback_path = SPEAKERS_DIR / f"{DEFAULT_SPEAKER}.wav"

        # Check if fallback is cached
        fallback_key = get_audio_file_hash(fallback_path)
        fallback_cached = get_cached_embedding(fallback_key, EMBEDS_CACHE_DIR)
        if fallback_cached is not None:
            return fallback_cached

        # Encode fallback
        logger.info(f"Encoding fallback speaker from {fallback_path}...")
        encode_dict = model.encode_prompt(
            prompt_audio=str(fallback_path),
            duration=DEFAULT_REF_DURATION,
            rms=DEFAULT_RMS,
        )
        cache_embedding(fallback_key, encode_dict, EMBEDS_CACHE_DIR)
        return encode_dict


async def init_speaker_cache(config: AppConfig):
    """
    Pre-encode all preset speaker voices on startup.

    Args:
        config: App configuration
    """
    model = load_model_if_needed(config)

    if not SPEAKERS_DIR.exists():
        logger.warning(f"Speakers directory not found: {SPEAKERS_DIR}")
        return

    preset_files = list(SPEAKERS_DIR.glob("*.wav"))
    logger.info(f"Pre-encoding {len(preset_files)} preset voices...")

    for audio_path in preset_files:
        try:
            cache_key = get_audio_file_hash(audio_path)

            # Skip if already cached
            if get_cached_embedding(cache_key, EMBEDS_CACHE_DIR) is not None:
                logger.debug(f"Already cached: {audio_path.name}")
                continue

            logger.info(f"Pre-encoding: {audio_path.name}")
            encode_dict = model.encode_prompt(
                prompt_audio=str(audio_path),
                duration=DEFAULT_REF_DURATION,
                rms=DEFAULT_RMS,
            )
            cache_embedding(cache_key, encode_dict, EMBEDS_CACHE_DIR)

        except Exception as e:
            logger.error(f"Failed to pre-encode {audio_path.name}: {e}")

    logger.info("Speaker cache initialization complete")
