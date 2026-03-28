"""Audio generation pipeline orchestration."""

import logging
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np

from utilities.app_constants import (
    EMBEDS_CACHE_DIR, SPEAKERS_DIR, DEFAULT_SPEAKER,
    OUTPUT_DIR, PING_TEXT, PING_DURATION_SEC, SAMPLE_RATE,
)
from utilities.model_utils import load_model_if_needed
from utilities.cache_utils import get_audio_file_hash, get_cached_embedding, cache_embedding
from utilities.audio_utils import save_wav_file, create_silence
from utilities.app_config import AppConfig


logger = logging.getLogger(__name__)


async def generate_audio(
    text: str,
    speaker_audio: Optional[str] = None,
    language: str = "en-us",
    cfg_scale: float = 3.0,
    seed: int = 420,
    randomize_seed: bool = True,
    speed: float = 1.0,
    num_steps: int = 4,
    config: AppConfig = None,
) -> Tuple[str, int]:
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
        config: App configuration

    Returns:
        Tuple of (output_wav_path, seed_used)
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
    encode_dict = await _get_speaker_encoding(
        speaker_audio, model, config
    )

    # Generate speech
    logger.info(f"Generating speech for text: {text[:50]}...")
    audio = model.generate_speech(
        text=text,
        encode_dict=encode_dict,
        num_steps=num_steps,
        guidance_scale=cfg_scale,
        speed=speed,
    )

    # Save output
    timestamp = int(time.time() * 1000)
    output_path = OUTPUT_DIR / f"output_{timestamp}.wav"
    save_wav_file(audio, output_path, sample_rate=SAMPLE_RATE)

    logger.info(f"Saved audio to {output_path}")
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
            duration=5,
            rms=0.01,
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
            duration=5,
            rms=0.01,
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
                duration=5,
                rms=0.01,
            )
            cache_embedding(cache_key, encode_dict, EMBEDS_CACHE_DIR)

        except Exception as e:
            logger.error(f"Failed to pre-encode {audio_path.name}: {e}")

    logger.info("Speaker cache initialization complete")
