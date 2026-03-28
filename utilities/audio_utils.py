"""Audio I/O utilities."""

import logging
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


def load_wav_file(wav_path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Load WAV file and return audio array with sample rate.

    Args:
        wav_path: Path to WAV file

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid WAV
    """
    wav_path = Path(wav_path)

    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    try:
        # Use Python's built-in wave module (no torchaudio dependency)
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            sample_rate = wf.getframerate()
            audio_bytes = wf.readframes(frames)

            # Convert to numpy array
            audio = np.frombuffer(audio_bytes, dtype=np.int16)

            # Normalize to [-1, 1] float range
            audio = audio.astype(np.float32) / 32768.0

            # Convert to mono if stereo
            if wf.getnchannels() == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            return audio, sample_rate

    except Exception as e:
        raise ValueError(f"Failed to load WAV file {wav_path}: {e}")


def save_wav_file(audio: torch.Tensor | np.ndarray, wav_path: str | Path, sample_rate: int = 48000):
    """
    Save audio tensor/array to WAV file.

    Args:
        audio: Audio data as torch tensor or numpy array (float, range [-1, 1])
        wav_path: Output path
        sample_rate: Sample rate in Hz
    """
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    # Ensure 1D
    if audio.ndim > 1:
        audio = audio.squeeze()

    # Convert to int16
    audio_int16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)

    # Write WAV
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    logger.debug(f"Saved audio to {wav_path}")


def create_silence(duration_sec: float, sample_rate: int = 48000) -> np.ndarray:
    """
    Create silence audio array.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Silent audio array
    """
    num_samples = int(duration_sec * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)
