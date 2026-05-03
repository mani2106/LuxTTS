"""Shared fixtures for audio quality tests."""

import json
from pathlib import Path

import numpy as np
import pytest

BASELINES_DIR = Path(__file__).parent / "baselines"


@pytest.fixture
def baselines_dir():
    return BASELINES_DIR


@pytest.fixture
def sample_48k_speech():
    """Generate synthetic 48kHz speech-like audio (mix of fundamental + harmonics)."""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Fundamental at 150Hz (typical male voice) + harmonics
    audio = (
        0.4 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 450 * t)
        + 0.05 * np.sin(2 * np.pi * 8000 * t)  # sibilant-like content
    )
    # Add amplitude envelope (attack-sustain-release)
    envelope = np.ones_like(t)
    attack = int(0.05 * sr)
    release = int(0.1 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    audio = (audio * envelope).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_48k_silence():
    """Generate 1 second of silence at 48kHz."""
    sr = 48000
    return np.zeros(sr, dtype=np.float32), sr


def load_baseline(name: str) -> dict:
    """Load a baseline JSON file by name."""
    path = BASELINES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Baseline file not found: {path}")
    with open(path) as f:
        return json.load(f)
