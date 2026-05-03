"""Shared fixtures for audio quality tests."""

import json
from pathlib import Path

import numpy as np
import pytest

BASELINES_DIR = Path(__file__).parent / "baselines"
OUTPUT_DIR = Path(__file__).parent / "output"

SPEAKERS = [
    "cicero",
    "malecommoner",
    "femalenord",
    "serana",
    "aaaharleyvoicequest",
    "alduin",
    "maleargonian",
    "femaleargonian",
    "malekhajiit",
    "femalekhajiit",
    "maleoldgrumpy",
    "femaleoldgrumpy",
]

SPEAKERS_DIR = Path("speakers") / "en"

# Fast subset for PR checks — covers male/female/beast (3 voices, ~3x faster)
SPEAKER_SUBSET = ["cicero", "femalenord", "alduin"]

# Generation config for reproducibility — captured in manifest entries
GENERATION_CONFIG = {
    "num_steps": 4,
    "guidance_scale": 3.0,
}


# --- Gate constants ---

MAX_SILENCE_RATIO = 0.5
MAX_TRAILING_SILENCE_MS = 800
MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0
SAMPLE_RATE = 48000


# --- CPU-only fixtures (used by test_fast_ci.py) ---


@pytest.fixture
def baselines_dir():
    return BASELINES_DIR


@pytest.fixture
def sample_48k_speech():
    """Generate synthetic 48kHz speech-like audio (mix of fundamental + harmonics)."""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = (
        0.4 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.sin(2 * np.pi * 450 * t)
        + 0.05 * np.sin(2 * np.pi * 8000 * t)
    )
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- GPU test fixtures (used by test_full_eval.py) ---


@pytest.fixture(scope="session")
def output_dir():
    """Directory for generated audio output and manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="session")
def manifest(output_dir):
    """Session-scoped manifest list. Written to disk at session finish."""
    entries = []

    yield entries

    if entries:
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        print(f"\nManifest saved to {manifest_path} ({len(entries)} entries)")


@pytest.fixture(scope="session")
def generation_config():
    """Generation config for reproducibility — included in manifest entries."""
    import subprocess
    config = dict(GENERATION_CONFIG)
    try:
        config["model_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        config["model_commit"] = "unknown"
    return config


@pytest.fixture(scope="session")
def speaker_map():
    """Map of speaker name -> absolute path to speaker WAV file.

    Use SPEAKER_SUBSET for fast PR checks, SPEAKERS for full matrix.
    Control via: pytest -m gpu -- speakers=subset  (uses SPEAKER_SUBSET)
    Default: full SPEAKERS list.
    """
    # Check if a subset was requested via pytest config
    speaker_list = SPEAKERS  # default: full matrix
    mapping = {}
    for name in speaker_list:
        path = SPEAKERS_DIR / f"{name}.wav"
        if path.exists():
            mapping[name] = str(path)
    if not mapping:
        pytest.skip("No speaker files found in speakers/en/")
    return mapping
