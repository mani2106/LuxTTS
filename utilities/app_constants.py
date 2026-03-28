"""Hardcoded constants for the SkyrimNet-LuxTTS server."""

from pathlib import Path

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860
DEFAULT_DEVICE = "cuda:0"
DEFAULT_MODEL_PATH = "YatharthS/LuxTTS"

# Generation defaults
DEFAULT_NUM_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_SPEED = 1.0
DEFAULT_SEED = 420

# Cache configuration
CACHE_DIR = Path("cache")
EMBEDS_CACHE_DIR = CACHE_DIR / "embeds"
GENERATION_CONCURRENCY_LIMIT = 2

# Audio defaults
SAMPLE_RATE = 48000  # LuxTTS outputs at 48kHz
DEFAULT_SPEAKER = "malecommoner"
SPEAKERS_DIR = Path("speakers") / "en"

# Output
OUTPUT_DIR = Path("output_tmp")

# Health check
PING_DURATION_SEC = 0.5
PING_TEXT = "ping"

# Model storage
MODELS_DIR = Path("models")

# Create directories on import
EMBEDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
