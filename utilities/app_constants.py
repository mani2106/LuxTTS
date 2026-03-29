"""Hardcoded constants for the SkyrimNet-LuxTTS server."""

from pathlib import Path

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860
DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL_PATH = "YatharthS/LuxTTS"

# Generation defaults
DEFAULT_NUM_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_SPEED = 0.8
DEFAULT_SEED = 420
DEFAULT_RMS = 0.03  # Higher makes it sound louder
DEFAULT_T_SHIFT = 0.9  # Sampling param, higher can sound better but worse WER
DEFAULT_RETURN_SMOOTH = True  # Makes it sound smoother possibly but less cleaner
DEFAULT_REF_DURATION = 10  # Lower speeds up inference; set to 1000 if artifacts in beginning

# Post-processing defaults
DEFAULT_POST_PROCESSING_ENABLED = True
DEFAULT_PITCH_SHIFT = None  # None = auto from text
DEFAULT_EQ_INTENSITY = 1.0
DEFAULT_COMPRESSOR_THRESHOLD_OFFSET = -6.0  # dB offset from signal RMS
DEFAULT_COMPRESSOR_RATIO = 4.0
DEFAULT_COMPRESSOR_KNEE_DB = 4.0
DEFAULT_COMPRESSOR_ATTACK_MS = 10.0
DEFAULT_COMPRESSOR_RELEASE_MS = 100.0
DEFAULT_MAX_GAIN_REDUCTION_DB = 12.0
DEFAULT_DE_ESS_INTENSITY = 0.5
DEFAULT_TARGET_LOUDNESS_LUFS = -16.0

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
