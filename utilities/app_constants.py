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
DEFAULT_REF_DURATION = 3  # Lower speeds up inference; ZipVoice recommends 1-3s

# Post-processing defaults
DEFAULT_POST_PROCESSING_ENABLED = True
DEFAULT_PITCH_SHIFT = None  # None = auto from text (only used when enable_auto_pitch_shift=True)
DEFAULT_EQ_INTENSITY = 1.0
DEFAULT_COMPRESSOR_THRESHOLD_OFFSET = -6.0  # dB offset from signal RMS
DEFAULT_COMPRESSOR_RATIO = 2.0
DEFAULT_COMPRESSOR_KNEE_DB = 8.0
DEFAULT_COMPRESSOR_ATTACK_MS = 10.0
DEFAULT_COMPRESSOR_RELEASE_MS = 100.0
DEFAULT_MAX_GAIN_REDUCTION_DB = 12.0
DEFAULT_DE_ESS_INTENSITY = 0.3  # Gentler for TTS (was 0.5)
DEFAULT_TARGET_LOUDNESS_LUFS = -18.0  # RPG dialogue standard (was -16.0)

# Signal-adaptive thresholds
SIBILANCE_RATIO_THRESHOLD = 0.15  # De-esser activates above this
CENTROID_LOW_THRESHOLD = 1500.0   # Hz - mud cut activates below this
CENTROID_HIGH_THRESHOLD = 3500.0  # Hz - presence boost activates above this
PEAK_LIMIT_THRESHOLD = 0.93      # Limiter activates above this
HPF_CUTOFF_HZ = 80.0             # High-pass filter cutoff (always on)

# Soft-knee limiter defaults
LIMITER_THRESHOLD_DB = -1.0      # dBTP
LIMITER_ATTACK_MS = 1.0          # 0-2ms range
LIMITER_RELEASE_MS = 80.0        # 50-150ms range for speech
LIMITER_MAX_REDUCTION_DB = 6.0   # Cap to avoid aggressive pumping

# De-esser proportional scaling
DE_ESS_SIBILANCE_FLOOR = 0.15    # Below this ratio, no de-essing
DE_ESS_SIBILANCE_SCALE = 4.0     # Multiplier for proportional intensity
DE_ESS_MAX_REDUCTION_DB = -6.0   # Max gain reduction in sibilance band

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

# TDR Nova VST3 plugin configuration
TDR_NOVA_VST3_PATH = Path(r"F:\Software\TDR Nova (no installer)\VST3\x64\TDR Nova.vst3")
TDR_NOVA_ENABLED = True  # Set False to use pedalboard/scipy fallback

# Create directories on import
EMBEDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
