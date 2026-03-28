"""Model singleton loading and device management."""

import logging
import os
from pathlib import Path
from huggingface_hub import snapshot_download

from utilities.app_config import AppConfig
from utilities.app_constants import MODELS_DIR


logger = logging.getLogger(__name__)

# Global singleton instances
_CURRENT_MODEL: "LuxTTS | None" = None
_CURRENT_CONFIG: "AppConfig | None" = None


def load_model_if_needed(config: AppConfig) -> "LuxTTS":
    """
    Load LuxTTS model singleton.

    Loads on first call, returns cached instance on subsequent calls.
    Downloads model to ./models/ if not present locally.

    Args:
        config: App configuration containing model_path and device

    Returns:
        Loaded LuxTTS instance
    """
    global _CURRENT_MODEL, _CURRENT_CONFIG

    # Return cached instance if config hasn't changed
    if _CURRENT_MODEL is not None and _CURRENT_CONFIG == config:
        return _CURRENT_MODEL

    # Clear previous model if switching
    if _CURRENT_MODEL is not None:
        logger.info("Model configuration changed, reloading...")
        _CURRENT_MODEL = None

    # Resolve model path (download if needed)
    model_path = _resolve_model_path(config.model_path)

    # Import LuxTTS (defer import to avoid unnecessary dependency loading)
    from zipvoice.luxvoice import LuxTTS

    logger.info(f"Loading LuxTTS from {model_path} on {config.device}...")
    _CURRENT_MODEL = LuxTTS(model_path=str(model_path), device=config.device)
    _CURRENT_CONFIG = config

    # Set to inference mode
    import torch
    _CURRENT_MODEL.model.eval()
    torch.inference_mode()

    logger.info("Model loaded successfully")
    return _CURRENT_MODEL


def _resolve_model_path(model_id_or_path: str) -> Path:
    """
    Resolve model path to local directory.

    If a HuggingFace ID is provided, downloads to ./models/.
    If a local path is provided, returns it as-is.

    Args:
        model_id_or_path: HuggingFace repo ID (e.g., "YatharthS/LuxTTS")
                        or local path

    Returns:
        Path to local model directory
    """
    # Check if it's already a local directory
    if os.path.isdir(model_id_or_path):
        return Path(model_id_or_path)

    # Sanitize repo ID for directory name
    safe_name = model_id_or_path.replace("/", "--")
    local_path = MODELS_DIR / safe_name

    # Download if not present
    if not local_path.exists():
        logger.info(f"Downloading model {model_id_or_path} to {local_path}...")
        local_path = Path(
            snapshot_download(
                repo_id=model_id_or_path,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
            )
        )
        logger.info("Download complete")

    return local_path


def get_current_model() -> "LuxTTS | None":
    """Get the current model singleton without triggering a load."""
    return _CURRENT_MODEL
