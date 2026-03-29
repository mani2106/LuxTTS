"""Configuration management for SkyrimNet-LuxTTS server."""

import argparse
from dataclasses import dataclass
from utilities.app_constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_PATH,
    GENERATION_CONCURRENCY_LIMIT,
    DEFAULT_NUM_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SPEED,
    DEFAULT_SEED,
    DEFAULT_RMS,
    DEFAULT_T_SHIFT,
    DEFAULT_RETURN_SMOOTH,
    DEFAULT_REF_DURATION,
)


@dataclass
class AppConfig:
    """Server configuration with CLI override support."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str = DEFAULT_DEVICE
    model_path: str = DEFAULT_MODEL_PATH
    concurrency_limit: int = GENERATION_CONCURRENCY_LIMIT
    default_num_steps: int = DEFAULT_NUM_STEPS
    default_guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    default_speed: float = DEFAULT_SPEED
    default_seed: int = DEFAULT_SEED
    default_rms: float = DEFAULT_RMS
    default_t_shift: float = DEFAULT_T_SHIFT
    default_return_smooth: bool = DEFAULT_RETURN_SMOOTH
    default_ref_duration: int = DEFAULT_REF_DURATION
    share: bool = False
    inbrowser: bool = False

    @classmethod
    def from_cli(cls) -> "AppConfig":
        """Parse CLI arguments and create config."""
        parser = argparse.ArgumentParser(description="SkyrimNet-LuxTTS: Gradio server for LuxTTS TTS backend")
        parser.add_argument("--server", default=DEFAULT_HOST, help="Server host to bind to")
        parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
        parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device (cuda:0, cpu, mps)")
        parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="HuggingFace model ID or local path")
        parser.add_argument("--share", action="store_true", help="Create public Gradio link")
        parser.add_argument("--inbrowser", action="store_true", help="Open browser automatically on launch")

        args = parser.parse_args()

        return cls(
            host=args.server,
            port=args.port,
            device=args.device,
            model_path=args.model_path,
            share=args.share,
            inbrowser=args.inbrowser,
        )
