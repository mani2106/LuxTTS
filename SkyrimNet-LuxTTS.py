#!/usr/bin/env python3
"""
SkyrimNet-LuxTTS: Gradio server for LuxTTS TTS backend.

Drop-in replacement for Chatterbox/Zonos backends in SkyrimNet GamePlugin.
"""

import asyncio
import logging
import sys

import gradio as gr

from utilities.app_config import AppConfig
from utilities.app_constants import GENERATION_CONCURRENCY_LIMIT
from utilities.model_utils import load_model_if_needed
from utilities.audio_generation_pipeline import (
    generate_audio,
    init_speaker_cache,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_interface(config: AppConfig) -> gr.Blocks:
    """
    Build Gradio Blocks interface.

    Args:
        config: App configuration

    Returns:
        Configured Gradio interface
    """
    with gr.Blocks(
        title="SkyrimNet-LuxTTS",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# SkyrimNet-LuxTTS")
        gr.Markdown("LuxTTS text-to-speech backend for SkyrimNet GamePlugin")

        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=3,
                )

                speaker_audio = gr.Audio(
                    label="Speaker Audio (optional)",
                    type="filepath",
                    sources=["upload"],
                )

                language = gr.Textbox(
                    label="Language",
                    value="en-us",
                    visible=False,  # LuxTTS has limited multilingual support
                )

            with gr.Column():
                with gr.Accordion("Generation Settings", open=True):
                    cfg_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=5.0,
                        step=0.1,
                        value=config.default_guidance_scale,
                    )

                    speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=config.default_speed,
                    )

                    num_steps = gr.Slider(
                        label="Num Steps",
                        minimum=2,
                        maximum=8,
                        step=1,
                        value=config.default_num_steps,
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=config.default_seed,
                        precision=0,
                    )

                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=True,
                    )

                with gr.Accordion("Post-Processing", open=False):
                    enable_post_processing = gr.Checkbox(
                        label="Enable Post-Processing",
                        value=True,
                    )

                    auto_pitch = gr.Checkbox(
                        label="Auto-detect pitch from text",
                        value=True,
                    )

                    pitch_shift = gr.Slider(
                        label="Pitch Shift (semitones, or 0 for auto-detect)",
                        minimum=-12.0,
                        maximum=12.0,
                        step=0.5,
                        value=0.0,
                        interactive=True,
                    )

                    eq_intensity = gr.Slider(
                        label="EQ Intensity",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=1.0,
                    )

                    de_ess_intensity = gr.Slider(
                        label="De-Ess Intensity",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                    )

                    compressor_threshold = gr.Slider(
                        label="Compressor Threshold Offset (dB)",
                        minimum=-30.0,
                        maximum=0.0,
                        step=1.0,
                        value=-6.0,
                    )

                    compressor_ratio = gr.Slider(
                        label="Compressor Ratio",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=4.0,
                    )

                    compressor_knee = gr.Slider(
                        label="Compressor Knee (dB)",
                        minimum=0.0,
                        maximum=12.0,
                        step=1.0,
                        value=4.0,
                    )

                    target_loudness = gr.Slider(
                        label="Target Loudness (LUFS)",
                        minimum=-30.0,
                        maximum=-5.0,
                        step=1.0,
                        value=-16.0,
                    )

                    save_raw_for_ab = gr.Checkbox(
                        label="Save Raw (A/B Preview)",
                        value=True,
                    )

        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            output_audio = gr.Audio(
                label="Processed Output",
                autoplay=True,
            )
            output_raw = gr.Audio(
                label="Raw Output (no post-processing)",
                autoplay=False,
            )
            output_seed = gr.Number(
                label="Seed Used",
                interactive=False,
            )

        # Wire up generation
        async def do_generate(
            text,
            speaker_audio_file,
            language,
            cfg_scale,
            speed,
            num_steps,
            seed,
            randomize_seed,
            enable_post_processing,
            auto_pitch,
            pitch_shift,
            eq_intensity,
            de_ess_intensity,
            compressor_threshold,
            compressor_ratio,
            compressor_knee,
            target_loudness,
            save_raw_for_ab,
        ):
            # Determine pitch_shift: if auto_pitch is True and pitch_shift is 0.0, use None (auto-detect)
            # Otherwise use the explicit value (including 0.0 if user explicitly set it)
            final_pitch_shift = None if (auto_pitch and pitch_shift == 0.0) else float(pitch_shift)

            result = await generate_audio(
                text=text or "",
                speaker_audio=speaker_audio_file,
                language=language,
                cfg_scale=cfg_scale,
                seed=int(seed),
                randomize_seed=randomize_seed,
                speed=speed,
                num_steps=int(num_steps),
                config=config,
                enable_post_processing=enable_post_processing,
                pitch_shift=final_pitch_shift,
                eq_intensity=eq_intensity,
                de_ess_intensity=de_ess_intensity,
                compressor_threshold_offset=compressor_threshold,
                compressor_ratio=compressor_ratio,
                compressor_knee_db=compressor_knee,
                compressor_attack_ms=config.default_compressor_attack_ms,
                compressor_release_ms=config.default_compressor_release_ms,
                max_gain_reduction_db=config.default_max_gain_reduction_db,
                target_loudness=target_loudness,
                save_raw=save_raw_for_ab,
            )

            # Handle different return types
            if isinstance(result, tuple) and len(result) == 4:
                processed_path, seed_used, raw_path, diagnostics = result
                return processed_path, seed_used, raw_path
            else:
                processed_path, seed_used = result
                return processed_path, seed_used, None

        generate_btn.click(
            fn=do_generate,
            inputs=[
                text_input,
                speaker_audio,
                language,
                cfg_scale,
                speed,
                num_steps,
                seed,
                randomize_seed,
                enable_post_processing,
                auto_pitch,
                pitch_shift,
                eq_intensity,
                de_ess_intensity,
                compressor_threshold,
                compressor_ratio,
                compressor_knee,
                target_loudness,
                save_raw_for_ab,
            ],
            outputs=[output_audio, output_seed, output_raw],
            api_name="generate_audio",  # This is the endpoint SkyrimNet calls
        )

    return demo


async def main():
    """Main entry point."""
    # Load configuration
    config = AppConfig.from_cli()
    logger.info(f"Starting SkyrimNet-LuxTTS server on {config.host}:{config.port}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Model: {config.model_path}")

    # Load model
    load_model_if_needed(config)

    # Initialize speaker cache
    await init_speaker_cache(config)

    # Build interface
    demo = build_interface(config)

    # Launch server
    demo.queue(max_size=GENERATION_CONCURRENCY_LIMIT)
    demo.launch(
        server_name=config.host,
        server_port=config.port,
        share=config.share,
        inbrowser=config.inbrowser,
        show_error=True,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
