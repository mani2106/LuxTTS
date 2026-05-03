"""Orchestrate TTS generation and DSP processing for vocalization tags."""

import logging
from typing import Dict

import numpy as np
import torch


from utilities.vocalization.recipes import get_recipe
from utilities.vocalization.dsp_engine import DSPEngine
from utilities.app_constants import SAMPLE_RATE


logger = logging.getLogger(__name__)


class VocalizationGenerator:
    """
    Generate audio for vocalization tags using TTS + DSP processing.

    For each tag:
    1. Load recipe (tts_text, effects, params)
    2. Generate speech using LuxTTS with recipe's tts_text
    3. Apply DSP effect chain from recipe
    4. Return processed audio

    Special cases:
    - "pause" tag: returns silence of specified duration
    - Unknown tags: returns 0.3s silence with warning log
    - whisper mode: recipe has mode="modify_speech" (handled elsewhere)
    """

    def __init__(
        self,
        model,
        num_steps: int = 8,
        guidance_scale: float = 3.0,
        speed: float = 1.0,
        t_shift: float = 0.9,
        return_smooth: bool = True,
    ):
        """
        Initialize vocalization generator.

        Args:
            model: LuxTTS model instance
            num_steps: Flow matching steps
            guidance_scale: CFG scale
            speed: Speech speed multiplier
            t_shift: Sampling shift
            return_smooth: If True, disable 48k upsampling
        """
        self.model = model
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.speed = speed
        self.t_shift = t_shift
        self.return_smooth = return_smooth
        self.dsp_engine = DSPEngine()

    def generate(self, segment, encode_dict: Dict) -> tuple[np.ndarray, int]:
        """
        Generate audio for a vocalization segment.

        Args:
            segment: Segment with type=VOCALIZATION
            encode_dict: Speaker encoding from LuxTTS.encode_prompt()

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        tag = segment.tag
        recipe = get_recipe(tag)

        if recipe is None:
            logger.warning(f"Unknown vocalization tag: [{tag}], treating as 0.3s pause")
            return self._create_silence(0.3), SAMPLE_RATE

        # Handle pause tag (special case)
        if recipe.get("tts_text") is None and "duration_s" in recipe:
            duration = recipe["duration_s"]
            return self._create_silence(duration), SAMPLE_RATE

        # Get TTS text and params from recipe
        tts_text = recipe["tts_text"]
        if tts_text is None:
            # Modify speech mode (e.g., whispers) - shouldn't reach here
            logger.warning(f"Tag [{tag}] has no tts_text, treating as 0.3s pause")
            return self._create_silence(0.3), SAMPLE_RATE

        # Generate TTS audio
        tts_speed = recipe.get("tts_speed", self.speed)
        audio = self._generate_tts(tts_text, encode_dict, tts_speed)

        # Convert to numpy
        if hasattr(audio, 'numpy'):
            audio = audio.numpy()
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        audio = audio.astype(np.float32).squeeze()

        # Apply max duration if specified
        max_duration = recipe.get("max_duration_s")
        if max_duration and len(audio) > int(max_duration * SAMPLE_RATE):
            audio = audio[:int(max_duration * SAMPLE_RATE)]

        # Apply DSP effects
        effects = recipe.get("effects", [])
        if effects:
            audio = self.dsp_engine.apply_chain(audio, SAMPLE_RATE, effects)

        return audio, SAMPLE_RATE

    def is_modify_speech(self, tag: str) -> bool:
        """
        Check if tag operates in modify-speech mode (e.g., whispers).

        Args:
            tag: Tag name

        Returns:
            True if tag modifies following speech instead of generating standalone audio
        """
        recipe = get_recipe(tag)
        if recipe is None:
            return False
        return recipe.get("mode") == "modify_speech"

    def _generate_tts(self, text: str, encode_dict: Dict, speed: float) -> torch.Tensor:
        """Generate speech using LuxTTS model."""
        return self.model.generate_speech(
            text=text,
            encode_dict=encode_dict,
            num_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            speed=speed,
            t_shift=self.t_shift,
            return_smooth=self.return_smooth,
        )

    def _create_silence(self, duration_sec: float) -> np.ndarray:
        """Create silence audio array."""
        num_samples = int(duration_sec * SAMPLE_RATE)
        return np.zeros(num_samples, dtype=np.float32)
