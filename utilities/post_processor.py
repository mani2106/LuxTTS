"""Audio post-processing pipeline for LuxTTS output quality improvement."""

import logging
import re
from typing import Optional

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)

# Optional dependency detection
try:
    import pedalboard
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

try:
    import pyloudnorm as pyln
    HAS_PLOUDNORM = True
except ImportError:
    HAS_PLOUDNORM = False


class PitchDetector:
    """
    Detects pitch offset from dialogue text using heuristic pattern matching.

    Rules evaluated in order (first match wins):
    1. Manual override provided → use provided value
    2. Text ALL CAPS (majority) → +2.0 (shouting)
    3. Text ends with '!' → +1.0 (excited)
    4. Text ends with '?' → +0.5 (questioning)
    5. Text contains '...' → -1.0 (hesitant/sad)
    6. Text length > 200 chars → -0.5 (narrative/calm)
    7. Default → 0.0 (neutral)
    """

    ALL_CAPS_THRESHOLD = 0.7  # 70% of characters must be uppercase
    LONG_TEXT_THRESHOLD = 200  # characters

    def detect_pitch(self, text: str, manual_pitch_shift: Optional[float] = None) -> float:
        """
        Detect pitch offset from dialogue text.

        Args:
            text: Dialogue text to analyze
            manual_pitch_shift: Manual override in semitones. If provided,
                all text analysis is skipped.

        Returns:
            Pitch offset in semitones (positive = higher, negative = lower)
        """
        # Manual override takes precedence
        if manual_pitch_shift is not None:
            return float(manual_pitch_shift)

        if not text:
            return 0.0

        # Rule: ALL CAPS majority → shouting
        if self._is_all_caps(text):
            return 2.0

        # Rule: Ends with exclamation → excited
        if text.rstrip().endswith('!'):
            return 1.0

        # Rule: Ends with question → questioning
        if text.rstrip().endswith('?'):
            return 0.5

        # Rule: Contains ellipsis → hesitant
        if '...' in text:
            return -1.0

        # Rule: Long text → narrative/calm
        if len(text) > self.LONG_TEXT_THRESHOLD:
            return -0.5

        # Default: neutral
        return 0.0

    def _is_all_caps(self, text: str) -> bool:
        """Check if majority of alphanumeric characters are uppercase."""
        # Remove whitespace and punctuation for analysis
        alnum_chars = [c for c in text if c.isalnum()]
        if not alnum_chars:
            return False

        uppercase_count = sum(1 for c in alnum_chars if c.isupper())
        ratio = uppercase_count / len(alnum_chars)
        return ratio >= self.ALL_CAPS_THRESHOLD
