"""Tier 2: Full evaluation test case definitions.

Runs with GPU. Generates fresh audio and runs full VERSA + custom metric suite.
Marked with @pytest.mark.gpu and @pytest.mark.slow.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FullEvalTestCase:
    """A test case for the full evaluation tier."""
    name: str
    description: str
    text: str
    speaker: Optional[str] = None  # Speaker preset name, or None for default
    enable_post_processing: bool = True
    has_vocalization_tags: bool = False
    is_batch: bool = False
    batch_count: int = 1
    reference_text: Optional[str] = None  # For WER, if different from text


FULL_EVAL_CASES = [
    FullEvalTestCase(
        name="basic_speech",
        description="Basic short speech with post-processing",
        text="Hello, how are you doing today?",
        reference_text="Hello, how are you doing today?",
    ),
    FullEvalTestCase(
        name="raw_tts_no_postproc",
        description="Raw TTS output without post-processing",
        text="Hello, how are you doing today?",
        enable_post_processing=False,
        reference_text="Hello, how are you doing today?",
    ),
    FullEvalTestCase(
        name="medium_text_chunked",
        description="Medium text that triggers punctuation chunking",
        text="I've been traveling across Skyrim for many years now. "
        "The roads are dangerous, but the scenery never fails to take my breath away.",
        reference_text="I've been traveling across Skyrim for many years now. "
        "The roads are dangerous, but the scenery never fails to take my breath away.",
    ),
    FullEvalTestCase(
        name="long_text_multi_chunk",
        description="Long text requiring multiple chunks",
        text="You know, when I first came to Riften, I thought it was the most beautiful "
        "city in all of Skyrim. The way the mist rises off the lake in the morning, the "
        "sound of the docks coming alive with merchants and fishermen. But then I learned "
        "about the Thieves Guild lurking beneath the city, and the corruption that runs "
        "deep through the Ratways. It changed my perspective entirely.",
        reference_text="You know, when I first came to Riften, I thought it was the most beautiful "
        "city in all of Skyrim. The way the mist rises off the lake in the morning, the "
        "sound of the docks coming alive with merchants and fishermen. But then I learned "
        "about the Thieves Guild lurking beneath the city, and the corruption that runs "
        "deep through the Ratways. It changed my perspective entirely.",
    ),
    FullEvalTestCase(
        name="vocalization_sighs",
        description="Vocalization: [sighs] with speech",
        text="[sighs] I can't believe we made it.",
        has_vocalization_tags=True,
        reference_text="I can't believe we made it.",
    ),
    FullEvalTestCase(
        name="vocalization_gasps",
        description="Vocalization: [gasps] with speech",
        text="[gasps] Who's there?",
        has_vocalization_tags=True,
        reference_text="Who's there?",
    ),
    FullEvalTestCase(
        name="vocalization_whispers",
        description="Vocalization: [whispers] modifies following speech",
        text="[whispers] Don't make a sound.",
        has_vocalization_tags=True,
        reference_text="Don't make a sound.",
    ),
    FullEvalTestCase(
        name="vocalization_screams",
        description="Vocalization: [screams] with speech",
        text="[screams] Get away from me!",
        has_vocalization_tags=True,
        reference_text="Get away from me!",
    ),
    FullEvalTestCase(
        name="batch_degradation_5",
        description="5 sequential generations — degradation test",
        text="The weather is quite pleasant today.",
        is_batch=True,
        batch_count=5,
        reference_text="The weather is quite pleasant today.",
    ),
]
