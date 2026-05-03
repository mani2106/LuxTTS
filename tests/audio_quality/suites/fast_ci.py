"""Tier 1: Fast CI test suite definitions.

Pure CPU, no GPU, no model loading. Runs on pre-generated audio fixtures.
Suitable for GitHub Actions free tier.
"""

from dataclasses import dataclass


@dataclass
class FastCITestCase:
    """A test case for the fast CI tier."""
    name: str
    description: str
    text: str  # Text that was used to generate the fixture audio
    has_vocalization_tags: bool = False
    is_batch: bool = False
    batch_count: int = 1
    expected_min_duration_s: float = 0.5
    expected_max_duration_s: float = 30.0


# Test cases covering the full range of TTS generation scenarios
FAST_CI_CASES = [
    FastCITestCase(
        name="short_text",
        description="Short text (<50 chars) — basic speech",
        text="Hello, how are you?",
        expected_min_duration_s=0.5,
        expected_max_duration_s=5.0,
    ),
    FastCITestCase(
        name="medium_text",
        description="Medium text (100-200 chars) — chunked generation",
        text="I've been traveling across Skyrim for many years now. "
        "The roads are dangerous, but the scenery never fails to take my breath away.",
        expected_min_duration_s=2.0,
        expected_max_duration_s=15.0,
    ),
    FastCITestCase(
        name="long_text",
        description="Long text (300+ chars) — multi-chunk",
        text="You know, when I first came to Riften, I thought it was the most beautiful "
        "city in all of Skyrim. The way the mist rises off the lake in the morning, the "
        "sound of the docks coming alive with merchants and fishermen. But then I learned "
        "about the Thieves Guild lurking beneath the city, and the corruption that runs "
        "deep through the Ratways. It changed my perspective entirely.",
        expected_min_duration_s=5.0,
        expected_max_duration_s=30.0,
    ),
    FastCITestCase(
        name="vocalization_sighs",
        description="Vocalization tag: [sighs]",
        text="[sighs] I can't believe we made it.",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_gasps",
        description="Vocalization tag: [gasps]",
        text="[gasps] Who's there?",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_whispers",
        description="Vocalization tag: [whispers]",
        text="[whispers] Don't make a sound.",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_screams",
        description="Vocalization tag: [screams]",
        text="[screams] Get away from me!",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="vocalization_pause",
        description="Vocalization tag: [pause]",
        text="Hello [pause] my old friend.",
        has_vocalization_tags=True,
    ),
    FastCITestCase(
        name="batch_sequential",
        description="5 sequential generations with same speaker — degradation test",
        text="The weather is nice today.",
        is_batch=True,
        batch_count=5,
    ),
    FastCITestCase(
        name="edge_all_caps",
        description="Edge case: ALL CAPS text (pitch shift trigger)",
        text="THIS IS AN EMERGENCY!",
    ),
    FastCITestCase(
        name="edge_question",
        description="Edge case: Question text (pitch shift trigger)",
        text="Where are you going?",
    ),
    FastCITestCase(
        name="edge_ellipsis",
        description="Edge case: Ellipsis text (pitch shift trigger)",
        text="I'm not so sure about that...",
    ),
]
