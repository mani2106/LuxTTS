"""Vocalization audio tag processing package.

This package provides ElevenLabs-style [bracket] audio tags for non-speech
vocalizations (sighs, screams, gasps, etc.) embedded within dialogue text.

Public API:
    - parse_tags(text) — Parse text into speech/vocalization segments
    - generate_vocalization(segment, model, config) — Generate audio for a vocalization segment
    - stitch_segments(segments, crossfade_ms) — Stitch audio segments with crossfades
"""

__all__ = [
    "parse_tags",
    "Segment",
    "SegmentType",
    "VocalizationGenerator",
    "stitch_segments",
]

# Lazy imports to allow modules to be added incrementally
def __getattr__(name):
    if name == "parse_tags":
        from utilities.vocalization.tag_parser import parse_tags
        return parse_tags
    if name == "Segment":
        from utilities.vocalization.tag_parser import Segment
        return Segment
    if name == "SegmentType":
        from utilities.vocalization.tag_parser import SegmentType
        return SegmentType
    if name == "VocalizationGenerator":
        from utilities.vocalization.vocalization_generator import VocalizationGenerator
        return VocalizationGenerator
    if name == "stitch_segments":
        from utilities.vocalization.stitcher import stitch_segments
        return stitch_segments
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
