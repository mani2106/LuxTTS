"""Parse ElevenLabs-style [bracket] audio tags from text."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List


logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Type of audio segment."""
    SPEECH = "speech"
    VOCALIZATION = "vocalization"


@dataclass
class Segment:
    """A segment of text representing either speech or a vocalization tag.

    Attributes:
        type: SPEECH or VOCALIZATION
        text: The text content (for SPEECH segments)
        tag: The tag name (for VOCALIZATION segments), e.g., "sighs"
        original_text: The original text that produced this segment
    """
    type: SegmentType
    text: str = ""
    tag: str = ""
    original_text: str = ""

    def __repr__(self) -> str:
        if self.type == SegmentType.SPEECH:
            return f"Segment(SPEECH, text={self.text!r})"
        return f"Segment(VOCALIZATION, tag={self.tag!r})"


# Regex pattern to match [tags]
# Matches: [word] or [multi_word_phrase] or [with_underscores]
# Does NOT match: malformed like [unclosed, nested [[brackets]]
TAG_PATTERN = re.compile(r'\[([a-zA-Z_]+(?:\s+[a-zA-Z_]+)*)\]')


def parse_tags(text: str) -> List[Segment]:
    """
    Parse text into speech and vocalization segments.

    Args:
        text: Input text that may contain [bracket] tags

    Returns:
        List of Segment objects in order
    """
    segments = []
    last_end = 0

    for match in TAG_PATTERN.finditer(text):
        tag = match.group(1)
        start, end = match.span()

        # Skip if inside nested brackets (check for unmatched opening bracket before match)
        before_match = text[:start]
        if before_match.count('[') > before_match.count(']'):
            # We're inside an unclosed bracket, skip this match
            continue

        # Add speech segment before this tag if there is any
        if start > last_end:
            speech_text = text[last_end:start]
            # Skip if it's just whitespace between consecutive tags
            if speech_text and not speech_text.isspace():
                segments.append(Segment(
                    type=SegmentType.SPEECH,
                    text=speech_text,
                    original_text=speech_text
                ))

        # Add vocalization segment
        segments.append(Segment(
            type=SegmentType.VOCALIZATION,
            tag=tag,
            original_text=match.group(0)
        ))

        last_end = end

    # Add remaining text as speech segment
    if last_end < len(text):
        remaining = text[last_end:]
        # Strip leading space if it follows a tag (for cleaner text)
        if last_end > 0 and remaining.startswith(' '):
            remaining = remaining[1:]
        if remaining:  # Only add if non-empty
            segments.append(Segment(
                type=SegmentType.SPEECH,
                text=remaining,
                original_text=remaining
            ))

    return segments


def has_vocalizations(segments: List[Segment]) -> bool:
    """
    Check if any segment is a vocalization.

    Args:
        segments: List of Segment objects

    Returns:
        True if any VOCALIZATION segment is present
    """
    return any(seg.type == SegmentType.VOCALIZATION for seg in segments)
