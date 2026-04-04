"""Tests for vocalization audio tag processing."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from utilities.vocalization.tag_parser import parse_tags, Segment, SegmentType


def test_parse_no_tags():
    """Text without tags returns single speech segment."""
    segments = parse_tags("Hello world")
    assert len(segments) == 1
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "Hello world"


def test_parse_single_tag_at_start():
    """Tag at start splits correctly."""
    segments = parse_tags("[sighs] Hello there")
    assert len(segments) == 2
    assert segments[0].type == SegmentType.VOCALIZATION
    assert segments[0].tag == "sighs"
    assert segments[1].type == SegmentType.SPEECH
    assert segments[1].text == "Hello there"


def test_parse_tag_in_middle():
    """Tag in middle splits into three segments."""
    segments = parse_tags("Hello [gasps] how are you?")
    assert len(segments) == 3
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "Hello "
    assert segments[1].type == SegmentType.VOCALIZATION
    assert segments[1].tag == "gasps"
    assert segments[2].type == SegmentType.SPEECH
    assert segments[2].text == "how are you?"


def test_parse_tag_at_end():
    """Tag at end splits correctly."""
    segments = parse_tags("Goodbye [sighs]")
    assert len(segments) == 2
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "Goodbye "
    assert segments[1].type == SegmentType.VOCALIZATION
    assert segments[1].tag == "sighs"


def test_parse_multiple_consecutive_tags():
    """Multiple consecutive tags each become separate segments."""
    segments = parse_tags("[gasps] [screams] Help!")
    assert len(segments) == 3
    assert segments[0].tag == "gasps"
    assert segments[1].tag == "screams"
    assert segments[2].type == SegmentType.SPEECH
    assert segments[2].text == "Help!"


def test_parse_multi_word_tag():
    """Multi-word tags are supported."""
    segments = parse_tags("[breathes heavily] We must go.")
    assert len(segments) == 2
    assert segments[0].tag == "breathes heavily"
    assert segments[1].text == "We must go."


def test_parse_unknown_tag_treated_as_pause():
    """Unknown tags become vocalization segments with tag name preserved."""
    segments = parse_tags("[unknown_tag] Hello")
    assert len(segments) == 2
    assert segments[0].type == SegmentType.VOCALIZATION
    assert segments[0].tag == "unknown_tag"


def test_parse_malformed_tag_treated_as_text():
    """Malformed tags (missing bracket, nested) are treated as regular text."""
    # Missing closing bracket
    segments = parse_tags("[sighs Hello world")
    assert len(segments) == 1
    assert segments[0].type == SegmentType.SPEECH
    assert segments[0].text == "[sighs Hello world"

    # Nested brackets
    segments = parse_tags("[sighs[gasps]] Hello")
    assert len(segments) == 1
    assert segments[0].type == SegmentType.SPEECH


def test_parse_empty_after_tags():
    """Text with only tags returns only vocalization segments."""
    segments = parse_tags("[sighs] [gasps]")
    assert len(segments) == 2
    assert all(s.type == SegmentType.VOCALIZATION for s in segments)


def test_has_vocalizations():
    """has_vocalizations() helper returns True when any vocalization segment present."""
    from utilities.vocalization.tag_parser import has_vocalizations

    assert has_vocalizations(parse_tags("[sighs] Hello"))
    assert has_vocalizations(parse_tags("Hello [gasps]"))
    assert not has_vocalizations(parse_tags("Hello world"))
