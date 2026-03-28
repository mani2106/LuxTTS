"""Tests for model_utils module."""

import pytest
from utilities.model_utils import _resolve_model_path
from pathlib import Path


def test_resolve_local_path():
    """Test that local paths are returned as-is."""
    # Test with fixtures directory (exists)
    fixtures_dir = Path(__file__).parent / "fixtures"
    result = _resolve_model_path(str(fixtures_dir))
    assert result == fixtures_dir

    # Test with models directory (created by app_constants)
    from utilities.app_constants import MODELS_DIR
    result = _resolve_model_path(str(MODELS_DIR))
    assert result == MODELS_DIR


def test_resolve_huggingface_id():
    """Test that HuggingFace IDs are sanitized."""
    result = _resolve_model_path("YatharthS/LuxTTS")
    assert "YatharthS--LuxTTS" in str(result)
    assert "models" in str(result)
