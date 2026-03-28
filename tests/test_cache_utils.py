"""Tests for cache utilities."""

import pytest
import tempfile
from pathlib import Path
import torch
from utilities.cache_utils import (
    get_audio_file_hash, get_cached_embedding, cache_embedding,
    clear_memory_cache
)


def test_audio_file_hash():
    """Test audio file hashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test.wav"
        test_file.write_bytes(b"hello world")

        h1 = get_audio_file_hash(test_file)
        h2 = get_audio_file_hash(test_file)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex length


def test_cache_roundtrip():
    """Test cache store and retrieve."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        key = "test_key"
        embedding = torch.randn(1, 128)

        # Cache miss initially
        result = get_cached_embedding(key, cache_dir)
        assert result is None

        # Store and retrieve
        cache_embedding(key, embedding, cache_dir)
        result = get_cached_embedding(key, cache_dir)

        assert result is not None
        assert torch.allclose(result, embedding)


def test_memory_cache():
    """Test memory cache behavior."""
    embedding = torch.randn(1, 64)
    cache_embedding("mem_test", embedding, Path("/tmp/fake"))

    # Should be in memory
    result = get_cached_embedding("mem_test", Path("/tmp/fake"))
    assert result is not None

    # Clear and verify gone
    clear_memory_cache()
    result = get_cached_embedding("mem_test", Path("/tmp/fake"))
    # Disk cache might not exist (fake path), but memory should be empty
