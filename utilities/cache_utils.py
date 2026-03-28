"""Two-tier caching for speaker embeddings."""

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Dict

import torch


logger = logging.getLogger(__name__)

# Thread-safe memory cache
_memory_cache: Dict[str, Any] = {}
_cache_lock = threading.RLock()


def get_audio_file_hash(audio_path: Path) -> str:
    """
    Compute SHA256 hash of audio file bytes.

    Args:
        audio_path: Path to audio file

    Returns:
        Hex digest hash string
    """
    sha256 = hashlib.sha256()
    with open(audio_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_cached_embedding(cache_key: str, cache_dir: Path):
    """
    Retrieve embedding from cache (memory → disk).

    Args:
        cache_key: Cache key (hash string)
        cache_dir: Disk cache directory

    Returns:
        Cached embedding or None if not found
    """
    # Check memory cache first
    with _cache_lock:
        if cache_key in _memory_cache:
            logger.debug(f"Memory cache hit: {cache_key[:8]}...")
            return _memory_cache[cache_key]

    # Check disk cache
    cache_file = cache_dir / f"{cache_key}.pt"
    if cache_file.exists():
        logger.debug(f"Disk cache hit: {cache_key[:8]}...")
        try:
            embedding = torch.load(cache_file, weights_only=False)
            # Store in memory for next access
            with _cache_lock:
                _memory_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")

    return None


def cache_embedding(cache_key: str, embedding: Any, cache_dir: Path):
    """
    Store embedding in cache (memory + disk).

    Args:
        cache_key: Cache key (hash string)
        embedding: Embedding to cache
        cache_dir: Disk cache directory
    """
    # Store in memory
    with _cache_lock:
        _memory_cache[cache_key] = embedding

    # Store on disk
    cache_file = cache_dir / f"{cache_key}.pt"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embedding, cache_file)
        logger.debug(f"Cached embedding: {cache_key[:8]}...")
    except Exception as e:
        logger.warning(f"Failed to save cache file {cache_file}: {e}")


def clear_memory_cache():
    """Clear all entries from memory cache."""
    with _cache_lock:
        _memory_cache.clear()
    logger.debug("Memory cache cleared")
