"""Load and manage vocalization DSP recipes from JSON config."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)

# Path to recipes JSON
RECIPES_PATH = Path(__file__).parent / "recipes.json"

# Cache for loaded recipes
_recipes_cache: Optional[Dict[str, Dict]] = None


def load_recipes() -> Dict[str, Dict]:
    """
    Load all recipes from JSON config file.

    Returns:
        Dictionary mapping tag names to recipe definitions

    Raises:
        FileNotFoundError: If recipes.json doesn't exist
        json.JSONDecodeError: If recipes.json is invalid JSON
    """
    global _recipes_cache

    if _recipes_cache is not None:
        return _recipes_cache

    if not RECIPES_PATH.exists():
        raise FileNotFoundError(f"Recipes file not found: {RECIPES_PATH}")

    with open(RECIPES_PATH, 'r', encoding='utf-8') as f:
        _recipes_cache = json.load(f)

    logger.debug(f"Loaded {len(_recipes_cache)} recipes from {RECIPES_PATH}")
    return _recipes_cache


def get_recipe(tag: str) -> Optional[Dict]:
    """
    Get recipe for a specific tag.

    Args:
        tag: Tag name (e.g., "sighs", "breathes heavily")

    Returns:
        Recipe dict or None if tag not found
    """
    recipes = load_recipes()
    return recipes.get(tag)


def reload_recipes() -> Dict[str, Dict]:
    """
    Force reload recipes from disk (clears cache).

    Returns:
        Dictionary mapping tag names to recipe definitions
    """
    global _recipes_cache
    _recipes_cache = None
    return load_recipes()
