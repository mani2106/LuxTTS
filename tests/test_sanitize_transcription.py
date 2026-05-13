"""Tests for Whisper transcription sanitization (hallucination collapse)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zipvoice.modeling_utils import _sanitize_transcription


class TestSanitizeTranscription:
    """Unit tests for _sanitize_transcription collapsing repeated characters."""

    def test_collapse_short_repeated_falls_back(self):
        # "SUU" is only 3 chars after collapse -> falls back to " speech"
        assert _sanitize_transcription("SUUUUUUUUUUUUU") == " speech"

    def test_collapse_repeated_h_falls_back(self):
        assert _sanitize_transcription("Ahhhhhhhhhhhh") == " speech"

    def test_collapse_repeated_o_falls_back(self):
        assert _sanitize_transcription("Noooooooooooo") == " speech"

    def test_collapse_dots_falls_back(self):
        assert _sanitize_transcription("...............") == " speech"

    def test_normal_text_unchanged(self):
        text = "Another member of the family!"
        assert _sanitize_transcription(text) == text

    def test_short_text_unchanged(self):
        text = "Let me go."
        assert _sanitize_transcription(text) == text

    def test_legitimate_elongation_preserved(self):
        # "sooooo" -> "soo" — 8 chars total, above threshold
        assert _sanitize_transcription("sooooo what?") == "soo what?"

    def test_whitespace_only_falls_back(self):
        result = _sanitize_transcription("   ")
        assert result == " speech"

    def test_empty_falls_back(self):
        result = _sanitize_transcription("")
        assert result == " speech"

    def test_long_text_capped(self):
        text = "a" * 500
        result = _sanitize_transcription(text)
        assert len(result) <= 200

    def test_mixed_repeated_and_normal(self):
        text = "Hello SUUUUUUU world"
        result = _sanitize_transcription(text)
        assert result == "Hello SUU world"

    def test_long_repeated_kept(self):
        # Long text with repeated chars but total length > 5 stays
        text = "Noooooooo way that happened"
        result = _sanitize_transcription(text)
        assert result == "Noo way that happened"


class TestSanitizeIntegration:
    """Integration: verify token counts are reasonable after sanitization."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from huggingface_hub import snapshot_download
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        model_path = snapshot_download('YatharthS/LuxTTS')
        return EmiliaTokenizer(token_file=f'{model_path}/tokens.txt')

    @pytest.mark.parametrize("raw_text,expected_max_tokens", [
        ("SUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU", 20),
        ("Ahhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh", 20),
        ("Nooooooooooooooooooooooooooooooooooooooooooooooooooooooo", 20),
        ("...........................................................", 20),
    ])
    def test_hallucinated_transcription_tokens_capped(self, tokenizer, raw_text, expected_max_tokens):
        cleaned = _sanitize_transcription(raw_text)
        tokens = tokenizer.texts_to_token_ids([cleaned])
        assert len(tokens[0]) <= expected_max_tokens, (
            f"Cleaned '{cleaned}' produced {len(tokens[0])} tokens, expected <= {expected_max_tokens}"
        )

    def test_normal_transcription_tokens_unchanged(self, tokenizer):
        text = "Another member of the family!"
        cleaned = _sanitize_transcription(text)
        tokens = tokenizer.texts_to_token_ids([cleaned])
        assert len(tokens[0]) > 0
