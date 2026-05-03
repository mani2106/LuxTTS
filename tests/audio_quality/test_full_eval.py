"""Tier 2: Full evaluation tests (GPU required).

Run with: pytest tests/audio_quality/test_full_eval.py -v -m gpu
"""

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_full_eval_importable():
    """Verify full eval suite module is importable."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    assert len(FULL_EVAL_CASES) > 0
    for case in FULL_EVAL_CASES:
        assert case.name, f"Test case missing name: {case}"
        assert case.text, f"Test case '{case.name}' missing text"


def test_full_eval_case_has_reference_text():
    """Each full eval case that tests WER should have reference_text."""
    from tests.audio_quality.suites.full_eval import FULL_EVAL_CASES

    for case in FULL_EVAL_CASES:
        if not case.is_batch and not case.has_vocalization_tags:
            assert case.reference_text is not None, (
                f"Non-batch, non-vocalization case '{case.name}' needs reference_text for WER testing"
            )
