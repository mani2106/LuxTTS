# Audio Quality Testing Refactor Design

Date: 2026-05-03
Branch: audio-quality-test (restructure)
Replaces: Original tiered framework (same branch)

## Problem

The current audio quality testing framework has a structural problem: GPU tests both generate audio AND score it, duplicating the scorer registry's score collection with an ad-hoc `_collected_scores` dict and `pytest_sessionfinish` hook. This makes the tests harder to maintain and prevents the CLI from being the single source of truth for quality analysis.

The tests also only use one speaker (`cicero.wav`), which means they can't catch quality issues that only appear with certain voice types (non-human voices, coarse voices, mature voices).

## Design

### Core Principle: Tests Gate, CLI Scores

Tests are responsible for one thing: generating audio and checking hard constraints (no clipping, not silent, reasonable duration). The CLI is responsible for everything else: running metrics, comparing baselines, producing reports.

This separation means:
- Tests are thin and fast — generate + gate + save
- Score collection lives in exactly one place (the CLI)
- The scorer registry is used by the CLI, not duplicated in tests
- An agent can re-score existing audio without re-running GPU tests

### Speaker Coverage

Tests parametrize across diverse voices to catch quality issues across voice types:

| Speaker | Category | Rationale |
|---------|----------|-----------|
| `cicero` | Human male, distinctive | Expressive, high pitch variation |
| `malecommoner` | Human male, standard | Baseline male |
| `femalenord` | Human female, standard | Baseline female |
| `serana` | Human female, high quality | Clean reference |
| `aaaharleyvoicequest` | Unique | Unusual characteristics, stress test |
| `alduin` | Beast (dragon) | Coarsest voice, hardest to clone |
| `maleargonian` | Non-human male | Reptilian voice |
| `femaleargonian` | Non-human female | Reptilian female |
| `malekhajiit` | Non-human male | Cat-like voice |
| `femalekhajiit` | Non-human female | Cat-like voice |
| `maleoldgrumpy` | Mature male | Age-related characteristics |
| `femaleoldgrumpy` | Mature female | Age-related characteristics |

This covers: male/female, young/mature, human/non-human, standard/extreme. Non-human voices (argonian, khajiit, alduin) are stress tests — they may score lower but gates should still pass.

### Test Layer: Gate-Only GPU Tests

**File:** `tests/audio_quality/test_full_eval.py`

Each test:
1. Generates audio using a speaker sample via `_run_generate()`
2. Runs gate checks (hard floors)
3. Saves the generated WAV file to `tests/audio_quality/output/`
4. Appends metadata to a session manifest

**Gate checks (hard constraints that cause test failure):**
- No clipping (`peak_amplitude < 0.99`)
- Trailing silence < 800ms
- Silence ratio < 50%
- Duration between 0.3s and 30s
- For vocalizations: duration >= 0.3s, RMS > 0.001

**Manifest format** (`tests/audio_quality/output/manifest.json`):
```json
[
  {
    "test_name": "basic_speech",
    "speaker": "cicero",
    "speaker_ref_path": "speakers/en/cicero.wav",
    "text": "Hello, how are you doing today?",
    "enable_post_processing": true,
    "audio_path": "tests/audio_quality/output/basic_speech_cicero.wav",
    "seed": 42,
    "duration_s": 1.8,
    "passed_gates": true,
    "gate_results": {
      "has_clipping": false,
      "trailing_silence_ms": 45.2,
      "silence_ratio": 0.05,
      "peak_amplitude": 0.85
    }
  }
]
```

The manifest is the contract between tests and CLI. Tests write it, CLI reads it.

**Test structure:**
- `conftest.py` provides session fixtures: `output_dir`, `manifest` (list that gets written at session finish)
- `conftest.py` provides `speaker_parametrize` fixture with all 12 speakers
- Tests use `@pytest.mark.parametrize("speaker", SPEAKER_LIST)` to generate audio per voice
- Structural tests (import checks, case validation) remain unchanged

**What gets removed from test_full_eval.py:**
- `_collected_scores` dict
- `_record_score()` function
- `pytest_sessionfinish()` hook
- All imports of `versa_scorer` (score_dnsmos, score_speaker_similarity, score_wer)
- All imports of `custom_scorers` except `detect_silence_artifacts` (used for gates)
- Post-processing delta test (moves to CLI)
- Speaker similarity test (moves to CLI)
- WER/intelligibility test (moves to CLI)
- Batch degradation test (moves to CLI — it's a measurement, not a gate)

**What stays in test_full_eval.py:**
- `_run_generate()` helper
- `_load_generated_audio()` helper
- `speaker_audio_path` / `speaker_audio_array` fixtures
- Gate checks within `test_generate_and_score_quality` (renamed to `test_generate_audio_gates`)
- Vocalization generation tests (duration + RMS + no-clipping gates)
- Batch degradation generation test — generates 5 clips, saves all to manifest, gates on duration/RMS. The CLI scores the drift between them.
- Structural tests (importable, reference_text)

### CLI Scoring Engine

**File:** `tests/eval_audio_quality.py`

Three commands. Zero config. Agent runs them in sequence.

```
python tests/eval_audio_quality.py score         # Score all audio in manifest
python tests/eval_audio_quality.py compare        # Compare scores to baseline
python tests/eval_audio_quality.py save-baseline  # Save current scores as baseline
```

#### `score`

Reads the manifest, loads each audio file, runs metrics, saves to `latest_scores.json`.

Metrics run per sample:
- DNSMOS (SIG, BAK, OVRL) via speechmos ONNX — CPU, ~200ms/sample
- Speaker similarity (generated vs reference) — VERSA, GPU recommended
- Post-processing delta (if both raw and processed versions exist) — DNSMOS on both
- Silence artifacts (trailing silence, clipping, silence ratio)

Output: `tests/audio_quality/baselines/latest_scores.json` using `results_to_json()` from scorer_registry.

Console output shows a readable table:
```
SCORING RESULTS
===============
basic_speech_cicero:       SIG=3.45 BAK=3.98 OVRL=3.12  SIM=0.35  PASS
basic_speech_alduin:       SIG=2.89 BAK=3.45 OVRL=2.67  SIM=0.15  PASS
basic_speech_malekhajiit:  SIG=3.12 BAK=3.78 OVRL=2.95  SIM=0.22  PASS
...
```

#### `compare`

Loads `latest_scores.json` and the baseline file, runs `BaselineManager.compare()` per sample, outputs regression report. Exits with code 1 if any metric regressed >5%.

Baseline file: `tests/audio_quality/baselines/master_baseline.json`

Console output shows regressions with actionable details:
```
REGRESSION CHECK
================
Comparing 36 samples against master_baseline (commit abc1234, 2026-05-03)

PASS: 34/36 samples within 5% of baseline

REGRESSION in basic_speech_malekhajiit:
  dnsmos_sig: 3.12 -> 2.85 (-8.7%, threshold: -5%)
  speaker_similarity: 0.22 -> 0.18 (-18.2%, threshold: -5%)

REGRESSION in basic_speech_alduin:
  dnsmos_ovrl: 2.67 -> 2.41 (-9.7%, threshold: -5%)
```

#### `save-baseline`

Copies `latest_scores.json` to `baselines/<name>.json` with metadata (commit, branch, date).

Usage: `python tests/eval_audio_quality.py save-baseline master_v2`

This creates `tests/audio_quality/baselines/master_v2.json`.

### File Changes Summary

| File | Action | What changes |
|------|--------|-------------|
| `tests/audio_quality/test_full_eval.py` | Rewrite | Remove scoring logic, add manifest output, parametrize speakers |
| `tests/audio_quality/conftest.py` | Modify | Add output_dir, manifest fixtures, SPEAKER_LIST |
| `tests/eval_audio_quality.py` | Rewrite | Replace multi-flag CLI with 3-verb CLI (score/compare/save-baseline) |
| `tests/audio_quality/scorers/scorer_registry.py` | Unchanged | Already has ScoreResult, results_to_json, compare_scores |
| `tests/audio_quality/scorers/versa_scorer.py` | Unchanged | Already working correctly |
| `tests/audio_quality/scorers/custom_scorers.py` | Unchanged | Already working correctly |
| `tests/audio_quality/suites/full_eval.py` | Unchanged | Test case definitions stay the same |
| `tests/audio_quality/test_fast_ci.py` | Unchanged | CPU tests unaffected |
| `tests/audio_quality/test_regression.py` | Unchanged | Baseline comparison tests unaffected |

### Agent Workflow

The intended workflow for an agent improving TTS quality:

```
1. Make code changes
2. pytest tests/audio_quality/ -m gpu          # Gates: did I break anything?
3. python tests/eval_audio_quality.py score     # Score: how does it measure?
4. python tests/eval_audio_quality.py compare   # Compare: better or worse?
5. If scores improved: save-baseline, commit
6. If scores regressed: investigate, fix, repeat
```

Re-scoring without re-generation (e.g., after adding a new metric):
```
python tests/eval_audio_quality.py score    # Re-scores existing audio from manifest
```

### What This Removes

- Ad-hoc `_collected_scores` dict in test_full_eval.py
- `pytest_sessionfinish` hook in test_full_eval.py
- Duplicate score collection logic
- Post-processing delta, speaker similarity, WER from tests (moved to CLI `score` command)
- Batch degradation scoring from tests (generation stays in tests, scoring moves to CLI)

### What This Adds

- 12-speaker parametrized gate tests (up from 1)
- Session manifest as contract between tests and CLI
- `score` CLI command for running all metrics on generated audio
- `save-baseline` CLI command for saving named baselines
- Speaker reference paths in manifest for similarity scoring

### Out of Scope

- Changes to scorer implementations (versa_scorer, custom_scorers)
- Changes to Tier 1 (test_fast_ci.py) or Tier 3 (test_regression.py)
- New metrics or scoring functions
- CI/CD pipeline configuration
