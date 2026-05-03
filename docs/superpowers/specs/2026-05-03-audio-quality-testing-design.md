# Audio Quality Testing Framework Design

Date: 2026-05-03
Branch: audio-quality-testing (new, from master — merged independently)

## Problem Statement

LuxTTS has no automated audio quality measurement. The existing test suite covers unit-level DSP correctness (individual post-processor methods with synthetic audio) but cannot answer:

1. **Does the TTS output sound good?** (naturalness, signal quality)
2. **Does the cloned voice match the reference?** (speaker similarity)
3. **Is the speech intelligible?** (word error rate)
4. **Do code changes improve or regress quality?** (regression detection)
5. **Do vocalization tags produce appropriate non-speech sounds?** (tag quality)

The `expressive-vocalizations` branch added post-processing, vocalization tags, and core model fixes. We need to measure the impact of these changes against `master`'s raw TTS output.

## Design

### Architecture

Three-layer system:

```
┌─────────────────────────────────────────────────┐
│              Test Runner / CLI                   │
│  pytest fixtures + eval_audio_quality.py CLI     │
├─────────────────────────────────────────────────┤
│              Evaluation Framework                │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ VERSA Metrics │  │ Custom LuxTTS Scorers    │ │
│  │              │  │                          │ │
│  │ - UTMOSv2    │  │ - Batch Degradation      │ │
│  │ - DNSMOS     │  │ - Vocalization Quality   │ │
│  │ - Speaker    │  │ - Post-Processing Delta  │ │
│  │   Similarity │  │ - Silence/Artifact       │ │
│  │ - WER        │  │   Detection              │ │
│  └──────────────┘  │ - Chunking Quality       │ │
│                    └──────────────────────────┘ │
├─────────────────────────────────────────────────┤
│              Test Audio Sources                  │
│                                                  │
│  ┌────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Pre-gen    │  │ On-the-fly  │  │ Regres-  │ │
│  │ Reference  │  │ Generation  │  │ sion     │ │
│  │ Fixtures   │  │ (GPU)       │  │ Baselines│ │
│  └────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────┘
```

### File Structure

```
tests/
  audio_quality/                          # New evaluation module
    __init__.py
    conftest.py                           # Shared fixtures
    scorers/
      __init__.py
      versa_scorer.py                     # VERSA integration wrapper
      custom_scorers.py                   # LuxTTS-specific metrics
      scorer_registry.py                  # Compose metric suites
    suites/
      __init__.py
      fast_ci.py                          # Tier 1: lightweight, no GPU
      full_eval.py                        # Tier 2: comprehensive, GPU
      regression.py                       # Tier 3: baseline comparison
    baselines/                            # Reference audio + scores
      master_baseline.json                # Baseline from master branch (raw TTS)
    test_fast_ci.py                       # pytest tests for Tier 1
    test_full_eval.py                     # pytest tests for Tier 2 (@pytest.mark.gpu)
    test_regression.py                    # pytest tests for Tier 3
  eval_audio_quality.py                   # CLI runner
requirements-test.txt                     # Test-only dependencies
```

### Metrics

#### VERSA Standard Metrics

| Metric | What it measures | Scale | Tier |
|--------|-----------------|-------|------|
| UTMOSv2 | Perceived naturalness | 1-5 MOS | Full eval |
| DNSMOS (SIG) | Signal quality | 1-5 | Both |
| DNSMOS (BAK) | Background noise | 1-5 | Both |
| DNSMOS (OVRL) | Overall quality | 1-5 | Both |
| Speaker Similarity | Cosine similarity between reference and generated voice | 0-1 | Full eval |
| WER | Word error rate via Whisper transcription | 0-1 | Full eval |

#### Custom LuxTTS Scorers

| Scorer | What it measures | Tier | Method |
|--------|-----------------|------|--------|
| Batch Degradation | Voice quality drift across sequential same-speaker generations | Full eval | Generate N utterances, compute speaker similarity + MOS delta between first and last |
| Vocalization Tag Quality | Do tags sound distinct from speech? Correct duration? | Full eval | Spectral centroid distance vs plain speech, duration bounds check |
| Post-Processing Delta | Does the DSP chain improve or hurt quality? | Both | Run VERSA metrics on raw TTS vs post-processed output, compute delta |
| Silence/Artifact Detector | Trailing silence >500ms, clipping, zero-sample regions | Fast CI | RMS envelope, zero-crossing rate, peak detection |
| Chunking Quality | Crossfade artifacts at chunk boundaries | Full eval | Energy dip detection at expected boundary timestamps |

### Test Tiers

#### Tier 1: Fast CI (< 2 min, pure CPU, no GPU)

**Trigger:** Every PR, every push. Must run on GitHub Actions free tier (CPU-only runner).
**Constraint:** No model loading, no GPU, no heavy ML inference. Pure signal analysis on pre-generated audio.
**Method:** Load pre-generated WAV fixtures, run lightweight metrics.

Test cases (pre-generated):
- Short text (< 50 chars) — basic speech
- Medium text (100-200 chars) — chunked generation
- Long text (300+ chars) — multi-chunk
- Vocalization tags: `[sighs]`, `[gasps]`, `[whispers]`, `[screams]`, `[pause]`
- Batch test: 5 sequential generations from same speaker
- Edge cases: all-caps, question, ellipsis

Metrics: DNSMOS (ONNX, CPU), silence/artifact detector, post-processing delta on fixtures.
Pass criterion: No metric drops >5% from baseline.

**GitHub CI compatible:** Yes. All Tier 1 tests run on CPU-only runners. DNSMOS ONNX model is ~5MB and runs in ~200ms per sample on CPU.

#### Tier 2: Full Evaluation (~10-15 min, GPU)

**Trigger:** Nightly, manual, pre-release.
**Method:** Load LuxTTS model, generate fresh audio, run full VERSA suite + custom scorers.

Metrics: UTMOSv2, DNSMOS, speaker similarity, WER, batch degradation, vocalization quality, chunking quality, post-processing delta.
Output: JSON report + optional HTML summary.

Marked with `@pytest.mark.gpu` and `@pytest.mark.slow`.

#### Tier 3: Regression Baselines

**Trigger:** After significant changes, pre-release.
**Method:** Generate samples from current code, score, compare against stored baselines.

Baseline sources:
- `master_baseline.json` — scores from master branch raw TTS (no post-processing, no vocalizations)
- Additional baselines can be generated from any branch/commit

Comparison modes:
1. Raw TTS comparison: `enable_post_processing=False`, no tags — compare core model output vs master baseline
2. Post-processed comparison: `enable_post_processing=True`, no tags — measure DSP chain impact
3. Vocalization comparison: with tags — measure vocalization quality

CLI:
```bash
python tests/eval_audio_quality.py --generate-baselines   # Create/update baseline
python tests/eval_audio_quality.py --compare-baselines     # Compare current vs baseline
python tests/eval_audio_quality.py --report                # Full HTML report
```

### Regression Baseline Format

```json
{
  "version": "master-v1",
  "commit": "abc1234",
  "branch": "master",
  "date": "2026-05-03",
  "config": {
    "enable_post_processing": false,
    "num_steps": 4,
    "guidance_scale": 3.0
  },
  "samples": {
    "hello_world": {
      "text": "Hello, how are you today?",
      "speaker": "speaker_01",
      "utmos": 3.85,
      "dnsmos_sig": 3.92,
      "dnsmos_bak": 4.10,
      "dnsmos_ovrl": 3.78,
      "speaker_similarity": 0.89,
      "wer": 0.0,
      "duration_s": 2.1,
      "silence_ratio": 0.08
    }
  }
}
```

### Dependencies

**Test-only** (`requirements-test.txt`):
```
# VERSA evaluation toolkit (UTMOS, DNSMOS, speaker similarity, WER)
versa-speech-audio-toolkit

# Speaker embeddings for custom similarity checks
resemblyzer

# Test infrastructure
pytest
pytest-timeout
pytest-xdist

# Reporting (optional)
jinja2
```

**No changes to production dependencies.**

### Refactoring for Testability

Minimal changes needed — the pipeline is already well-structured:

1. `generate_audio()` already accepts `enable_post_processing` flag — no change needed
2. `AudioPostProcessor.process()` already individually callable — no change needed
3. Vocalization pipeline only triggers on `[bracket]` tags — no change needed
4. Model loading is already a singleton via `model_utils.py` — expose as pytest fixture

### Branch Strategy

- Create `audio-quality-testing` branch from `master`
- Merge independently of `expressive-vocalizations`
- Both branches can be evaluated against each other via the regression system

## Out of Scope

- Human MOS evaluation infrastructure
- Training data quality assessment
- Real-time/streaming quality monitoring
- Model fine-tuning evaluation
- CI/CD pipeline configuration (users run tests manually or in their own CI)

## Testing the Testing Framework

Meta-tests:
- Verify VERSA scorer produces reasonable scores on known audio samples
- Verify custom scorers produce expected results on synthetic signals
- Verify baseline comparison detects a known-intentional regression
- Verify CLI runner exits with correct codes
