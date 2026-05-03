# Audio Quality Testing: Guide

## Quick Reference

```bash
# Install test dependencies (first time only)
uv pip install -r requirements-test.txt

# Step 1: Run GPU gate tests (generates audio + manifest)
pytest tests/audio_quality/test_full_eval.py -v -m gpu

# Step 2: Score the generated audio
python tests/eval_audio_quality.py score

# Step 3: Compare against baseline
python tests/eval_audio_quality.py compare

# Step 4 (optional): Save improved scores as new baseline
python tests/eval_audio_quality.py save-baseline master_v2

# Run CPU-only Tier 1 tests (no GPU needed)
pytest tests/audio_quality/test_fast_ci.py -v
```

---

## The Three Tiers

| Tier | Command | Runtime | GPU? | Purpose |
|------|---------|---------|------|---------|
| **Tier 1** | `test_fast_ci.py` | ~10s | No | Smoke test scorers on synthetic audio. Catches bugs in the evaluation code itself. |
| **Tier 2** | `test_full_eval.py -m gpu` | ~10-15min | Yes | **The real tests.** Loads LuxTTS model, generates audio from speaker samples, scores with DNSMOS/WER/speaker similarity, asserts quality thresholds. |
| **Tier 3** | `test_regression.py` | <1s | No | Compare stored JSON scores against a baseline. The actual regression gate. |

---

## Workflow

### Agent Workflow

The intended workflow for iterative quality improvement:

```
1. Make code changes
2. pytest tests/audio_quality/ -m gpu          # Gates: did I break anything?
3. python tests/eval_audio_quality.py score     # Score: how does it measure?
4. python tests/eval_audio_quality.py compare   # Compare: better or worse?
5. If scores improved: save-baseline, commit
6. If scores regressed: investigate, fix, repeat
```

### Commands

| Command | What it does |
|---------|-------------|
| `python tests/eval_audio_quality.py score` | Reads manifest, scores all audio with DNSMOS + speaker similarity |
| `python tests/eval_audio_quality.py compare` | Compares latest scores against stored baseline |
| `python tests/eval_audio_quality.py save-baseline <name>` | Saves current scores as a named baseline file |
| `python tests/eval_audio_quality.py score --no-sim` | Score without speaker similarity (faster, CPU-only) |

### First Time Setup

1. Run GPU tests to generate audio and manifest:
   ```bash
   pytest tests/audio_quality/test_full_eval.py -v -m gpu
   ```

2. Score the results:
   ```bash
   python tests/eval_audio_quality.py score
   ```

3. Save as your first baseline:
   ```bash
   python tests/eval_audio_quality.py save-baseline master_baseline
   ```

### Comparing After Changes

1. Make your code changes
2. Re-run GPU tests: `pytest tests/audio_quality/test_full_eval.py -v -m gpu`
3. Score: `python tests/eval_audio_quality.py score`
4. Compare: `python tests/eval_audio_quality.py compare`
5. The compare command shows any regressions with actionable details

### Re-scoring Without Regeneration

If you want to re-score existing audio (e.g., after adding a new metric):
```bash
python tests/eval_audio_quality.py score    # Re-scores existing audio from manifest
```

---

## Comparison Modes

You can create different baselines for different comparison scenarios:

### Raw TTS Comparison

Test core model quality without post-processing:

```bash
python tests/eval_audio_quality.py save-baseline raw_tts_baseline
```

### Post-Processing Comparison

Measure whether the DSP chain helps or hurts — gate tests generate both raw and post-processed audio.

### Vocalization Comparison

Test whether vocalization tags produce appropriate sounds:

```bash
python tests/eval_audio_quality.py save-baseline vocalization_baseline
```

---

## Baseline JSON Format

Every baseline file follows this schema:

```json
{
  "version": "string - baseline name",
  "commit": "string - git commit SHA",
  "branch": "string - source branch name",
  "date": "YYYY-MM-DD",
  "generation_config": {
    "num_steps": 4,
    "guidance_scale": 3.0,
    "model_commit": "abc1234",
    "seed": 42
  },
  "speakers": ["cicero", "femalenord", "alduin"],
  "samples": {
    "<sample_name>": {
      "dnsmos_sig": 3.92,
      "dnsmos_bak": 4.10,
      "dnsmos_ovrl": 3.78,
      "silence_ratio": 0.08,
      "trailing_silence_ms": 120.5,
      "peak_amplitude": 0.95,
      "duration_s": 2.1
    }
  }
}
```

Metric fields in `samples` can include any score from the scorers:

| Metric | Source | Range | Notes |
|--------|--------|-------|-------|
| `dnsmos_sig` | DNSMOS | 1-5 | Signal quality |
| `dnsmos_bak` | DNSMOS | 1-5 | Background noise (higher = less noise) |
| `dnsmos_ovrl` | DNSMOS | 1-5 | Overall quality |
| `utmos` | UTMOS | 1-5 | Perceived naturalness |
| `speaker_similarity` | VERSA | 0-1 | Voice cloning accuracy (>0.8 = same speaker) |
| `wer` | Whisper | 0-1 | Word error rate (lower = more intelligible) |
| `silence_ratio` | Custom | 0-1 | Fraction of audio that is silent |
| `trailing_silence_ms` | Custom | 0+ | Milliseconds of silence at end |
| `has_clipping` | Custom | bool | Whether audio clips at peaks |

---

## Interpreting Regression Reports

When a regression is detected, the output looks like this:

```
REGRESSION in hello_world: 2 metric(s) dropped >5%
  dnsmos_sig: 3.92 -> 3.40 (-13.3%, threshold: -5%)
  silence_ratio: 0.08 -> 0.15 (+87.5%, threshold: -5%)
```

Wait — `silence_ratio` went UP, not down. That's because the regression check flags any metric that changes by more than the threshold in the negative direction. For metrics where lower is better (like `silence_ratio`), an increase IS a regression. For metrics where higher is better (like `dnsmos_sig`), a decrease is a regression.

**The system currently treats all metrics the same way**: it flags any metric where the current value drops by more than X% relative to the baseline. For metrics where higher is better (DNSMOS, UTMOS, speaker similarity), a drop is bad. For metrics where lower is better (silence_ratio, trailing_silence_ms, wer), an increase is bad.

**Actionable steps when you see a regression:**

1. Check which metric dropped and by how much
2. Look at the `delta_pct` — a -2% change is noise, -15% is a real problem
3. Cross-reference with the `config` — did you change post-processing settings?
4. Listen to the actual audio files if available
5. If the change was intentional (e.g., you traded signal quality for lower latency), update the baseline

---

## Multiple Baselines

You can maintain multiple baselines for different comparison scenarios:

```
tests/audio_quality/baselines/
  master_baseline.json          # Raw TTS from master (no post-processing)
  postproc_baseline.json        # With post-processing enabled
  vocalization_baseline.json    # Vocalization tag quality scores
```

Compare against any of them:

```bash
python tests/eval_audio_quality.py compare --baseline=master_baseline
python tests/eval_audio_quality.py compare --baseline=postproc_baseline
```

---

## Metrics Available

### Gate Checks (run by tests)

These run during GPU tests as hard constraints — if any fails, the test fails:

| Check | What it catches |
|-------|----------------|
| Duration range (0.3-30s) | Too short or too long output |
| Clipping detection | Audio peaks exceeding threshold |
| Trailing silence (>800ms) | Excessive silence at end |
| Silence ratio (>50%) | Audio is mostly silent |

### CLI Scoring Metrics

These run via the CLI (`python tests/eval_audio_quality.py score`):

| Scorer | What it measures |
|--------|-----------------|
| `score_dnsmos()` | Signal quality, background noise, overall quality (1-5 MOS) |
| `score_speaker_similarity()` | Voice cloning accuracy (cosine similarity, 0-1) |
| `detect_silence_artifacts()` | Trailing silence, clipping, silence ratio |

### Speaker Similarity Benchmarks

| Score | Meaning |
|-------|---------|
| > 0.8 | Same speaker (cloning is accurate) |
| 0.5 - 0.8 | Possibly same speaker (acceptable range) |
| < 0.5 | Different speaker (cloning failed) |
| < 0.3 | Completely different voice |

---

## Programmatic Usage

You can use the scorers directly in Python for custom analysis:

```python
from tests.audio_quality.scorers.versa_scorer import score_dnsmos
from tests.audio_quality.scorers.custom_scorers import detect_silence_artifacts
import librosa

# Load any WAV file
audio, sr = librosa.load("output.wav", sr=48000)

# Score it
quality = score_dnsmos(audio, sr)
artifacts = detect_silence_artifacts(audio, sr)

print(f"Signal quality: {quality['dnsmos_sig']:.2f}/5")
print(f"Trailing silence: {artifacts['trailing_silence_ms']:.0f}ms")
print(f"Clipping detected: {artifacts['has_clipping']}")
```

### Comparing Two Audio Files

```python
from tests.audio_quality.suites.regression import BaselineManager

manager = BaselineManager(regression_threshold_pct=5.0)

result = manager.compare(
    sample_name="my_test",
    baseline_scores={"dnsmos_sig": 3.92, "dnsmos_ovrl": 3.78},
    current_scores={"dnsmos_sig": 3.85, "dnsmos_ovrl": 3.80},
)

print(result["passed"])          # True (no metric dropped >5%)
print(result["regressed_metrics"])  # []
print(result["details"])         # "All metrics within 5% of baseline for my_test."
```

---

## Troubleshooting

### "DNSMOS not available"

```bash
uv pip install speechmos onnxruntime
```

### "ModuleNotFoundError: tests.audio_quality"

Run from the project root with the venv activated:

```bash
cd F:/Studies/LuxTTS
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pytest tests/audio_quality/ -v
```

### Tests pass but scores look wrong

DNSMOS scores on synthetic audio (sine waves) will be lower than on real speech. This is expected. The scores are meaningful when comparing real TTS output against real TTS output, not against synthetic test fixtures.

### Baseline file is empty

Run the full workflow to populate baselines with real scores:

```bash
pytest tests/audio_quality/test_full_eval.py -v -m gpu
python tests/eval_audio_quality.py score
python tests/eval_audio_quality.py save-baseline master_baseline
```
