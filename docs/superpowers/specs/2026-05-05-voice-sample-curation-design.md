# Voice Sample Curation Pipeline

**Date:** 2026-05-05
**Status:** Proposed

## Problem

Skyrim voice assets are being extracted as WAV files into `speakers/en1/sound/voice/{dlc}/{voice_type}/`, organized by DLC and voice type. Each voice type folder contains anywhere from 1 to ~4000 individual audio clips. The TTS system needs a single high-quality composite sample per speaker that it can use as a voice prompt.

## Goal

Build a reusable script that scans extracted voice files, selects the best clips per speaker, concatenates them into a composite sample, and validates the result through the TTS model's `encode_prompt()` pipeline.

## Input

- **Location:** `speakers/en1/sound/voice/{dawnguard.esm,dragonborn.esm,hearthfires.esm,skyrim.esm}/{voice_type}/*.wav`
- **Format:** WAV files (various sample rates, mono)
- **134 unique voice types** across 4 DLCs
- WAV extraction from `.fuz` is handled externally

## Output

- **Location:** `speakers/en1/{voice_type}.wav`
- **Format:** 44100Hz mono WAV
- **Target duration:** ~10-15 seconds per composite
- Speakers that fail TTS validation are excluded from output with a warning

## Architecture

Single script: `scripts/build_speaker_samples.py`

### Phase 1: Score

For each WAV clip in a speaker's folder:

| Metric | Method | Ideal Range |
|--------|--------|-------------|
| Energy | RMS in dB | -30dB to -3dB |
| Silence ratio | Proportion of frames below -40dB | < 30% |
| Duration | Length in seconds | 2-8s |

Composite score = weighted sum. Weights: energy 0.3, silence ratio 0.4, duration 0.3.

### Phase 2: Concatenate

1. Select top 3-5 clips by score to reach ~10-15s total
2. Sort by score (best first) so the model's `duration` window captures the strongest material
3. Crossfade 50ms between adjacent clips
4. Normalize final composite RMS to -20dB (matching existing speaker samples)
5. Resample to 44100Hz, mono
6. Write to output directory

### Phase 3: Validate

1. Load LuxTTS model (CPU mode)
2. For each composite, run `encode_prompt()`
3. Check Whisper transcription output:
   - Flag if empty or fewer than 3 words for clips >5s
   - Flag if word count / duration ratio is implausibly low (<0.5 words/sec)
4. Print summary report: pass/fail per speaker
5. Move failed composites to a `rejected/` subdirectory instead of deleting

## CLI Interface

```
python scripts/build_speaker_samples.py \
  --input speakers/en1/sound/voice \
  --output speakers/en1 \
  [--target-duration 12] \
  [--max-clips 5] \
  [--skip-validate] \
  [--dry-run]
```

- `--skip-validate`: Skip TTS model validation (faster, for iteration)
- `--dry-run`: Print selection results without writing files

## Constraints

- Dependencies: numpy, librosa, soundfile (all already in requirements.txt)
- TTS validation requires the LuxTTS model (downloads from HuggingFace on first run)
- Script must be idempotent: re-running produces the same output for unchanged inputs
- Speakers with fewer than 3 scorable clips are skipped with a warning

## Out of Scope

- Replacing or modifying existing `speakers/en/` samples
- Real-time directory watching
- GUI or interactive selection
- Manual clip curation/override
