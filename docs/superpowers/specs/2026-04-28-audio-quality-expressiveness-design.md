# Audio Quality & Expressiveness Improvement Design

Date: 2026-04-28
Branch: expressive-vocalizations

## Problem Statement

LuxTTS in-game dialogue has three quality issues:
1. **Gradual degradation** across batched requests — tone/pitch drifts as more lines are generated
2. **Trailing syllable distortion** — last word/syllable sounds garbled ("old TV noises")
3. **Robotic timbre** on medium-length text (100-300 chars) — flat, synthetic quality

Additionally, the post-processing chain is entirely subtractive (de-ess, compress, EQ cut) and doesn't add expressiveness.

## Root Cause Analysis

### Gradual Degradation
`cache_utils.py` stores tensor objects by reference. `luxvoice.py:50` extracts `encode_dict.values()` as references to cached tensors. The model's forward pass mutates these in-place. Each subsequent call uses increasingly corrupted prompt data.

### Trailing Artifacts
`modeling_utils.py:87-91` pads by repeating the last mel frame 15 times. If the last frame is degraded, the padding extends the degradation. The vocoder synthesizes this as distortion.

More fundamentally: `model.sample()` returns `pred_features_lens` (actual predicted lengths) but LuxTTS ignores it and pads instead. The official ZipVoice code uses `pred_features_lens` to trim vocoder input cleanly.

### Robotic Timbre
Three causes:
- No text chunking — entire passage goes through one flow-matching pass, quality degrades on longer sequences
- `DEFAULT_REF_DURATION = 10` seconds — ZipVoice recommends <3s, warns >10s degrades quality
- Compressor at 4:1 ratio flattens natural dynamics

### Lack of Expressiveness
The entire post-processing chain is subtractive. Nothing adds prosodic variation, natural room presence, or spectral warmth.

## Design

### Fix 1: Deep-Copy Cached Encodings

**File:** `zipvoice/luxvoice.py`

Before passing cached encode_dict to the model, deep-copy all tensors so the cache stays pristine.

```python
import copy

prompt_tokens = copy.deepcopy(encode_dict["prompt_tokens"])
prompt_features_lens = encode_dict["prompt_features_lens"].clone()
prompt_features = encode_dict["prompt_features"].clone()
prompt_rms = encode_dict["prompt_rms"].clone()
```

- `.clone()` for GPU tensors (CUDA-aware copy)
- `deepcopy` for `prompt_tokens` (list of lists)

### Fix 2: Use pred_features_lens for Vocoder Trimming

**File:** `zipvoice/modeling_utils.py`

Replace the repeated-frame padding with proper length-based trimming + small decay tail.

```python
# Capture all 4 return values (currently ignoring indices 1 and 3)
(pred_features, _, _, pred_features_lens) = model.sample(...)
pred_features = pred_features.permute(0, 2, 1) / 0.1

# Trim to actual predicted length + small decay tail
actual_len = pred_features_lens[0].item()
last_frame = pred_features[:, :, actual_len-1:actual_len]
decay = torch.linspace(1.0, 0.0, 5).to(pred_features.device).view(1, 1, -1)
tail = last_frame * decay
pred_features = torch.cat([pred_features[:, :, :actual_len], tail], dim=2)
wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
```

### Fix 3: Punctuation-Based Chunking

**File:** `zipvoice/modeling_utils.py`

For texts > 120 characters, use ZipVoice's own `chunk_tokens_punctuation()` from `zipvoice/utils/infer.py`.

Steps:
1. Tokenize text to string tokens
2. Call `chunk_tokens_punctuation(tokens, max_tokens)` with calibrated max_tokens
3. Generate each chunk via `model.sample()`
4. Vocode each chunk with proper `pred_features_lens` trimming
5. Merge chunks with `cross_fade_concat(fade_duration=0.1)`

Short texts (< 120 chars) go through the existing single-pass path unchanged.

### Fix 3b: Reduce Prompt Duration

**File:** `utilities/app_constants.py`

```python
DEFAULT_REF_DURATION = 3  # was 10; ZipVoice recommends 1-3s
```

### Fix 4: Reduce Compressor Aggressiveness

**File:** `utilities/app_constants.py`

```python
DEFAULT_COMPRESSOR_RATIO = 2.0     # was 4.0
DEFAULT_COMPRESSOR_KNEE_DB = 8.0   # was 4.0
```

Gentle leveling instead of dynamic squashing.

### Enhancement 5a: Prosodic Micro-Modulation

**File:** `utilities/post_processor.py`

New method `prosodic_modulation(audio, sr, text)` added to `AudioPostProcessor`.

Amplitude tremolo at 2-3 Hz (natural vocal tremor rate) with depth scaled by emotional context from PitchDetector:
- Calm/neutral: ±0.15 semitone equivalent
- Questioning: ±0.2
- Excited (!): ±0.3
- Intense (ALL CAPS): ±0.4

Implementation: energy modulation via sine wave scaled by emotional depth. No per-window pitch shifting needed — amplitude variation alone is perceptible as expressiveness.

### Enhancement 5b: Room Presence

**File:** `utilities/post_processor.py`

New method `room_presence(audio, sr, room_size="small", wet_db=-12)`.

Generates a synthetic impulse response:
- RT60: 80ms (small room)
- Bandpass filtered 200Hz-8kHz
- Exponential decay envelope
- Mixed at -12dB wet level

Implementation: `np.convolve` with synthetic IR. No external reverb plugin needed.

### Enhancement 5c: Spectral Enrichment

**File:** `utilities/post_processor.py`

New method `spectral_enrich(audio, sr, intensity=0.3)`.

Harmonic exciter via nonlinear waveshaping:
1. High-pass extract above 2kHz
2. Apply `tanh` soft saturation to generate upper harmonics
3. Mix back at -15dB equivalent level

### Updated Processing Chain

```
De-esser → EQ → Compressor (2:1) → Pitch Shift →
Prosodic Modulation → Room Presence → Spectral Enrich →
Loudness Normalize
```

All new stages are numpy/scipy, no GPU, ~10-20ms total per generation.

## Files Modified

| File | Change |
|------|--------|
| `zipvoice/luxvoice.py` | Deep-copy cached encodings |
| `zipvoice/modeling_utils.py` | Use pred_features_lens, add chunking |
| `utilities/app_constants.py` | Reduce ref duration, compressor ratio |
| `utilities/post_processor.py` | Add 3 expressiveness stages |
| `utilities/audio_generation_pipeline.py` | Wire new post-processing stages |

## Out of Scope

- Model fine-tuning or retraining
- Adding new TTS model variants
- Changing the Vocos vocoder
- Neural voice enhancement models
- Streaming/incremental generation

## Testing

- A/B comparison with `save_raw=True` for before/after samples
- Batch generation test: generate 10 sequential lines with same speaker, verify no degradation
- Long text test: 200+ char passage, verify no trailing artifacts
- Unit tests for new DSP stages (prosodic modulation, room presence, spectral enrich)
- Existing test suite must continue passing
