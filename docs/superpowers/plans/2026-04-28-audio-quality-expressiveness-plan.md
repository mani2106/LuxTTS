# Audio Quality & Expressiveness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three audio quality bugs (gradual degradation, trailing artifacts, robotic timbre) and add an expressiveness layer (prosodic modulation, room presence, spectral enrichment).

**Architecture:** Foundation fixes target the TTS generation layer (luxvoice.py, modeling_utils.py) to stop producing degraded audio. Expressiveness layer adds three numpy/scipy DSP stages to the existing post-processing chain in post_processor.py, wired between pitch shift and loudness normalization.

**Tech Stack:** Python, PyTorch, numpy, scipy, librosa

**Spec:** `docs/superpowers/specs/2026-04-28-audio-quality-expressiveness-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `zipvoice/luxvoice.py` | Deep-copy cached encodings before model inference | Modify |
| `zipvoice/modeling_utils.py` | Use pred_features_lens, add chunked generation | Modify |
| `utilities/app_constants.py` | Reduce ref duration and compressor ratio | Modify |
| `utilities/post_processor.py` | Add 3 expressiveness DSP stages, update chain | Modify |
| `utilities/audio_generation_pipeline.py` | Wire new stages into pipeline params | Modify |
| `tests/test_post_processor.py` | Tests for new DSP stages | Modify |

---

### Task 1: Deep-Copy Cached Encodings

**Files:**
- Modify: `zipvoice/luxvoice.py:47-62`

- [ ] **Step 1: Add import and deep-copy logic in `generate_speech()`**

In `zipvoice/luxvoice.py`, add `import copy` at the top and replace line 50 to deep-copy all encode_dict values:

```python
import copy
```

Then replace the body of `generate_speech()` — change this:
```python
    def generate_speech(self, text, encode_dict, num_steps=4, guidance_scale=3.0, t_shift=0.5, speed=1.0, return_smooth=False):
        """encodes text and generates speech using flow matching model according to steps, guidance scale, and t_shift(like temp)"""

        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = encode_dict.values()
```

To this:
```python
    def generate_speech(self, text, encode_dict, num_steps=4, guidance_scale=3.0, t_shift=0.5, speed=1.0, return_smooth=False):
        """encodes text and generates speech using flow matching model according to steps, guidance scale, and t_shift(like temp)"""

        prompt_tokens = copy.deepcopy(encode_dict["prompt_tokens"])
        prompt_features_lens = encode_dict["prompt_features_lens"].clone()
        prompt_features = encode_dict["prompt_features"].clone()
        prompt_rms = encode_dict["prompt_rms"].clone()
```

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/ -v --timeout=30 -x`
Expected: All existing tests PASS.

- [ ] **Step 3: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add zipvoice/luxvoice.py
git commit -m "fix: deep-copy cached speaker encodings to prevent degradation across requests"
```

---

### Task 2: Use pred_features_lens for Vocoder Trimming

**Files:**
- Modify: `zipvoice/modeling_utils.py:64-99`

- [ ] **Step 1: Update `generate()` to use pred_features_lens**

In `zipvoice/modeling_utils.py`, replace the `generate()` function (lines 64-99) with:

```python
def generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1):
    tokens = tokenizer.texts_to_token_ids([text])
    device = next(model.parameters()).device

    speed = speed * 1.3

    with torch.inference_mode():
        (pred_features, _, _, pred_lens) = model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration='predict',
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

    # Convert to waveform
    pred_features = pred_features.permute(0, 2, 1) / 0.1

    # Trim to actual predicted length + small decay tail for natural ending
    actual_len = pred_lens[0].item()
    actual_len = min(actual_len, pred_features.size(2))
    last_frame = pred_features[:, :, actual_len - 1:actual_len]
    decay = torch.linspace(1.0, 0.0, 5).to(pred_features.device).view(1, 1, -1)
    tail = last_frame * decay
    pred_features = torch.cat([pred_features[:, :, :actual_len], tail], dim=2)

    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Volume matching
    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)

    return wav
```

Key change: `(_, _, _, _)` becomes `(pred_features, _, _, pred_lens)` and we trim to `pred_lens[0]` instead of padding with repeated frames.

- [ ] **Step 2: Run existing tests**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/ -v --timeout=30 -x`
Expected: All existing tests PASS.

- [ ] **Step 3: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add zipvoice/modeling_utils.py
git commit -m "fix: use pred_features_lens for vocoder trimming instead of repeated-frame padding"
```

---

### Task 3: Add Punctuation-Based Chunked Generation

**Files:**
- Modify: `zipvoice/modeling_utils.py`

- [ ] **Step 1: Add import for chunking utilities**

At the top of `zipvoice/modeling_utils.py`, add to the existing imports:

```python
from zipvoice.utils.infer import chunk_tokens_punctuation, cross_fade_concat
```

- [ ] **Step 2: Add `_generate_chunked()` helper function**

Add this function after the `generate()` function in `modeling_utils.py`:

```python
def _generate_chunked(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1, chunk_char_threshold=120):
    """Generate speech for longer texts by chunking at punctuation boundaries."""
    device = next(model.parameters()).device
    speed_internal = speed * 1.3

    # Tokenize to string tokens for chunking
    tokens_str = tokenizer.texts_to_tokens([text])[0]

    # Estimate max_tokens per chunk targeting ~25s total (prompt + generated)
    prompt_duration_s = prompt_features.size(1) * 0.01  # rough estimate
    token_duration_s = prompt_duration_s / max(len(tokens_str), 1) / speed
    max_tokens = int(max((25 - prompt_duration_s) / max(token_duration_s, 0.01), 20))
    max_tokens = min(max_tokens, 150)  # cap to avoid very long chunks

    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

    if len(chunked_tokens_str) <= 1:
        # Only one chunk — use normal generation
        return generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step, guidance_scale, speed, t_shift, target_rms)

    # Generate each chunk
    chunk_wavs = []
    with torch.inference_mode():
        for chunk_str_tokens in chunked_tokens_str:
            chunk_token_ids = tokenizer.tokens_to_token_ids([chunk_str_tokens])

            (pred_features, _, _, pred_lens) = model.sample(
                tokens=chunk_token_ids,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed=speed_internal,
                t_shift=t_shift,
                duration='predict',
                num_step=num_step,
                guidance_scale=guidance_scale,
            )

            pred_features = pred_features.permute(0, 2, 1) / 0.1

            actual_len = pred_lens[0].item()
            actual_len = min(actual_len, pred_features.size(2))
            last_frame = pred_features[:, :, actual_len - 1:actual_len]
            decay = torch.linspace(1.0, 0.0, 5).to(pred_features.device).view(1, 1, -1)
            tail = last_frame * decay
            pred_features = torch.cat([pred_features[:, :, :actual_len], tail], dim=2)

            wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

            if prompt_rms < target_rms:
                wav = wav * (prompt_rms / target_rms)

            chunk_wavs.append(wav)

    # Crossfade merge chunks
    final_wav = cross_fade_concat(chunk_wavs, fade_duration=0.1, sample_rate=48000)
    return final_wav
```

Note: This function reuses the `pred_features_lens` trimming logic from Task 2 for each chunk.

- [ ] **Step 3: Update `generate()` to dispatch to chunking for long text**

In `zipvoice/modeling_utils.py`, modify the `generate()` function. Add this dispatch logic at the top of `generate()`, before `tokens = tokenizer.texts_to_token_ids([text])`:

```python
def generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1):
    CHUNK_CHAR_THRESHOLD = 120

    if len(text) > CHUNK_CHAR_THRESHOLD:
        return _generate_chunked(
            prompt_tokens, prompt_features_lens, prompt_features, prompt_rms,
            text, model, vocoder, tokenizer,
            num_step, guidance_scale, speed, t_shift, target_rms,
            chunk_char_threshold=CHUNK_CHAR_THRESHOLD,
        )

    tokens = tokenizer.texts_to_token_ids([text])
    # ... rest of existing generate() unchanged
```

- [ ] **Step 4: Run existing tests**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/ -v --timeout=30 -x`
Expected: All existing tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add zipvoice/modeling_utils.py
git commit -m "feat: add punctuation-based chunking for texts > 120 chars"
```

---

### Task 4: Update Default Constants

**Files:**
- Modify: `utilities/app_constants.py`

- [ ] **Step 1: Update constants**

In `utilities/app_constants.py`, change these three values:

```python
DEFAULT_REF_DURATION = 3  # Lower speeds up inference; ZipVoice recommends 1-3s
```

```python
DEFAULT_COMPRESSOR_RATIO = 2.0
```

```python
DEFAULT_COMPRESSOR_KNEE_DB = 8.0
```

The comments for other lines should remain as-is. Only the values change.

- [ ] **Step 2: Run existing tests**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/ -v --timeout=30 -x`
Expected: All existing tests PASS. Note: `test_compressor_reduces_dynamic_range` passes explicit ratio=4.0, so it won't be affected by the default change.

- [ ] **Step 3: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add utilities/app_constants.py
git commit -m "tune: reduce ref duration to 3s, compressor ratio to 2:1, widen knee to 8dB"
```

---

### Task 5: Add Prosodic Micro-Modulation

**Files:**
- Modify: `utilities/post_processor.py`
- Modify: `tests/test_post_processor.py`

- [ ] **Step 1: Write failing test for prosodic_modulation**

Add to `tests/test_post_processor.py`:

```python
def test_prosodic_modulation_changes_audio(sample_48k_audio):
    """Prosodic modulation should subtly vary amplitude."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.prosodic_modulation(audio, sr, text="This is exciting!")

    assert processed is not None
    assert len(processed) == len(audio)
    assert processed.dtype == np.float32
    # Should differ from input (modulation was applied)
    assert not np.allclose(processed, audio, atol=1e-6)
    assert 'emotion' in diagnostics


def test_prosodic_modulation_calm_text_shallow(sample_48k_audio):
    """Calm text should have shallower modulation than excited text."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    calm_processed, _ = processor.prosodic_modulation(audio, sr, text="Hello world")
    excited_processed, _ = processor.prosodic_modulation(audio, sr, text="This is exciting!")

    # Measure how much each deviates from input
    calm_diff = np.sqrt(np.mean((calm_processed - audio) ** 2))
    excited_diff = np.sqrt(np.mean((excited_processed - audio) ** 2))

    # Excited should have more modulation than calm
    assert excited_diff > calm_diff


def test_prosodic_modulation_silence(sample_48k_audio):
    """Prosodic modulation on silence should remain silence."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)
    processor = AudioPostProcessor()

    processed, _ = processor.prosodic_modulation(silence, sr, text="Hello!")

    # Silence * (1 + modulation) = silence
    np.testing.assert_allclose(processed, silence, atol=1e-7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/test_post_processor.py::test_prosodic_modulation_changes_audio -v`
Expected: FAIL — `AttributeError: 'AudioPostProcessor' object has no attribute 'prosodic_modulation'`

- [ ] **Step 3: Implement `prosodic_modulation()`**

Add this method to the `AudioPostProcessor` class in `utilities/post_processor.py`, after the `pitch_shift()` method:

```python
    def prosodic_modulation(
        self,
        audio: np.ndarray,
        sr: int,
        text: str = "",
    ) -> tuple[np.ndarray, dict]:
        """
        Apply subtle amplitude modulation to break flat/robotic quality.

        Modulation depth scales with detected emotional context:
        calm=0.05, question=0.08, excited=0.12, intense=0.15.

        Args:
            audio: Input audio (float32)
            sr: Sample rate
            text: Dialogue text for emotion detection

        Returns:
            (processed_audio, diagnostics_dict)
        """
        # Detect emotional context
        detector = PitchDetector()
        pitch = detector.detect_pitch(text)

        if pitch >= 2.0:
            emotion = "intense"
            depth = 0.15
        elif pitch >= 1.0:
            emotion = "excited"
            depth = 0.12
        elif pitch >= 0.5:
            emotion = "question"
            depth = 0.08
        else:
            emotion = "calm"
            depth = 0.05

        # Amplitude tremolo at natural vocal tremor rate (3-4 Hz)
        mod_freq = 3.5
        t = np.arange(len(audio), dtype=np.float64) / sr
        mod_signal = np.sin(2 * np.pi * mod_freq * t) * depth
        mod_signal = mod_signal.astype(np.float32)

        processed = audio * (1.0 + mod_signal)

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['emotion'] = emotion
            diagnostics['modulation_depth'] = depth
            diagnostics['modulation_freq_hz'] = mod_freq

        return processed.astype(np.float32), diagnostics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/test_post_processor.py -k "prosodic_modulation" -v`
Expected: All 3 prosodic_modulation tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: add prosodic micro-modulation DSP stage"
```

---

### Task 6: Add Room Presence

**Files:**
- Modify: `utilities/post_processor.py`
- Modify: `tests/test_post_processor.py`

- [ ] **Step 1: Write failing tests for room_presence**

Add to `tests/test_post_processor.py`:

```python
def test_room_presence_adds_reverb(sample_48k_audio):
    """Room presence should add subtle reverb tail."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.room_presence(audio, sr)

    assert processed is not None
    assert len(processed) == len(audio)
    assert processed.dtype == np.float32
    # Should differ from input (reverb was added)
    assert not np.allclose(processed, audio, atol=1e-6)
    assert 'wet_level_db' in diagnostics
    assert diagnostics['wet_level_db'] == -12.0


def test_room_presence_silence(sample_48k_audio):
    """Room presence on silence should remain near-silence."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)
    processor = AudioPostProcessor()

    processed, _ = processor.room_presence(silence, sr)

    # Silence convolved with IR = silence
    assert np.max(np.abs(processed)) < 1e-6


def test_room_presence_louder_wet_signal():
    """Higher wet level should produce more noticeable reverb."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5))
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    processor = AudioPostProcessor()

    quiet, _ = processor.room_presence(audio, sr, wet_db=-20)
    loud, _ = processor.room_presence(audio, sr, wet_db=-6)

    quiet_diff = np.sqrt(np.mean((quiet - audio) ** 2))
    loud_diff = np.sqrt(np.mean((loud - audio) ** 2))

    assert loud_diff > quiet_diff
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/test_post_processor.py::test_room_presence_adds_reverb -v`
Expected: FAIL — `AttributeError: 'AudioPostProcessor' object has no attribute 'room_presence'`

- [ ] **Step 3: Implement `room_presence()`**

Add this method to `AudioPostProcessor` class in `utilities/post_processor.py`, after `prosodic_modulation()`:

```python
    def room_presence(
        self,
        audio: np.ndarray,
        sr: int,
        room_size: str = "small",
        wet_db: float = -12.0,
    ) -> tuple[np.ndarray, dict]:
        """
        Add subtle room presence via synthetic early reflections.

        Generates a bandpass-filtered exponentially-decaying impulse response
        and convolves it with the audio at a low wet level.

        Args:
            audio: Input audio (float32)
            sr: Sample rate
            room_size: "small" (80ms RT60) or "medium" (150ms RT60)
            wet_db: Wet signal level in dB (default -12 = subtle)

        Returns:
            (processed_audio, diagnostics_dict)
        """
        rt60 = 0.08 if room_size == "small" else 0.15
        ir_length = int(rt60 * sr)

        # Generate synthetic impulse response
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        ir = rng.randn(ir_length).astype(np.float32)

        # Exponential decay
        ir *= np.exp(-np.linspace(0, 6, ir_length)).astype(np.float32)

        # Bandpass to natural speech range (200Hz-8kHz)
        b, a = signal.butter(2, [200 / (sr / 2), 8000 / (sr / 2)], btype='band')
        ir = signal.filtfilt(b, a, ir).astype(np.float32)
        ir /= np.max(np.abs(ir)) + 1e-10

        # Convolve and mix
        reverb = np.convolve(audio, ir, mode='full')[:len(audio)].astype(np.float32)
        wet_gain = 10 ** (wet_db / 20)
        processed = audio + reverb * wet_gain

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['wet_level_db'] = wet_db
            diagnostics['rt60_ms'] = rt60 * 1000

        return processed.astype(np.float32), diagnostics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/test_post_processor.py -k "room_presence" -v`
Expected: All 3 room_presence tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: add room presence reverb DSP stage"
```

---

### Task 7: Add Spectral Enrichment

**Files:**
- Modify: `utilities/post_processor.py`
- Modify: `tests/test_post_processor.py`

- [ ] **Step 1: Write failing tests for spectral_enrich**

Add to `tests/test_post_processor.py`:

```python
def test_spectral_enrich_adds_harmonics(sample_48k_audio):
    """Spectral enrichment should modify audio by adding upper harmonics."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.spectral_enrich(audio, sr)

    assert processed is not None
    assert len(processed) == len(audio)
    assert processed.dtype == np.float32
    assert not np.allclose(processed, audio, atol=1e-6)
    assert 'intensity' in diagnostics


def test_spectral_enrich_zero_intensity_bypass(sample_48k_audio):
    """Zero intensity should bypass spectral enrichment."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor()

    processed, _ = processor.spectral_enrich(audio, sr, intensity=0.0)

    np.testing.assert_allclose(processed, audio, atol=1e-6)


def test_spectral_enrich_silence(sample_48k_audio):
    """Spectral enrichment on silence should remain silence."""
    sr = 48000
    silence = np.zeros(48000, dtype=np.float32)
    processor = AudioPostProcessor()

    processed, _ = processor.spectral_enrich(silence, sr)

    np.testing.assert_allclose(processed, silence, atol=1e-7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/test_post_processor.py::test_spectral_enrich_adds_harmonics -v`
Expected: FAIL — `AttributeError: 'AudioPostProcessor' object has no attribute 'spectral_enrich'`

- [ ] **Step 3: Implement `spectral_enrich()`**

Add this method to `AudioPostProcessor` class in `utilities/post_processor.py`, after `room_presence()`:

```python
    def spectral_enrich(
        self,
        audio: np.ndarray,
        sr: int,
        intensity: float = 0.3,
    ) -> tuple[np.ndarray, dict]:
        """
        Add subtle upper harmonics via nonlinear waveshaping.

        High-pass extracts content above 2kHz, applies soft saturation
        to generate harmonics, then mixes back at low level.

        Args:
            audio: Input audio (float32)
            sr: Sample rate
            intensity: Mix amount (0.0 = bypass, 1.0 = full)

        Returns:
            (processed_audio, diagnostics_dict)
        """
        if intensity <= 0.0:
            return audio.copy(), {}

        # High-pass extract above 2kHz
        b, a = signal.butter(2, 2000 / (sr / 2), btype='high')
        hf = signal.filtfilt(b, a, audio)

        # Waveshape to generate harmonics (soft saturation)
        hf_enriched = np.tanh(hf * 2.0) / 2.0

        # Mix back at controlled level
        mix = intensity * 0.3  # max ~-10dB wet
        processed = audio * (1.0 - mix) + hf_enriched * mix

        diagnostics = {}
        if self.return_diagnostics:
            diagnostics['intensity'] = intensity
            diagnostics['mix_level'] = mix

        return processed.astype(np.float32), diagnostics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/test_post_processor.py -k "spectral_enrich" -v`
Expected: All 3 spectral_enrich tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: add spectral enrichment harmonic exciter DSP stage"
```

---

### Task 8: Wire Expressiveness Stages Into Processing Chain

**Files:**
- Modify: `utilities/post_processor.py:824-938` (the `process()` method)
- Modify: `tests/test_post_processor.py`

- [ ] **Step 1: Update `process()` to include the three new stages**

In `utilities/post_processor.py`, locate the `process()` method. Insert the three new stages between the pitch shift stage (Stage 4) and the normalize stage (Stage 5). The new stages go after the pitch shift block and before the normalize block:

After this existing block (Stage 4 — pitch shift):
```python
        audio, pitch_diagnostics = self.pitch_shift(audio, sr, n_steps=detected_pitch)
        if pitch_diagnostics:
            all_diagnostics['pitch_shift'] = pitch_diagnostics
        # Add detected pitch value to diagnostics even if empty
        if 'pitch_shift' not in all_diagnostics:
            all_diagnostics['pitch_shift'] = {'detected_semitones': detected_pitch}
        else:
            all_diagnostics['pitch_shift']['detected_semitones'] = detected_pitch
```

Insert these three new stages:
```python
        # Stage 5: Prosodic micro-modulation
        audio, prosodic_diagnostics = self.prosodic_modulation(audio, sr, text=text or "")
        if prosodic_diagnostics:
            all_diagnostics['prosodic_modulation'] = prosodic_diagnostics

        # Stage 6: Room presence
        audio, room_diagnostics = self.room_presence(audio, sr)
        if room_diagnostics:
            all_diagnostics['room_presence'] = room_diagnostics

        # Stage 7: Spectral enrichment
        audio, enrich_diagnostics = self.spectral_enrich(audio, sr)
        if enrich_diagnostics:
            all_diagnostics['spectral_enrich'] = enrich_diagnostics
```

The existing Stage 5 (normalize) becomes Stage 8 — its code remains unchanged.

Update the docstring processing order comment in `process()` to reflect:
```python
        Processing order:
        1. De-esser (reduce sibilance)
        2. EQ (tame harshness, add warmth)
        3. Compressor (soft-knee, adaptive threshold, look-ahead, makeup gain)
        4. Pitch shift (adjust pitch)
        5. Prosodic modulation (micro amplitude variation)
        6. Room presence (subtle early reflections)
        7. Spectral enrichment (harmonic exciter)
        8. Normalize loudness (EBU R128)
```

- [ ] **Step 2: Write test for updated full chain**

Add to `tests/test_post_processor.py`:

```python
def test_process_full_chain_includes_expressiveness(sample_48k_audio):
    """Full chain should include prosodic_modulation, room_presence, spectral_enrich."""
    audio, sr = sample_48k_audio
    processor = AudioPostProcessor(return_diagnostics=True)

    processed, diagnostics = processor.process(
        audio, sr,
        text="Hello world",
        eq_intensity=1.0,
        de_ess_intensity=0.5,
        target_loudness=-16.0,
    )

    assert processed is not None
    assert len(processed) > 0
    assert not np.any(np.isnan(processed))
    assert 'prosodic_modulation' in diagnostics
    assert 'room_presence' in diagnostics
    assert 'spectral_enrich' in diagnostics
```

- [ ] **Step 3: Run all tests**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/test_post_processor.py -v`
Expected: ALL tests PASS, including the new one.

- [ ] **Step 4: Run full test suite**

Run: `cd "F:/Studies/LuxTTS" && python -m pytest tests/ -v --timeout=30`
Expected: ALL tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "F:/Studies/LuxTTS"
git add utilities/post_processor.py tests/test_post_processor.py
git commit -m "feat: wire expressiveness stages into post-processing chain"
```

---

## Self-Review

### Spec Coverage
| Spec Section | Task |
|---|---|
| Fix 1: Deep-copy encodings | Task 1 |
| Fix 2: pred_features_lens trimming | Task 2 |
| Fix 3: Punctuation chunking | Task 3 |
| Fix 3b: Reduce ref duration | Task 4 |
| Fix 4: Compressor tuning | Task 4 |
| Enhancement 5a: Prosodic modulation | Task 5 |
| Enhancement 5b: Room presence | Task 6 |
| Enhancement 5c: Spectral enrichment | Task 7 |
| Wire stages into chain | Task 8 |

All spec sections covered. No gaps.

### Placeholder Scan
No TBD, TODO, or placeholder patterns found. All steps contain complete code.

### Type Consistency
- `prosodic_modulation()` returns `tuple[np.ndarray, dict]` — consistent with all other methods
- `room_presence()` returns `tuple[np.ndarray, dict]` — consistent
- `spectral_enrich()` returns `tuple[np.ndarray, dict]` — consistent
- All methods use `self.return_diagnostics` for conditional diagnostics — consistent with existing pattern
- `generate()` and `_generate_chunked()` return `torch.Tensor` (wav) — consistent with existing return type
