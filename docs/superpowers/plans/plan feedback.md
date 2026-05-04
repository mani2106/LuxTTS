Below is a **practical, musician / audio‑engineer focused review** of the plans you shared, with **only the absolutely necessary changes** prioritized first, then a short list of useful but optional refinements. I read both the *Signal‑Adaptive Post‑Processing* and *Audio Quality Testing* plans and used them to shape recommendations.

> From your plan: **“Replace the uniform 8‑stage post‑processing pipeline with a signal‑adaptive pipeline that only applies processing when the audio actually needs it, reducing OVRL degradation from 30% to under 5%.”**
> From your testing plan: **“Build a tiered audio quality evaluation framework that scores TTS output across naturalness, signal quality, speaker similarity, and intelligibility, with regression baselines and agent‑parseable reports.”**

---

### 1) **Top‑priority (must do) — fixes that will most directly improve perceived audio quality**
These are minimal, low‑risk changes that address the biggest causes of audible degradation.

#### 1.1 Always measure and enforce **true‑peak** and LUFS before/after processing
- **Why:** Perceptual clipping and loudness jumps are the most obvious quality regressions in game dialogue. LUFS alone misses inter‑sample/true‑peak overs.
- **What to do:**
  - Measure **true‑peak** (dBTP) and **integrated LUFS** on raw output and after post‑processing. Use `pyloudnorm` for LUFS and an oversampling true‑peak routine (or `libebur128`/`ffmpeg` true‑peak).
  - **Fail the pipeline** (or apply corrective limiting) if true‑peak > -1 dBTP after processing. Target integrated LUFS = **-18 LUFS** (already in plan) but ensure true‑peak headroom.
- **Implementation note:** Add `true_peak` and `integrated_lufs` fields to the diagnostics manifest for every sample.

#### 1.2 Replace any hard brick‑wall limiting with a **soft, look‑ahead limiter** (or at least a soft‑knee brick‑wall)
- **Why:** Brick‑wall clipping causes distortion and timbral change; a soft look‑ahead limiter preserves transients and reduces pumping.
- **What to do:**
  - If you can’t add a look‑ahead limiter, implement a **soft‑knee limiter** with a short attack (0–2 ms) and release tuned to speech (50–150 ms).
  - Keep the limiter threshold conservative (e.g., -1 to -0.5 dBTP) and allow small gain‑reduction (max 3–6 dB typical for dialogue).
- **Diagnostics:** Report **gain reduction** statistics (max, median) so regressions are visible.

#### 1.3 Make the **de‑esser intensity proportional** to the measured sibilance ratio
- **Why:** Binary on/off de‑essing either does nothing or over‑smears consonants. Scaling keeps sibilants natural.
- **What to do:** Map `sibilance_ratio` → de‑ess gain reduction (e.g., 0.0–0.6 dB per 0.05 ratio above threshold), with a maximum cap. Use a narrow band (4–8 kHz) and a gentle Q.

#### 1.4 Always run a **gentle HPF** (80 Hz) and check for low‑end energy
- **Why:** Low‑frequency rumble muddies dialogue and triggers unnecessary compression/limiting. HPF at 80 Hz is a good default for Skyrim dialogue.
- **What to do:** 2nd–4th order Butterworth HPF at 80 Hz (as in plan). Also **report low‑frequency energy** (e.g., energy < 200 Hz) so you can tune mud‑cut thresholds per voice archetype.

#### 1.5 Per‑speaker baselines and thresholds
- **Why:** Skyrim voice archetypes (Nord, Khajiit, Argonian, etc.) have different spectral centers and dynamics. A single threshold will either under‑process or over‑process some voices.
- **What to do:** Store **per‑speaker** baseline stats (mean LUFS, spectral centroid, typical peak) and use them to adapt EQ/limiter thresholds. This is essential for reliable regression detection.

---

### 2) **Critical testing & measurement changes (must do to trust results)**
These ensure your automated metrics reflect audible quality.

#### 2.1 Add **true blind A/B listening** to the regression workflow
- **Why:** Objective metrics miss subtle prosody and naturalness issues. A small human panel (3–5 listeners) on a handful of golden lines catches what metrics miss.
- **What to do:** Maintain a **golden set** of 20 iconic Skyrim lines (per archetype) and run a quick blind A/B test whenever a baseline changes significantly.

#### 2.2 Expand diagnostics to include **crest factor, spectral tilt, and formant shift indicators**
- **Why:** Compression/processing often changes perceived “body” and intelligibility. Crest factor and spectral tilt are quick proxies for over‑compression and unnatural timbre.
- **What to do:** Add these to the manifest and baseline comparisons.

#### 2.3 Make DNSMOS/UTMOS optional but keep **WER + short‑term intelligibility checks** mandatory
- **Why:** DNSMOS is useful but can be flaky; WER (or a lightweight ASR) directly measures intelligibility which is critical for gameplay.
- **What to do:** Ensure WER runs on all lines with reference text; if Whisper is too heavy, use a smaller ASR for CI.

---

### 3) **DSP / pipeline behavior changes (high impact, low complexity)**
These keep the chain adaptive but conservative.

#### 3.1 **Scale EQ gains** and prefer **shelves** over aggressive peaking
- **Why:** Large peaking boosts (±3–6 dB) change character; gentle shelves (±1–2 dB) improve clarity without sounding processed.
- **What to do:** For mud cut, use a low‑shelf or gentle peaking at 200–400 Hz with -1.5 to -3 dB max. For presence, use a high‑shelf around 3–5 kHz with +0.5 to +1.5 dB max.

#### 3.2 Make compressor opt‑in and prefer **multiband or program‑dependent gentle compression** if used
- **Why:** Single‑band compressors can squash sibilants or low end. If you must compress, keep ratio low (1.5–2.5:1) and use slow attack/medium release for dialogue.
- **What to do:** Keep the default pipeline using the adaptive limiter; only enable full compressor for special cases and report its gain‑reduction.

#### 3.3 De‑esser band and Q: **narrow Q, dynamic threshold**
- **Why:** A narrow Q avoids dulling the whole voice. Dynamic threshold based on sibilance energy avoids over‑processing.

---

### 4) **Vocalization / content‑specific recommendations (important for Skyrim)**
Skyrim relies on expressive vocalizations; treat them differently.

#### 4.1 Treat vocalizations as a separate class
- **Why:** Sighs, whispers, moans, groans have different spectral and dynamic profiles.
- **What to do:**
  - Detect vocalization tags and use a **separate preset**: lower target LUFS (so they don’t overpower), different HPF/LPF, and different de‑ess settings.
  - For whispers: apply band‑pass (600–4k) and add breath noise carefully (as in your recipes), but **do not** normalize them to the same LUFS as speech — keep them quieter relative to dialogue.

#### 4.2 Add a “vocalization distinctness” check
- **Why:** Vocalizations must be audible but not clipped or overly loud.
- **What to do:** Use the vocalization scorer already planned (spectral centroid vs speech, energy ratio) and fail if the vocalization is too close to speech in centroid/energy.

---

### 5) **CI / workflow and data hygiene (must do for reliable comparisons)**
These are essential so your automated tests mean something.

#### 5.1 Capture **true‑peak, LUFS, spectral centroid, sibilance_ratio, crest factor** in the manifest for every generated file
- **Why:** You need consistent, comparable numbers to detect regressions.
- **What to do:** Add these fields to the manifest JSON and to baseline files.

#### 5.2 Use **per‑speaker golden references** and store speaker role metadata
- **Why:** Regression detection must be traceable to voice archetype. Store `speaker_role` (e.g., “Nord male commoner”) in manifest entries.

#### 5.3 Keep the fast CI path but run **full scoring on a representative subset** only
- **Why:** Full VERSA/UTMOS + speaker sim on 12× lines is expensive. Run full scoring on a curated subset (golden lines + 1 sample per archetype) and gate PRs on fast checks.

---

### 6) **Low‑effort, high‑value monitoring & ergonomics**
Small changes that make life easier and help spot problems early.

- **Add per‑run summary emails or Slack posts** with top 5 regressions (if you have CI hooks).
- **Log histograms** of LUFS and true‑peak over time for each speaker to spot drift.
- **Expose a single “processing intensity” knob** (0–1) that scales de‑esser, EQ gain, and limiter aggressiveness — useful for tuning without changing code paths.
- **Keep opt‑in creative effects** (spectral enrich, room presence) disabled by default for game builds.

---

## Quick checklist you can hand to an engineer (copy/paste)
1. Add true‑peak measurement and store `true_peak_db` in manifest.
2. Use `pyloudnorm` for integrated LUFS; target **-18 LUFS**.
3. Replace brick‑wall limiting with soft look‑ahead limiter or soft‑knee limiter; report max gain reduction.
4. Scale de‑esser by `sibilance_ratio` (narrow band 4–8 kHz).
5. HPF at 80 Hz always; report LF energy (<200 Hz).
6. Per‑speaker baselines and `speaker_role` metadata.
7. Add crest factor, spectral tilt, and spectral centroid to diagnostics.
8. Add blind A/B listening on golden lines for any baseline change > 3% OVRL.
9. Treat vocalizations with separate presets and checks.
10. Keep compressor opt‑in; default to adaptive limiter.

---

## Final note — what I’d listen for first (practical listening checklist)
When you run a new build, do a quick 2‑minute listening pass on these items (one sentence each):
- **Clarity:** Are consonants intelligible? (If not, check de‑esser and presence EQ.)
- **Naturalness:** Any pumping, breathing artifacts, or metallicness? (Check compressor/limiter behavior.)
- **Dynamics:** Are loudness and peaks consistent with baseline? (Check LUFS and true‑peak.)
- **Character:** Has the voice lost body or become thin? (Check spectral tilt and low‑shelf EQ.)
- **Vocalizations:** Do sighs/whispers sound natural and sit correctly in the mix?

---

### Manifest fields (full schema) — required for reliable QA and regression tracing

Below is a **complete manifest schema** to include for every generated audio sample. Store this as JSON per sample in your session manifest; every field is required unless marked optional. Use consistent units (dB, Hz, seconds) and numeric types.

```json
{
  "test_name": "string",
  "sample_name": "string",                # e.g., "basic_speech_cicero"
  "speaker": "string",                    # speaker id (file basename)
  "speaker_role": "string",               # human-friendly archetype (e.g., "Nord male commoner")
  "text": "string",                       # prompt or reference text (nullable for vocalizations)
  "vocalization_tag": "string|null",      # e.g., "whispers", "moans" (null if none)
  "seed": "int",
  "audio_path": "string",                 # relative or absolute path to WAV
  "duration_s": "float",
  "sample_rate": "int",                   # e.g., 48000

  # Core signal diagnostics (pre- and post-processing)
  "raw": {
    "integrated_lufs": "float",           # LUFS before post-processing
    "true_peak_db": "float",              # dBTP before post-processing
    "peak_sample": "float",               # sample peak (linear 0-1)
    "rms": "float",
    "spectral_centroid_hz": "float",
    "sibilance_ratio": "float",           # energy 4-8kHz / total energy
    "crest_factor_db": "float",           # 20*log10(peak/rms)
    "spectral_tilt_db_per_octave": "float"
  },

  "processed": {
    "integrated_lufs": "float",           # LUFS after post-processing
    "true_peak_db": "float",              # dBTP after post-processing
    "peak_sample": "float",
    "rms": "float",
    "spectral_centroid_hz": "float",
    "sibilance_ratio": "float",
    "crest_factor_db": "float",
    "spectral_tilt_db_per_octave": "float",
    "max_gain_reduction_db": "float",     # limiter/compressor max reduction observed
    "limiter_applied": "bool",
    "compressor_applied": "bool",
    "deesser_applied": "bool",
    "adaptive_eq_applied": "bool"
  },

  # Gate and artifact detection (pure signal checks)
  "gate_results": {
    "has_clipping": "bool",
    "clipping_peak_db": "float|null",
    "trailing_silence_ms": "float",
    "leading_silence_ms": "float",
    "silence_ratio": "float",             # fraction of samples below silence threshold
    "zero_crossing_rate": "float",
    "has_trailing_artifact": "bool"
  },

  # Optional perceptual / ML scores (may be null if unavailable)
  "scores": {
    "dnsmos_sig": "float|null",
    "dnsmos_bak": "float|null",
    "dnsmos_ovrl": "float|null",
    "utmos": "float|null",
    "speaker_similarity": "float|null",
    "wer": "float|null",
    "cer": "float|null"
  },

  # Post-processing delta (processed minus raw)
  "post_processing_delta": {
    "rms_change_db": "float",
    "peak_change_db": "float",
    "spectral_centroid_change_hz": "float",
    "dnsmos_delta_sig": "float|null",
    "dnsmos_delta_ovrl": "float|null"
  },

  # Metadata for traceability
  "generation_config": {
    "num_steps": "int",
    "guidance_scale": "float",
    "model_commit": "string",
    "seed": "int",
    "other_params": "object|null"
  },

  "notes": "string|null"                   # free text for human notes (optional)
}
```

**Implementation notes (how to compute / why):**
- **Integrated LUFS:** use `pyloudnorm` integrated measurement. Target default: **-18 LUFS** for dialogue.
- **True peak (dBTP):** compute via 4× or 8× oversampling and measure peak in dBTP; target ≤ **-1 dBTP** after processing.
- **Sibilance ratio:** compute FFT magnitudes; ratio = energy(4–8 kHz) / total energy. Use this to scale de‑esser intensity.
- **Crest factor:** \(20\log_{10}(\text{peak}/\text{rms})\) — low crest factor often indicates over‑compression.
- **Spectral tilt:** measure slope of log‑spectrum (dB per octave) to detect thinning or excessive brightening.
- **Max gain reduction:** capture the maximum instantaneous gain reduction reported by limiter/compressor (helps detect aggressive processing).
- **Gate thresholds:** clipping if any sample ≥ 0.995 (or true‑peak > 0 dBTP); silence threshold ~ -60 dBFS.

---

### Short tuning preset table — conservative defaults for Skyrim dialogue
Yes — based on your plans these presets are necessary: they give engineers concrete, safe numbers to implement the adaptive pipeline without guesswork. Use them as **starting points**; tune per‑speaker baselines afterward.

| **Setting** | **Recommended value** | **Notes** |
|-------------|-----------------------|----------|
| **HPF cutoff** | **80 Hz** | 2nd–4th order Butterworth; always on for dialogue. |
| **Mud cut** | **Peaking @ 250–350 Hz, gain -1.5 dB, Q 0.7** | Gentle reduction; use only if centroid < 1500 Hz. |
| **Presence boost** | **High‑shelf @ 3.0–4.5 kHz, gain +0.8 dB, Q shelf** | Apply only if centroid > 3500 Hz and RMS < 0.15. |
| **De‑esser band** | **4.0–8.0 kHz, Q 6–10** | Narrow Q; dynamic gain reduction scaled by sibilance_ratio. |
| **De‑esser max reduction** | **-6 dB** | Cap to avoid dulling; typical applied < -3 dB. |
| **Limiter type** | **Look‑ahead or soft‑knee** | If look‑ahead unavailable, soft‑knee with short attack. |
| **Limiter threshold** | **-1.0 dBTP** | Keep headroom; prevents inter‑sample clipping. |
| **Limiter attack / release** | **Attack 0–2 ms, Release 50–150 ms** | Speech‑friendly settings. |
| **Max limiter reduction** | **3–6 dB** | Report this in diagnostics; flag if exceeded. |
| **Compressor (opt‑in)** | **Ratio 1.5–2.5:1, Threshold = RMS -6 dB, Knee 8 dB** | Slow attack (10 ms), release 80–150 ms; opt‑in only. |
| **EQ gain limits** | **±1.5 dB (shelves), ±3 dB (peaks)** | Keep boosts/cuts conservative to preserve character. |
| **Target LUFS** | **-18 LUFS** | Integrated target for normalized dialogue. |
| **Sibilance threshold** | **sibilance_ratio > 0.15** | Above this, de‑esser engages; intensity scaled to ratio. |
| **Vocalization whisper preset** | **BP 600–4000 Hz, volume factor 0.35, breath noise 0.10** | Do not normalize to same LUFS as speech; keep quieter. |
| **Vocalization moan/whimper** | **LPF 700–900 Hz, pitch shift -3 to -4 semitones, breath 0.08–0.10** | Separate preset; lower target LUFS and different EQ. |

---

### Short rationale: are these presets necessary given your plans?
Yes. Your plan already moves to a **signal‑adaptive** pipeline and changes LUFS to -18; the presets above are the minimal, conservative DSP parameters that let the adaptive logic behave predictably and audibly well across Skyrim voice archetypes. Without concrete numeric defaults, different engineers will implement inconsistent behavior and automated metrics will be noisy. These numbers keep processing **transparent, reversible, and safe** for game dialogue.

---
