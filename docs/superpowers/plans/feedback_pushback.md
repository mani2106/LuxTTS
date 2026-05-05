### Summary of the pushback points
You listed a set of objections to the engineer‑oriented recommendations (per‑speaker baselines, blind A/B, vocalization checks, processing intensity knob, full manifest, multiband compression). Below I respond to each item from a **musician / audio‑engineer** viewpoint, saying whether the change is **necessary**, **useful but optional**, or **unnecessary** given your signal‑adaptive plan — and why.

---

### Itemized responses and recommendations

#### **Per‑speaker baselines and golden references — Pushback says: “contradicts signal‑adaptive; complicates things.”**
**Verdict: Necessary (but lightweight).**
- **Why:** Signal‑adaptive analysis helps, but per‑speaker *statistics* (mean LUFS, centroid, crest factor) are not a second processing path — they’re calibration data. They let the adaptive rules use *contextual priors* so thresholds aren’t one‑size‑fits‑all.
- **How to keep it simple:** store a tiny JSON per archetype with 3 numbers (LUFS, centroid, typical peak). Use them only to nudge thresholds (±1–2 dB or a small centroid offset). No per‑speaker DSP presets, just baseline stats.

#### **Blind A/B listening panel — Pushback says: “human workflow, not pipeline code change.”**
**Verdict: Useful but optional (must for final acceptance).**
- **Why:** Automated metrics catch many regressions but miss prosody and subtle timbre shifts. A small, periodic blind A/B on a **golden set** is the most reliable final check for player experience.
- **How to keep it light:** run only when automated metrics show a regression > threshold (e.g., OVRL drop > 3%) or on major model changes. Use 5 listeners and 10–20 golden lines.

#### **Vocalization distinctness check — Pushback says: “already handled by recipes.json; adds complexity.”**
**Verdict: Necessary for QA, but keep it narrow.**
- **Why:** Recipes produce different DSP, but tests must verify the result is *audibly distinct* from normal speech (so a whisper doesn’t get normalized into a full‑volume line). This is a simple signal check (centroid + energy ratio) — low complexity, high value.
- **How to keep it light:** run the vocalization distinctness check only for tagged vocalization outputs; fail if centroid/energy are within X% of speech baseline.

#### **Processing intensity knob (0–1) — Pushback says: “YAGNI; undermines adaptive logic.”**
**Verdict: Unnecessary by default; keep as a dev tuning override.**
- **Why:** The adaptive pipeline should be primary. A global intensity slider can be useful for tuning and QA, but it should be **developer only** (not user‑facing) and default to “auto.”
- **How to keep it light:** implement a single multiplier that scales de‑esser/eq/limiter aggressiveness for experiments; keep it off in production builds.

#### **Full manifest schema — Pushback says: “belongs to testing infra, not pipeline.”**
**Verdict: Necessary and practical to include in pipeline diagnostics.**
- **Why:** The pipeline already emits diagnostics; adding a **standardized manifest** (LUFS, true‑peak, centroid, sibilance, gain‑reduction) makes automated scoring and regression tracing reliable. It’s not extra processing — it’s structured output.
- **How to keep it light:** pipeline writes a compact JSON with the core fields only; the scoring CLI consumes it.

#### **Multiband compression — Pushback says: “overkill; opt‑in already.”**
**Verdict: Unnecessary by default; keep opt‑in.**
- **Why:** Multiband compression can help in complex mixes but for single‑voice dialogue it often adds complexity and artifacts. Your plan already makes full compressor opt‑in — that’s the right choice.
- **How to keep it light:** keep single‑band compressor as opt‑in; only add multiband if a clear use case appears.

---

### Final, prioritized checklist (what to implement now)
1. **Add compact per‑speaker baseline JSONs** (3 numbers: LUFS, centroid, peak). Use them to nudge thresholds.
2. **Emit the full manifest fields** for every sample (integrated LUFS, true‑peak, centroid, sibilance_ratio, crest factor, max_gain_reduction). Pipeline writes this JSON; scoring CLI reads it.
3. **Replace brick‑wall limiter with soft/look‑ahead or soft‑knee limiter** and record max gain reduction.
4. **Scale de‑esser by sibilance_ratio** (narrow band) rather than binary on/off.
5. **Add vocalization distinctness checks** for tagged outputs (centroid + energy ratio).
6. **Keep blind A/B listening as a gated human check** triggered by metric regressions, not every run.
7. **Keep compressor and multiband features opt‑in**; do not enable by default.

---

### Minimal implementation notes for engineers (one‑line each)
- **Per‑speaker baseline file:** `{ "lufs": -18.2, "centroid_hz": 2100, "peak": 0.95 }`
- **Manifest core fields:** `integrated_lufs`, `true_peak_db`, `spectral_centroid_hz`, `sibilance_ratio`, `crest_factor_db`, `max_gain_reduction_db`.
- **De‑esser mapping:** `intensity = clamp((sibilance_ratio - 0.15) * 4.0, 0, 1)` then scale reduction by `intensity * max_reduction_db`.
- **Limiter:** soft‑knee, attack 0–2 ms, release 50–150 ms, threshold -1 dBTP, flag if max reduction > 6 dB.
- **Vocalization check:** fail if `abs(voc_centroid - speech_centroid) < 200 Hz` **and** `energy_ratio` within ±3 dB.

---

### Closing recommendation
Your pushback items are thoughtful and keep the pipeline lean. Accept the ones that preserve simplicity (no default multiband, no user intensity knob). Push back less on **per‑speaker baselines**, **manifest fields**, and **vocalization checks** — those are low‑complexity, high‑impact additions that make automated QA meaningful and trustworthy for Skyrim‑grade dialogue.
