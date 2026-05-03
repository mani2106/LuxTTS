# Plan: Integrate TDR Nova VST3 Plugin via Pedalboard

## Context

The LuxTTS post-processor (`utilities/post_processor.py`) currently uses a fallback pattern: pedalboard built-ins if available, otherwise scipy. TDR Nova is a free dynamic EQ VST3 plugin that can handle **both de-essing and EQ in a single pass** with higher quality than the current scipy/pedalboard-built-in implementations. The user has the plugin at `F:\Software\TDR Nova (no installer)\VST3\x64\TDR Nova.vst3`.

## Files to Modify

| File | Change |
|------|--------|
| `utilities/app_constants.py` | Add TDR Nova path + enable flag |
| `utilities/post_processor.py` | Add TDR Nova backend (combined de-esser + EQ) |
| `tests/test_post_processor.py` | Add TDR Nova tests |

New file:
| File | Purpose |
|------|---------|
| `utilities/tdr_nova_config.py` | TDR Nova parameter presets + discovery utility |

No changes needed: `audio_generation_pipeline.py` (API unchanged).

## Step 1: Add constants to `utilities/app_constants.py`

Add after the post-processing defaults block (line 32):

```python
# TDR Nova VST3 plugin
TDR_NOVA_VST3_PATH = Path(r"F:\Software\TDR Nova (no installer)\VST3\x64\TDR Nova.vst3")
TDR_NOVA_ENABLED = True  # Set False to use pedalboard/scipy fallback
```

## Step 2: Create `utilities/tdr_nova_config.py`

Encapsulates all TDR Nova knowledge:
- Band type constants (Low Shelf, Bell, High Shelf as raw values)
- Dynamics mode constants (Off, On, Sticky)
- `build_tts_preset(de_ess_intensity, eq_intensity) -> dict` — builds parameter dict for TTS speech
  - Band 1: Low-shelf warmth at 200Hz (+2dB × eq_intensity)
  - Band 2: Bell boxiness cut at 400Hz (-1.5dB × eq_intensity)
  - Band 3: Bell presence at 3kHz (+2dB × eq_intensity)
  - Band 4: High-shelf de-esser at 6500Hz with dynamic compression (scaled by de_ess_intensity)
  - HP: 80Hz high-pass to remove rumble
- `discover_parameters(plugin) -> dict` — utility to introspect plugin parameters

**IMPORTANT**: TDR Nova's internal parameter names are unknown until we load the plugin at runtime. The preset function will need to be adjusted after we run parameter discovery. The first implementation will include a discovery script that prints all parameter names/values, and we'll use that output to finalize the parameter mapping.

## Step 3: Modify `utilities/post_processor.py`

### 3a: Add TDR Nova detection (after line 23)

```python
from utilities.app_constants import TDR_NOVA_VST3_PATH, TDR_NOVA_ENABLED

HAS_TDR_NOVA = False
if HAS_PEDALBOARD and TDR_NOVA_ENABLED:
    if TDR_NOVA_VST3_PATH.exists():
        HAS_TDR_NOVA = True
    else:
        logger.info(f"TDR Nova VST3 not found at {TDR_NOVA_VST3_PATH}")
```

### 3b: Add lazy-loaded plugin singleton to `AudioPostProcessor`

Add `self._tdr_nova_plugin = None` in `__init__`, add a `tdr_nova_plugin` property that calls `pedalboard.load_plugin()` on first access and caches the result.

### 3c: Add `_process_tdr_nova()` method

Combined de-esser + EQ processing in one pass:
- Calls `build_tts_preset()` to get parameters
- Sets all parameters on the plugin
- Processes audio through the plugin
- Returns `(processed_audio, diagnostics)` — same format as other methods
- Wraps in try/except with fallback to original audio on failure
- NaN safety check on output

### 3d: Modify `process()` pipeline

Replace stages 1+2 (de-esser + EQ) with a branching path:

```python
# Stage 1+2: De-esser + EQ
if HAS_TDR_NOVA and self.tdr_nova_plugin is not None:
    audio, nova_diag = self._process_tdr_nova(audio, sr, de_ess_intensity, eq_intensity)
    if nova_diag:
        all_diagnostics['tdr_nova'] = nova_diag
else:
    # Existing fallback: separate de-esser + EQ stages
    audio, d = self.de_esser(audio, sr, intensity=de_ess_intensity)
    ...
    audio, d = self.equalize(audio, sr, intensity=eq_intensity)
    ...
```

Stages 3-5 (compress, pitch shift, normalize) remain unchanged.

## Step 4: Add tests to `tests/test_post_processor.py`

- `test_tdr_nova_combined_deess_and_eq` — processes audio, checks output shape
- `test_process_uses_tdr_nova_when_available` — verifies `process()` uses TDR Nova
- `test_tdr_nova_zero_intensity` — bypass with zero intensities
- `test_tdr_nova_fallback_when_missing` — monkeypatch `HAS_TDR_NOVA = False`, verify fallback
- All tests use `pytest.skip("TDR Nova not available")` when plugin absent (for CI)

## Step 5: Parameter Discovery (critical first step)

Before finalizing `build_tts_preset()`, we must discover TDR Nova's actual parameter names. Run with project venv activated:

```bash
cd F:\Studies\LuxTTS && .venv\Scripts\activate
python -c "
import pedalboard
plugin = pedalboard.load_plugin(r'F:\Software\TDR Nova (no installer)\VST3\x64\TDR Nova.vst3')
for name, param in plugin.parameters.items():
    print(f'{name}: {param} (raw={param.raw_value})')
"
```

This prints all ~76 parameters with their internal names and current values. We'll use this output to:
1. Map the correct parameter names (e.g., `band_4_frequency_hz` vs `Band 4 Freq`)
2. Identify raw_value encoding for enum params (band type, dynamics mode)
3. Set sensible defaults for TTS speech processing

After discovery, optionally use `plugin.show_editor()` to tune by ear on sample TTS output, then capture the tuned values.

## Fallback Chain

```
TDR Nova VST3 (best quality, combined de-ess + EQ)
    ↓ not available
Pedalboard built-ins (DeEsser + HighShelfFilter + PeakFilter)
    ↓ not installed
Scipy native implementations (bandpass + peaking EQ)
```

## Environment

**All Python commands MUST use the project venv at `F:\Studies\LuxTTS\.venv`**:
- Activation: `.venv\Scripts\activate`
- Package installs: `uv pip install <package>` (not bare `pip`)
- Script execution: run from project root with venv activated
- Tests: `pytest tests/test_post_processor.py -v` (from project root, venv activated)
- Subagents must also use this venv exclusively

## Verification

1. Activate venv, then run parameter discovery script to get actual parameter names
2. `python -c "from utilities.post_processor import HAS_TDR_NOVA; print(HAS_TDR_NOVA)"` → `True`
3. `pytest tests/test_post_processor.py -v` — all existing + new tests pass
4. Generate audio through the pipeline with/without TDR Nova and compare quality
5. Temporarily set `TDR_NOVA_ENABLED = False` → verify fallback works with no errors
