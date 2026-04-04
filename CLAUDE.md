# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LuxTTS is a lightweight text-to-speech system built on ZipVoice architecture with distillation (4-step flow matching vs original 16), a custom 48kHz Vocos vocoder, and voice cloning. Models are hosted on HuggingFace (`YatharthS/LuxTTS`).

## Setup

```bash
# Activate the project virtual environment (required)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

Python >=3.10 required. The `linacodec` dependency is fetched from a custom git repo (`https://github.com/ysharma3501/LinaCodec.git`). The `piper_phonemize` package uses a custom find-links index.

For training: `uv pip install -e ".[train]"` (adds `einops`).

## Virtual Environment

**IMPORTANT**: This project uses a virtual environment located at `.venv/` in the project root.

- **All Python commands must use this venv** — ensure `.venv/bin/activate` (or `.venv\Scripts\activate` on Windows) is sourced before running any Python commands
- **Subagents must use this venv exclusively** — when spawning agents for development tasks, ensure they activate the project venv at `.venv` rather than creating or using other environments
- Do not create additional virtual environments for this project
- **Always use `uv` prefix for Python package installations** — e.g., `uv pip install <package>` instead of `pip install <package>` to ensure packages are installed to the correct venv

## Architecture

### SkyrimNet-LuxTTS Server (NEW)
- **`SkyrimNet-LuxTTS.py`** — Gradio HTTP server with Web UI for interactive use
- **`skyrimnet_api.py`** — Standalone FastAPI server for SkyrimNet GamePlugin. Implements Gradio-compatible endpoints (`/gradio_api/upload`, `/gradio_api/call/generate_audio`, `/gradio_api/file=`, `/gradio_api/config`) without Gradio dependency. Includes voice file indexing, non-blocking generation via `asyncio.to_thread()`, generation timeout, and cached file eviction.
- **`utilities/`** — Server support modules:
  - `app_config.py` — Configuration with CLI argument parsing
  - `app_constants.py` — Server defaults and constants
  - `model_utils.py` — Model singleton loading with HuggingFace downloads
  - `audio_utils.py` — Audio I/O (WAV load/save, silence generation)
  - `cache_utils.py` — Two-tier caching (memory + disk) for speaker embeddings
  - `audio_generation_pipeline.py` — Orchestration: encode → generate → vocode → save. Routes through vocalization pipeline when `[bracket]` tags detected, otherwise uses normal TTS path.
  - `vocalization/` — Vocalization audio tag processing:
    - `tag_parser.py` — Parse ElevenLabs-style `[bracket]` audio tags from text
    - `dsp_engine.py` — Audio DSP effects (pitch shift, filters, distortion, breath noise, etc.)
    - `recipes.py` — Load tag → DSP recipe mappings from JSON config
    - `vocalization_generator.py` — Orchestrate TTS + DSP for vocalization segments
    - `stitcher.py` — Crossfade audio segments together
    - `recipes.json` — DSP recipe definitions for 14 supported tags
- **`speakers/en/`** — Bundled preset voice samples for Skyrim characters

### User-Facing API
- **`zipvoice/luxvoice.py`** — `LuxTTS` class: the primary entry point. Handles model loading (auto-detects CUDA/MPS/CPU), `encode_prompt()`, and `generate_speech()`.
- **`zipvoice/modeling_utils.py`** — Core inference plumbing: `load_models_gpu()`, `load_models_cpu()`, `process_audio()`, `generate()`. Config via `LuxTTSConfig` dataclass.
- **`zipvoice/onnx_modeling.py`** — `OnnxModel` class and `generate_cpu()` for ONNX-based CPU inference.

### Model Hierarchy
- **`zipvoice/models/zipvoice.py`** — Base `ZipVoice` model (flow matching with ZipFormer encoders).
- **`zipvoice/models/zipvoice_distill.py`** — `ZipVoiceDistill(ZipVoice)` — distilled variant used by LuxTTS. Uses `DistillEulerSolver` for fewer sampling steps.
- **`zipvoice/models/zipvoice_dialog.py`** — Dialog-oriented variant (two-stream ZipFormer).
- **`zipvoice/models/modules/`** — Core neural net building blocks: `zipformer.py`, `zipformer_two_stream.py`, `solver.py` (Euler solver for flow matching), `scaling.py`.

### Supporting Systems
- **`zipvoice/tokenizer/`** — Text tokenization: `EmiliaTokenizer` (default), `LibriTTSTokenizer`, `EspeakTokenizer`, `SimpleTokenizer`. Normalization in `normalizer.py`.
- **`zipvoice/utils/`** — Feature extraction (`VocosFbank`), inference utilities (chunking, cross-fade, silence removal, RMS norm), checkpoint loading, LR scheduling, TensorRT export helpers.
- **`zipvoice/bin/`** — CLI scripts for training (`train_zipvoice*.py`), inference (`infer_zipvoice*.py`), dataset prep, ONNX/TensorRT export, and model averaging.

### Inference Flow
1. **Prompt encoding**: Audio → librosa load (24kHz) → Whisper transcription → RMS normalization → VocosFbank features → tokenization.
2. **Speech generation**: Text tokenization → ZipVoiceDistill.sample() (flow matching, configurable steps/guidance/speed) → Vocos vocoder decode (48kHz, freq_range=12000) → volume matching.
3. **CPU path** uses ONNX runtime for text encoder + flow-matching decoder, PyTorch Vocos for vocoding.

### Key Model Config
Model config is loaded from `config.json` in the HuggingFace repo. `ZipVoiceDistill.__init__` requires: `feat_dim`, `fm_decoder_*` params, `query_head_dim`, `pos_head_dim`, `value_head_dim`, `pos_dim`, `time_embed_dim`.

## Linting

Ruff is configured with `line-length = 120`.

## Notes

- GPU inference uses PyTorch directly; CPU inference uses ONNX Runtime with `whisper-tiny` (vs `whisper-base` on GPU).
- The Vocos vocoder has weight parametrizations that must be removed before `load_state_dict` (see `load_models_gpu`/`load_models_cpu`).
- Output audio is always 48kHz. `return_smooth=True` disables 48k upsampling for a different quality tradeoff.
- **Server test suite**: The SkyrimNet-LuxTTS server has tests in `tests/` — run with `pytest tests/ -v`.

## Vocalization Audio Tags

The server supports ElevenLabs-style `[bracket]` audio tags for non-speech vocalizations embedded within dialogue text.

### Supported Tags

**Human Reactions:** sighs, groans, moans
**Emotional:** gasps, screams, shouts, laughs, sobs, whimpers
**Breath/Air:** breathes heavily, clears throat, coughs
**Delivery:** whispers (modifies next speech segment), pause

### Usage Examples

```
[sighs] I can't believe we made it.
Stop! [gasps] How did you find me?
[whispers] Don't make a sound.
[screams] Get away from me!
[breathes heavily] We need to keep moving.
Hello [pause] my friend.
```

### Implementation

**Location:** `utilities/vocalization/` package

**Pipeline:**
1. `tag_parser.py` — Parse text into speech/vocalization segments using regex
2. `vocalization_generator.py` — For each segment:
   - Vocalization: TTS with recipe's `tts_text` → apply DSP effects
   - Speech: Normal TTS generation
   - Whisper mode: Apply DSP effects to following speech
3. `stitcher.py` — Crossfade stitch all segments together
4. Post-processing: Apply existing compression, EQ, normalization

**DSP Effects** (`dsp_engine.py`):
- pitch_shift, time_stretch, speed_up
- low_pass_filter, high_pass_filter, high_shelf_boost
- breath_noise, distortion, compress
- fade_in, fade_out, volume

**Recipes** (`recipes.json`): JSON config mapping tags to TTS text and DSP effect chains. No YAML dependency.

### Performance

- **Zero overhead** when no tags present (single regex scan)
- **With tags**: ~100-300ms extra DSP per vocalization segment
- All DSP uses numpy/scipy — no GPU required

### Future Extension

The `VocalizationGenerator` class is an abstraction boundary. Can be replaced with neural vocalization models (NVSpeech, CosyVoice fine-tune) without changing parser/stitcher/integration.


<!-- Add your custom instructions below. Repowise will never modify anything outside the REPOWISE markers. -->
<!-- Examples: coding style rules, test commands, workflow preferences, constraints -->

<!-- REPOWISE:START — Do not edit below this line. Auto-generated by Repowise. -->
## IMPORTANT: Codebase Intelligence Instructions for LuxTTS

> **CRITICAL**: This repository is indexed by [Repowise](https://repowise.dev).
> You MUST use the repowise MCP tools below instead of reading raw source files.
> They deliver richer context — documentation, ownership, history, decisions —
> in a single call. Raw `read_file` calls are a last resort only.

Last indexed: 2026-04-04
### Tech Stack
**Languages:** Python
**Frameworks:** PyTorch

### Hotspots (High Churn)
| File | Churn | 90d Commits | Owner |
|------|-------|-------------|-------|
| `utilities/post_processor.py` | 98.4th %ile | 9 | mani2106 |
| `tests/test_vocalization.py` | 96.7th %ile | 7 | mani2106 |
| `tests/test_post_processor.py` | 95.1th %ile | 7 | mani2106 |
| `utilities/audio_generation_pipeline.py` | 93.4th %ile | 6 | mani2106 |
| `utilities/app_constants.py` | 91.8th %ile | 5 | mani2106 |

### Repowise MCP Tools

This project has a Repowise MCP server configured. **ALWAYS use these tools before reading source files** — they provide documentation, ownership, architectural decisions, and risk signals in a single call.

**MANDATORY workflow:**

1. **ALWAYS start with `get_overview()`** on any new task to orient yourself.
2. **MUST call `get_context(targets=["path/to/file.py"])`** before reading or editing any file — faster and richer than reading source directly.
3. **MUST call `get_risk(targets=["path/to/file.py"])`** before making changes — especially for hotspot files.
4. Don't know where something lives? Call `search_codebase(query="authentication flow")` — prefer over grep/find.
5. Need to understand why code is structured a certain way? **MUST call `get_why(query="why JWT over sessions")`** before any architectural change.
6. **After completing any code changes**, **MUST call `update_decision_records(action="list")`** to review existing decisions, then:
   - Call `update_decision_records(action="create", title="...", decision="...", rationale="...", affected_files=[...])` to record any new architectural decisions you made.
   - Call `update_decision_records(action="update", decision_id="...", ...)` to update existing decisions affected by your changes.
   - Call `update_decision_records(action="update_status", decision_id="...", status="deprecated")` if your changes invalidate an existing decision.
   **This is not optional** — keeping decision records current is a required part of every coding task.
7. Need to understand how two modules connect? Call `get_dependency_path(source="src/auth", target="src/db")`.
8. Before cleanup tasks, call `get_dead_code()` to find confirmed unused code.
9. For documentation or diagrams, call `get_architecture_diagram(scope="src/auth")`.

| Tool | WHEN you MUST use it |
|------|----------------------|
| `get_overview()` | **FIRST call on every new task** |
| `get_context(targets=[...])` | **Before reading or modifying any file** |
| `get_risk(targets=[...])` | Before changing files — REQUIRED for hotspots |
| `get_why(query="...")` | Before architectural changes — REQUIRED |
| `update_decision_records(action=...)` | **After every coding task** — record and update decisions |
| `search_codebase(query="...")` | When locating code — prefer over grep/find |
| `get_dependency_path(source=..., target=...)` | When tracing module connections |
| `get_dead_code()` | Before any cleanup or removal |
| `get_architecture_diagram(scope=...)` | For visual structure or documentation |

### Codebase Conventions
**Commands:**
- Test: `pytest`
- Lint: `ruff check .`

<!-- REPOWISE:END -->
