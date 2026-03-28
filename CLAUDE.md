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
- **`SkyrimNet-LuxTTS.py`** — Gradio HTTP server that wraps LuxTTS as a SkyrimNet GamePlugin backend
- **`utilities/`** — Server support modules:
  - `app_config.py` — Configuration with CLI argument parsing
  - `app_constants.py` — Server defaults and constants
  - `model_utils.py` — Model singleton loading with HuggingFace downloads
  - `audio_utils.py` — Audio I/O (WAV load/save, silence generation)
  - `cache_utils.py` — Two-tier caching (memory + disk) for speaker embeddings
  - `audio_generation_pipeline.py` — Orchestration: encode → generate → vocode → save
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
