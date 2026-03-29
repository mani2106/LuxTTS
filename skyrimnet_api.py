#!/usr/bin/env python3
"""
SkyrimNet-LuxTTS: Standalone FastAPI server for SkyrimNet GamePlugin.

Implements Gradio-compatible API endpoints that SkyrimNet calls:
  POST /gradio_api/upload                     — upload voice files
  POST /gradio_api/call/generate_audio        — start generation
  GET  /gradio_api/call/generate_audio/{id}   — poll for result
  GET  /gradio_api/file={filename}            — fetch generated audio
  GET  /gradio_api/config                     — API discovery

Usage:
    python skyrimnet_api.py --server 0.0.0.0 --port 7860
"""

import asyncio
import io
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from utilities.app_config import AppConfig
from utilities.app_constants import (
    OUTPUT_DIR,
    SAMPLE_RATE,
    SPEAKERS_DIR,
)
from utilities.audio_generation_pipeline import generate_audio, init_speaker_cache
from utilities.audio_utils import create_silence, save_wav_file
from utilities.model_utils import load_model_if_needed

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence file for ping responses
SILENCE_PATH = OUTPUT_DIR / "silence_100ms.wav"

# Max generated files to keep in the path lookup (evicts oldest when exceeded)
MAX_CACHED_FILES = 256

# Timeout for a single TTS generation (seconds)
GENERATION_TIMEOUT = 60

# Voice file index for O(1) lookups  {lowercase_name: Path}
VOICE_FILE_MAP: dict[str, Path] = {}


def _index_voices():
    """Scan speakers/ directories and build a name→path index."""
    for speakers_root in [SPEAKERS_DIR, Path("speakers")]:
        if not speakers_root.exists():
            continue
        for wav in speakers_root.rglob("*.wav"):
            # Index by stem (e.g. "malecommoner") and by full name
            VOICE_FILE_MAP[wav.stem.lower()] = wav
            VOICE_FILE_MAP[wav.name.lower()] = wav
    logger.info(f"Indexed {len(VOICE_FILE_MAP)} voice files")


# ---------------------------------------------------------------------------
# Speaker path resolution
# ---------------------------------------------------------------------------

def _extract_voice_name(raw_voice) -> str | None:
    """Extract a usable name/string from whatever SkyrimNet sends for voice."""
    if raw_voice is None:
        return None
    if isinstance(raw_voice, dict):
        return raw_voice.get("orig_name") or raw_voice.get("path") or raw_voice.get("name")
    return str(raw_voice)


def _resolve_speaker_path(name: str, language: str = "en") -> str | None:
    """Resolve a speaker name to a file path.

    Lookup order:
      1. Already a valid file on disk
      2. VOICE_FILE_MAP index (O(1))
      3. speakers/{language}/{name}.wav
      4. speakers/{name}.wav
    """
    if not name or not name.strip():
        return None

    name = str(name).strip()

    # Already a valid file path
    p = Path(name)
    if p.is_file():
        return str(p)

    # O(1) index lookup
    clean = name.replace(".wav", "").lower()
    if clean in VOICE_FILE_MAP:
        return str(VOICE_FILE_MAP[clean])
    full_key = f"{clean}.wav"
    if full_key in VOICE_FILE_MAP:
        return str(VOICE_FILE_MAP[full_key])

    # Filesystem fallback
    for candidate in [
        SPEAKERS_DIR / f"{clean}.wav",
        Path("speakers") / language / f"{clean}.wav",
        Path("speakers") / f"{clean}.wav",
    ]:
        if candidate.is_file():
            logger.info(f"Resolved speaker '{name}' -> {candidate}")
            return str(candidate)

    return None


def _validate_audio_path(raw_value, language: str = "en") -> str | None:
    """Validate and resolve audio path from SkyrimNet's various formats."""
    if raw_value is None:
        return None

    # Dict from Gradio-style upload
    if isinstance(raw_value, dict) and "path" in raw_value:
        raw_value = raw_value["path"]

    if not raw_value or (isinstance(raw_value, str) and not raw_value.strip()):
        return None

    raw_str = str(raw_value).strip()

    # Reject directories
    if Path(raw_str).is_dir():
        return None

    # Already a file
    if Path(raw_str).is_file():
        return raw_str

    # Try speaker name resolution
    return _resolve_speaker_path(raw_str, language)


# ---------------------------------------------------------------------------
# Generate silence on import (for ping responses)
# ---------------------------------------------------------------------------

def _ensure_silence_file():
    """Create a short silence WAV if it doesn't exist."""
    if not SILENCE_PATH.exists():
        silence = create_silence(0.1, SAMPLE_RATE)
        save_wav_file(silence, SILENCE_PATH, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialize caches on startup."""
    config: AppConfig = app.state.config

    logger.info(f"Loading model on {config.device}...")
    load_model_if_needed(config)
    logger.info("Model loaded")

    await init_speaker_cache(config)
    _ensure_silence_file()
    _index_voices()
    logger.info("Speaker cache initialized")

    yield


app = FastAPI(title="SkyrimNet-LuxTTS", lifespan=lifespan)


# ---------------------------------------------------------------------------
# POST /gradio_api/upload
# ---------------------------------------------------------------------------

@app.post("/gradio_api/upload")
async def upload_voice_files(files: list[UploadFile] = File(...)):
    """Accept voice reference file uploads from SkyrimNet Web UI."""
    paths = []
    for f in files:
        dest = SPEAKERS_DIR / f.filename
        if not dest.exists() or dest.stat().st_size == 0:
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            # Update live index
            VOICE_FILE_MAP[dest.stem.lower()] = dest
            VOICE_FILE_MAP[dest.name.lower()] = dest
            logger.info(f"Uploaded voice: {dest}")
        else:
            logger.debug(f"Voice already exists, skipping: {dest}")
        paths.append(str(dest))
    return paths


# ---------------------------------------------------------------------------
# POST /gradio_api/call/generate_audio
# ---------------------------------------------------------------------------

@app.post("/gradio_api/call/generate_audio")
async def start_generation(request: Request):
    """
    SkyrimNet calls this to start TTS generation.

    Expects Gradio-format body: ``{"data": [...]}``
    Returns: ``{"event_id": "..."}``
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    data = body.get("data", [])

    # --- Extract parameters by index (SkyrimNet contract) ---
    text = data[1] if len(data) > 1 else ""
    language = data[2] if len(data) > 2 else "en"
    raw_voice = data[3] if len(data) > 3 else None
    cfg_scale = data[19] if len(data) > 19 else None
    seed_num = data[26] if len(data) > 26 else None
    randomize_seed = data[27] if len(data) > 27 else True

    # --- Post-processing parameters (optional, use defaults if not provided) ---
    enable_post_processing = data[28] if len(data) > 28 and data[28] is not None else True
    pitch_shift = data[29] if len(data) > 29 else None
    eq_intensity = data[30] if len(data) > 30 else 1.0
    de_ess_intensity = data[31] if len(data) > 31 else 0.5
    compressor_threshold = data[32] if len(data) > 32 else -6.0
    compressor_ratio = data[33] if len(data) > 33 else 4.0
    target_loudness = data[34] if len(data) > 34 else -16.0

    # --- Ping / health-check ---
    if text == "ping":
        event_id = str(uuid.uuid4())
        filename = SILENCE_PATH.name
        result = [_file_result(str(SILENCE_PATH), filename)]
        app.state.job_store[event_id] = result
        app.state.audio_files[filename] = str(SILENCE_PATH)
        return {"event_id": event_id}

    # --- Resolve speaker ---
    voice_name = _extract_voice_name(raw_voice)
    speaker_path = _validate_audio_path(voice_name, language=language or "en")

    # --- Resolve generation params ---
    config: AppConfig = app.state.config
    cfg = float(cfg_scale) if cfg_scale is not None else config.default_guidance_scale
    seed = int(seed_num) if seed_num is not None else config.default_seed
    rand_seed = bool(randomize_seed)

    logger.info(
        f"Generating: text={text!r:.60}  speaker={Path(speaker_path).stem if speaker_path else 'default'}  "
        f"cfg={cfg}  seed={seed}"
    )

    # --- Run generation in worker thread (keeps event loop responsive) ---
    try:
        output_path, seed_used = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: asyncio.run(generate_audio(
                    text=text or "",
                    speaker_audio=speaker_path,
                    language=f"{language}-us",
                    cfg_scale=cfg,
                    seed=seed,
                    randomize_seed=rand_seed,
                    speed=config.default_speed,
                    num_steps=config.default_num_steps,
                    config=config,
                    enable_post_processing=enable_post_processing,
                    pitch_shift=pitch_shift,
                    eq_intensity=eq_intensity,
                    de_ess_intensity=de_ess_intensity,
                    compressor_threshold_offset=compressor_threshold,
                    compressor_ratio=compressor_ratio,
                    target_loudness=target_loudness,
                ))
            ),
            timeout=GENERATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f"Generation timed out after {GENERATION_TIMEOUT}s")
        return JSONResponse({"error": f"Generation timed out after {GENERATION_TIMEOUT}s"}, status_code=504)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    # --- Build Gradio-compatible result ---
    filename = Path(output_path).name
    event_id = str(uuid.uuid4())
    result = [_file_result(output_path, filename)]
    app.state.job_store[event_id] = result
    app.state.audio_files[filename] = output_path

    # Evict oldest entries if cache exceeds limit (dicts preserve insertion order)
    while len(app.state.audio_files) > MAX_CACHED_FILES:
        app.state.audio_files.pop(next(iter(app.state.audio_files)))

    logger.info(f"Generated {filename} (seed={seed_used})")
    return {"event_id": event_id}


# ---------------------------------------------------------------------------
# GET /gradio_api/call/generate_audio/{event_id}
# ---------------------------------------------------------------------------

@app.get("/gradio_api/call/generate_audio/{event_id}")
async def poll_generation(event_id: str):
    """Poll for a completed generation result (SSE stream)."""
    if event_id not in app.state.job_store:
        raise HTTPException(status_code=404, detail="Unknown event_id")

    payload = json.dumps(app.state.job_store.pop(event_id))
    sse = f"event: complete\ndata: {payload}\n\n"
    return StreamingResponse(
        io.BytesIO(sse.encode()),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# GET /gradio_api/file={filename}
# ---------------------------------------------------------------------------

@app.api_route("/gradio_api/file={file_path:path}", methods=["GET", "HEAD"])
async def fetch_file(file_path: str):
    """Serve a generated audio file."""
    filename = Path(file_path).name

    # Check in-memory map first
    resolved = app.state.audio_files.get(filename)
    if resolved and Path(resolved).is_file():
        return FileResponse(resolved, media_type="audio/wav")

    # Fallback to output directory
    disk_path = OUTPUT_DIR / filename
    if disk_path.is_file():
        return FileResponse(str(disk_path), media_type="audio/wav")

    return Response(status_code=404)


# ---------------------------------------------------------------------------
# GET /gradio_api/config  — Gradio JS client API discovery
# ---------------------------------------------------------------------------

@app.get("/gradio_api/config")
async def gradio_config():
    """Return a minimal Gradio interface config for JS client discovery."""
    return {
        "mode": "blocks",
        "app_id": str(uuid.uuid4()),
        "components": [
            {"id": 0, "type": "textbox", "props": {"label": "Text"}},
            {"id": 1, "type": "textbox", "props": {"label": "Text to synthesize", "visible": True}},
            {"id": 2, "type": "dropdown", "props": {"label": "Language", "choices": ["en"], "value": "en"}},
            {"id": 3, "type": "audio", "props": {"label": "Speaker Audio", "type": "filepath"}},
            {"id": 4, "type": "audio", "props": {"label": "Prefix Audio", "type": "filepath"}},
            *[
                {"id": i, "type": "number", "props": {"label": f"param_{i}", "visible": False}}
                for i in range(5, 26)
            ],
            {"id": 26, "type": "number", "props": {"label": "Seed", "value": 420}},
            {"id": 27, "type": "checkbox", "props": {"label": "Randomize Seed", "value": True}},
            {"id": 28, "type": "checkbox", "props": {"label": "Enable Post-Processing", "value": True}},
            {"id": 29, "type": "slider", "props": {"label": "Pitch Shift", "minimum": -12, "maximum": 12, "step": 0.5, "value": 0}},
            {"id": 30, "type": "slider", "props": {"label": "EQ Intensity", "minimum": 0, "maximum": 1, "step": 0.1, "value": 1.0}},
            {"id": 31, "type": "slider", "props": {"label": "De-Ess Intensity", "minimum": 0, "maximum": 1, "step": 0.1, "value": 0.5}},
            {"id": 32, "type": "slider", "props": {"label": "Compressor Threshold", "minimum": -30, "maximum": 0, "step": 1, "value": -6.0}},
            {"id": 33, "type": "slider", "props": {"label": "Compressor Ratio", "minimum": 1, "maximum": 20, "step": 0.5, "value": 4.0}},
            {"id": 34, "type": "slider", "props": {"label": "Target Loudness", "minimum": -30, "maximum": -5, "step": 1, "value": -16.0}},
            {"id": 35, "type": "audio", "props": {"label": "Output Audio", "type": "filepath"}},
            {"id": 36, "type": "number", "props": {"label": "Seed Used"}},
        ],
        "dependencies": [
            {
                "id": 0,
                "api_name": "generate_audio",
                "inputs": list(range(35)),
                "outputs": [35, 36],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_result(path: str, filename: str) -> dict:
    """Build a Gradio-compatible file-result dict."""
    return {
        "path": path,
        "url": f"/gradio_api/file={filename}",
        "orig_name": filename,
        "mime_type": "audio/wav",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = AppConfig.from_cli()

    # Initialize shared state before uvicorn starts
    app.state.config = config
    app.state.job_store = {}       # event_id -> result payload
    app.state.audio_files = {}     # filename -> absolute path

    logger.info(f"Starting SkyrimNet-LuxTTS API on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)
