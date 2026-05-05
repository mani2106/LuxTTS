"""Tests for speaker sample curation pipeline."""
import os
import sys
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.build_speaker_samples import (
    resolve_speaker_voice_map,
    score_clip,
    crossfade_concat,
    normalize_rms,
)


def test_resolve_mapping():
    voice_types = {
        "femalecommander": ["/fake/a.wav"],
        "femaleuniqueastrid": ["/fake/b.wav"],
        "maleuniqueancano": ["/fake/c.wav"],
        "cruniquealduin": ["/fake/d.wav"],
        "dlc1seranavoice": ["/fake/e.wav"],
        "dlc2maleuniqueadril": ["/fake/f.wav"],
    }
    with tempfile.TemporaryDirectory() as td:
        for name in ["femalecommander", "astrid", "ancano", "alduin", "serana", "adril"]:
            Path(td, f"{name}.wav").touch()
        mapping = resolve_speaker_voice_map(td, voice_types)

    assert mapping["femalecommander"] == "femalecommander"
    assert mapping["astrid"] == "femaleuniqueastrid"
    assert mapping["ancano"] == "maleuniqueancano"
    assert mapping["alduin"] == "cruniquealduin"
    assert mapping["serana"] == "dlc1seranavoice"
    assert mapping["adril"] == "dlc2maleuniqueadril"


def test_score_clip_good():
    sr = 22050
    t = np.linspace(0, 3, 3 * sr, dtype=np.float32)
    y = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        result = score_clip(f.name)
    assert result is not None
    score, dur, rms_db = result
    assert 2.5 < dur < 3.5
    assert score > 0.5


def test_score_clip_silence():
    sr = 22050
    y = np.zeros(3 * sr, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        result = score_clip(f.name)
    assert result is None


def test_crossfade_concat():
    sr = 22050
    t = np.linspace(0, 1, sr, dtype=np.float32)
    y1 = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y2 = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)

    paths = []
    for y in [y1, y2]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, y, sr)
            paths.append(f.name)

    result, out_sr = crossfade_concat(paths, crossfade_ms=50, target_sr=sr)
    expected_len = 2 * sr - int(sr * 0.05)
    assert abs(len(result) - expected_len) < 100

    for p in paths:
        Path(p).unlink(missing_ok=True)


def test_normalize_rms():
    y = np.ones(1000, dtype=np.float32) * 0.1
    result = normalize_rms(y, target_db=-20)
    expected_rms = 10 ** (-20 / 20)
    actual_rms = np.sqrt(np.mean(result ** 2))
    assert abs(actual_rms - expected_rms) < 1e-6
