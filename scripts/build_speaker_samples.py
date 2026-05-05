#!/usr/bin/env python
"""Build composite speaker samples from extracted Skyrim voice clips."""

import argparse
import os
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser(description="Build speaker samples from extracted voice clips")
    parser.add_argument("--input", required=True, help="Root directory containing DLC voice folders")
    parser.add_argument("--output", required=True, help="Output directory for composite samples")
    parser.add_argument("--speakers-dir", default="speakers/en", help="Directory of existing speaker WAVs to match names against")
    parser.add_argument("--target-duration", type=float, default=12.0, help="Target composite duration in seconds")
    parser.add_argument("--max-clips", type=int, default=5, help="Maximum clips per composite")
    parser.add_argument("--skip-validate", action="store_true", help="Skip TTS model validation")
    parser.add_argument("--dry-run", action="store_true", help="Print selection results without writing files")
    return parser.parse_args()


def get_existing_speaker_names(speakers_dir):
    """Return set of lowercase speaker names from existing WAV files."""
    names = set()
    if not os.path.isdir(speakers_dir):
        return names
    for f in os.listdir(speakers_dir):
        if f.endswith(".wav") and "Hardlink" not in f and f != "empty_100ms.wav":
            names.add(f.replace(".wav", "").lower())
    return names


def discover_voice_types(input_dir):
    """Discover all voice type folders across DLC directories.

    Returns dict: {voice_type_folder_name: [list of wav paths]}
    """
    voice_types = {}
    for dlc_dir in sorted(Path(input_dir).iterdir()):
        if not dlc_dir.is_dir():
            continue
        for vt_dir in sorted(dlc_dir.iterdir()):
            if not vt_dir.is_dir():
                continue
            wavs = sorted(str(p) for p in vt_dir.glob("*.wav"))
            if wavs:
                vt_name = vt_dir.name.lower()
                if vt_name not in voice_types:
                    voice_types[vt_name] = []
                voice_types[vt_name].extend(wavs)
    return voice_types


def resolve_speaker_voice_map(speakers_dir, voice_types):
    """Map existing speaker names to voice type folders.

    Strategy (first match wins):
    1. Exact match (e.g. "femalecommander" -> "femalecommander")
    2. Voice type contains speaker name with "unique" prefix/suffix
       (e.g. "astrid" -> "femaleuniqueastrid")
    3. Speaker name appears in voice type with dlc prefix
       (e.g. "serana" -> "dlc1seranavoice")

    Returns dict: {speaker_name: voice_type_name}
    """
    existing = get_existing_speaker_names(speakers_dir)
    if not existing:
        return {}

    # Filter out names that are clearly not voice types
    skip_names = {"aaaharleyvoicequest", "ciri_new_combined", "vp_11_paxti", "night_mother"}
    existing -= skip_names

    vt_names = set(voice_types.keys())
    mapping = {}

    for name in sorted(existing):
        matched_vt = None

        # 1. Exact match
        if name in vt_names:
            matched_vt = name
        # 2. femaleunique{name} or maleunique{name}
        elif f"femaleunique{name}" in vt_names:
            matched_vt = f"femaleunique{name}"
        elif f"maleunique{name}" in vt_names:
            matched_vt = f"maleunique{name}"
        # 3. crunique{name} (creature uniques like alduin, paarthurnax, odahviing)
        elif f"crunique{name}" in vt_names:
            matched_vt = f"crunique{name}"
        # 4. dlc prefix containing name (serana -> dlc1seranavoice)
        else:
            for vt in vt_names:
                if name in vt and vt.startswith("dlc"):
                    matched_vt = vt
                    break
                if name in vt and vt.startswith("cr"):
                    matched_vt = vt
                    break

        if matched_vt and matched_vt in voice_types:
            mapping[name] = matched_vt

    return mapping


def test_resolve_speaker_voice_map():
    voice_types = {
        "femalecommander": ["/fake/a.wav"],
        "femaleuniqueastrid": ["/fake/b.wav"],
        "maleuniqueancano": ["/fake/c.wav"],
        "cruniquealduin": ["/fake/d.wav"],
        "dlc1seranavoice": ["/fake/e.wav"],
        "dlc2maleuniqueadril": ["/fake/f.wav"],
    }
    speakers_dir = "/nonexistent"  # will return empty set
    mapping = resolve_speaker_voice_map(speakers_dir, voice_types)
    assert mapping == {}

    # Mock with a temp dir containing speaker files
    import tempfile
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


if __name__ == "__main__":
    test_resolve_speaker_voice_map()
    print("Mapping test passed")
