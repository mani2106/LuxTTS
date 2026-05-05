#!/usr/bin/env python
"""Build composite speaker samples from extracted Skyrim voice clips."""

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm







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


def score_clip(wav_path, frame_length=2048, hop_length=512):
    """Score a single WAV clip on energy, silence ratio, and duration.

    Returns (score, duration, rms_db) tuple. Returns None if clip is unusable.
    """
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception:
        return None

    duration = len(y) / sr
    if duration < 0.5:
        return None

    # RMS energy in dB
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return None
    rms_db = 20 * np.log10(rms)

    # Energy score: peak at -15dB, degrade outside -30 to -3dB range
    if rms_db < -30 or rms_db > -3:
        energy_score = 0.0
    else:
        energy_score = max(0.0, 1.0 - abs(rms_db - (-15)) / 15.0)

    # Silence ratio: proportion of frames below -40dB
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    frame_rms = np.sqrt(np.mean(S ** 2, axis=0))
    if len(frame_rms) == 0:
        return None
    silence_threshold = 10 ** (-40 / 20)
    silence_ratio = np.mean(frame_rms < silence_threshold)
    silence_score = max(0.0, 1.0 - silence_ratio)

    # Duration score: peak at 3-6s, degrade outside 1-10s
    if duration < 1 or duration > 15:
        dur_score = 0.0
    elif 2 <= duration <= 8:
        dur_score = 1.0
    else:
        dur_score = max(0.0, 1.0 - abs(duration - 5) / 5.0)

    # Weighted composite
    score = 0.3 * energy_score + 0.4 * silence_score + 0.3 * dur_score
    return (score, duration, rms_db)


def select_clips(scored_clips, target_duration, max_clips):
    """Select top-scoring clips to reach target duration.

    Args:
        scored_clips: list of (score, duration, rms_db, wav_path)
        target_duration: target composite length in seconds
        max_clips: maximum number of clips to use

    Returns list of wav_path in selection order (best first).
    """
    scored_clips.sort(key=lambda x: x[0], reverse=True)
    selected = []
    total_dur = 0.0
    for score, dur, rms_db, path in scored_clips:
        if len(selected) >= max_clips:
            break
        if total_dur >= target_duration:
            break
        selected.append(path)
        total_dur += dur
    return selected


def crossfade_concat(wav_paths, crossfade_ms=50, target_sr=44100):
    """Load, resample, and crossfade-concatenate WAV files.

    Returns numpy array and sample rate.
    """
    clips = []
    for path in wav_paths:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        clips.append(y)

    if len(clips) == 1:
        return clips[0], target_sr

    crossfade_samples = int(target_sr * crossfade_ms / 1000)
    result = clips[0]
    for clip in clips[1:]:
        if crossfade_samples > len(result) or crossfade_samples > len(clip):
            crossfade_samples = min(len(result), len(clip)) // 2
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
        result[-crossfade_samples:] = result[-crossfade_samples:] * fade_out + clip[:crossfade_samples] * fade_in
        result = np.concatenate([result, clip[crossfade_samples:]])

    return result, target_sr


def normalize_rms(audio, target_db=-20):
    """Normalize audio RMS to target dB level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    target_rms = 10 ** (target_db / 20)
    return audio * (target_rms / rms)


def build_samples(args):
    """Main pipeline: discover -> map -> score -> select -> concat -> write."""
    voice_types = discover_voice_types(args.input)
    print(f"Discovered {len(voice_types)} voice types")

    mapping = resolve_speaker_voice_map(args.speakers_dir, voice_types)
    print(f"Matched {len(mapping)} speakers to voice types")
    for speaker, vt in sorted(mapping.items()):
        print(f"  {speaker} -> {vt} ({len(voice_types[vt])} clips)")

    os.makedirs(args.output, exist_ok=True)
    rejected_dir = os.path.join(args.output, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)

    results = []

    for speaker, vt_name in tqdm(mapping.items(), desc="Building samples", unit="speaker"):
        wav_paths = voice_types[vt_name]

        # Score all clips
        scored = []
        for wp in tqdm(wav_paths, desc=f"  Scoring {speaker}", leave=False, unit="clip"):
            result = score_clip(wp)
            if result is not None:
                scored.append((*result, wp))

        if len(scored) < 3:
            print(f"  WARNING: {speaker} has only {len(scored)} scorable clips (need >=3), skipping")
            results.append((speaker, "skipped", f"only {len(scored)} scorable clips"))
            continue

        selected = select_clips(scored, args.target_duration, args.max_clips)
        if not selected:
            results.append((speaker, "skipped", "no clips selected"))
            continue

        composite, sr = crossfade_concat(selected)
        composite = normalize_rms(composite)

        out_path = os.path.join(args.output, f"{speaker}.wav")

        if args.dry_run:
            total_dur = sum(librosa.get_duration(path=p) for p in selected)
            print(f"  [DRY RUN] {speaker}: {len(selected)} clips, {total_dur:.1f}s -> {out_path}")
            results.append((speaker, "dry_run", f"{len(selected)} clips, {total_dur:.1f}s"))
        else:
            sf.write(out_path, composite, sr)
            dur = len(composite) / sr
            print(f"  {speaker}: {len(selected)} clips -> {dur:.1f}s -> {out_path}")
            results.append((speaker, "built", f"{dur:.1f}s"))

    print("\n=== Summary ===")
    for speaker, status, detail in results:
        print(f"  {status:10s} {speaker}: {detail}")
    print(f"\nTotal: {len(results)} speakers processed")

    return results


def validate_samples(args, results):
    """Run TTS model validation on built composites.

    Checks Whisper transcription quality: flags if empty, too few words,
    or implausibly low word rate.
    """
    built = [(speaker, detail) for speaker, status, detail in results if status == "built"]
    if not built:
        print("No samples to validate.")
        return

    print("\nLoading LuxTTS model for validation...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from zipvoice.luxvoice import LuxTTS

    tts = LuxTTS(device="cpu")
    print("Model loaded.\n")

    rejected_dir = os.path.join(args.output, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)

    validated = []
    for speaker, detail in tqdm(built, desc="Validating", unit="speaker"):
        wav_path = os.path.join(args.output, f"{speaker}.wav")
        if not os.path.exists(wav_path):
            validated.append((speaker, "missing", detail))
            continue

        try:
            wav_16k, _ = librosa.load(wav_path, sr=16000, duration=10)
            transcription = tts.transcriber(wav_16k)["text"]
        except Exception as e:
            print(f"  ERROR validating {speaker}: {e}")
            validated.append((speaker, "error", str(e)))
            import shutil
            shutil.move(wav_path, os.path.join(rejected_dir, f"{speaker}.wav"))
            continue

        words = transcription.strip().split()
        dur = librosa.get_duration(path=wav_path)
        word_rate = len(words) / dur if dur > 0 else 0

        if len(words) == 0:
            status = "rejected"
            reason = "empty transcription"
        elif len(words) < 3 and dur > 5:
            status = "rejected"
            reason = f"too few words ({len(words)}) for {dur:.1f}s clip"
        elif word_rate < 0.5 and dur > 5:
            status = "rejected"
            reason = f"low word rate ({word_rate:.1f} w/s)"
        else:
            status = "valid"
            reason = f"{len(words)} words, {word_rate:.1f} w/s"

        if status == "rejected":
            import shutil
            shutil.move(wav_path, os.path.join(rejected_dir, f"{speaker}.wav"))

        validated.append((speaker, status, reason))
        print(f"  {status:10s} {speaker}: {reason}")

    print("\n=== Validation Summary ===")
    passed = sum(1 for _, s, _ in validated if s == "valid")
    rejected = sum(1 for _, s, _ in validated if s == "rejected")
    errors = sum(1 for _, s, _ in validated if s == "error")
    print(f"  Valid: {passed}, Rejected: {rejected}, Errors: {errors}")

    return validated


def main():
    args = parse_args()
    results = build_samples(args)

    if args.skip_validate:
        print("\nValidation skipped (--skip-validate)")
        return

    validate_samples(args, results)


if __name__ == "__main__":
    main()
