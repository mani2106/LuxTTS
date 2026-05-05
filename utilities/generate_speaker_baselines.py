"""Generate per-speaker baseline JSONs from audio samples.

Usage:
    python -m utilities.generate_speaker_baselines --speakers-dir speakers/en --output-dir baselines
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utilities.audio_utils import load_wav_file
from utilities.post_processor import analyze_signal


def generate_baseline(audio_path: Path) -> dict:
    """Analyze a speaker reference file and produce baseline stats."""
    audio, sr = load_wav_file(str(audio_path))
    profile = analyze_signal(audio, sr)
    return {
        'speaker': audio_path.stem,
        'mean_lufs': round(-20 * np.log10(profile.rms + 1e-10) - 4.0, 1),  # RMS-to-LUFS approx
        'spectral_centroid_hz': round(profile.spectral_centroid, 1),
        'peak': round(profile.peak, 4),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate per-speaker baseline stats')
    parser.add_argument('--speakers-dir', type=Path, default=Path('speakers/en'),
                        help='Directory containing speaker reference WAV files')
    parser.add_argument('--output-dir', type=Path, default=Path('baselines'),
                        help='Output directory for baseline JSON files')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for wav_file in sorted(args.speakers_dir.glob('*.wav')):
        baseline = generate_baseline(wav_file)
        out_path = args.output_dir / f"{wav_file.stem}_baseline.json"
        out_path.write_text(json.dumps(baseline, indent=2))
        print(f"  {wav_file.stem}: LUFS={baseline['mean_lufs']}, "
              f"centroid={baseline['spectral_centroid_hz']}Hz, peak={baseline['peak']}")

    print(f"\nGenerated {len(list(args.output_dir.glob('*.json')))} baselines in {args.output_dir}/")


if __name__ == '__main__':
    main()
