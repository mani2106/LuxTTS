"""Manual verification test for vocalization tags.

This script generates sample audio with various tags for manual listening test.
Run with: python tests/manual_vocalization_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.audio_generation_pipeline import generate_audio
from utilities.app_config import AppConfig


async def main():
    """Generate test audio files."""
    config = AppConfig()

    test_cases = [
        ("[sighs] I can't believe we made it.", "test_sighs"),
        ("[gasps] Who's there?", "test_gasps"),
        ("[screams] Get away from me!", "test_screams"),
        ("[whispers] Don't make a sound.", "test_whispers"),
        ("Hello [pause] my friend.", "test_pause"),
        ("[breathes heavily] We need to keep moving.", "test_heavy_breath"),
        ("Normal speech without tags.", "test_normal"),
        ("[groans] My head... where am I?", "test_groans"),
        ("[laughs] That's hilarious!", "test_laughs"),
        ("[clears throat] Ahem, attention please.", "test_clears_throat"),
    ]

    # Create output directory
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)

    print(f"Generating {len(test_cases)} test audio files...")
    print(f"Output directory: {output_dir}\n")

    for text, filename in test_cases:
        print(f"Generating: {text}")
        try:
            output_path, seed = await generate_audio(
                text=text,
                config=config,
                enable_post_processing=True,
            )

            # Copy to test output with descriptive name
            src = Path(output_path)
            dest = output_dir / f"{filename}.wav"
            if src.exists():
                src.rename(dest)
                print(f"  -> Saved to {dest}")
            else:
                print(f"  -> ERROR: Output file not found")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    print(f"\nAll test files generated in {output_dir}/")
    print("Listen to verify vocalization quality.")


if __name__ == "__main__":
    asyncio.run(main())
