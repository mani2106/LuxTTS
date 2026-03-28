# Speaker Preset Voices

Place WAV audio files in the `en/` subdirectory for use as preset voices.

Common preset names for Skyrim characters:
- malecommoner.wav
- femalecommoner.wav
- malebrute.wav
- femaleargonian.wav
- etc.

Audio files should be:
- WAV format
- Mono or stereo (will be converted to mono)
- Any sample rate (will be resampled by LuxTTS)
- 3-10 seconds recommended for voice cloning

Place your own voice samples here for use as preset voices.

## Voice Lookup

The standalone server (`skyrimnet_api.py`) indexes all WAV files recursively on startup
for O(1) name-based lookups. Voice files uploaded at runtime via the API are also
indexed immediately. Lookup matches by stem (e.g. `malecommoner`) and full filename
(e.g. `malecommoner.wav`), case-insensitive.
