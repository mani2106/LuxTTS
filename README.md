# LuxTTS
<p align="center">
  <a href="https://huggingface.co/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing">
    <img src="https://img.shields.io/badge/Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white" alt="Colab Notebook">
  </a>
</p>

A lightweight zipvoice-based text-to-speech model with voice cloning, 48kHz output, and speeds exceeding 150x realtime. This fork focuses on **SkyrimNet GamePlugin integration** with a standalone FastAPI server, preset voice management, and DSP post-processing.

---

## SkyrimNet GamePlugin Integration

To use LuxTTS as a backend for SkyrimNet GamePlugin:

### Installation

```bash
# Install LuxTTS
uv pip install -r requirements.txt

# Install server dependencies
uv pip install -r requirements-server.txt
```

### Running the Server

**Option 1: Gradio server (with Web UI)**
```bash
python SkyrimNet-LuxTTS.py --server 0.0.0.0 --port 7860
```

**Option 2: Standalone FastAPI server (recommended for SkyrimNet)**
```bash
python skyrimnet_api.py --server 0.0.0.0 --port 7860
```

The standalone server has lower overhead, supports voice file uploads from SkyrimNet, and implements the full Gradio API protocol without the Gradio dependency.

### SkyrimNet Configuration

In the SkyrimNet Web UI:
1. Go to "Test & Easy Setup"
2. Select TTS Backend: Custom/LuxTTS
3. Enter TTS Server URL: `http://localhost:7860`
4. Click "Test Connection" — should return silence (ping response)

### Preset Voices

Place voice sample WAV files in `speakers/en/` directory. The server will
pre-encode these on startup for fast access.

### Building Windows Executable

```bash
python build_exe.py
```

The executable will be in `dist/SkyrimNet-LuxTTS.exe`. Include the `speakers/`
directory when distributing.

## Audio Post-Processing

LuxTTS includes a composable DSP post-processing pipeline to improve output quality:

- **De-esser**: Reduces harsh sibilance ("s", "sh" sounds)
- **EQ**: Tames metallic high-frequency artifacts, adds vocal warmth
- **Compressor**: Evens out volume inconsistencies with soft-knee dynamics
- **Pitch Shift**: Adjusts voice pitch (manual or auto-detected from dialogue text)
- **Loudness Normalization**: Ensures consistent output levels

### Usage

The post-processing pipeline is enabled by default in both the Gradio UI and SkyrimNet API.

#### Automatic Pitch Detection

Pitch is automatically detected from dialogue text using these heuristics:
- `ALL CAPS` → +2 semitones (shouting)
- Ends with `!` → +1 semitone (excited)
- Ends with `?` → +0.5 semitone (questioning)
- Contains `...` → -1 semitone (hesitant)
- Long text (>200 chars) → -0.5 semitone (narrative)
- Manual override available via `pitch_shift` parameter

#### Optional Dependencies

For higher-quality DSP processing, install optional dependencies:

```bash
uv pip install pedalboard pyloudnorm
```

- `pedalboard`: Provides high-quality EQ, compressor, and de-esser implementations
- `pyloudnorm`: Provides accurate ITU-R BS.1770 LUFS loudness measurement

Without these, the pipeline falls back to `scipy`/`librosa` implementations (fully functional).

### API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_post_processing` | bool | True | Master toggle for the chain |
| `pitch_shift` | float | None | Manual semitone override (None = auto) |
| `eq_intensity` | float | 1.0 | EQ aggressiveness (0-1) |
| `de_ess_intensity` | float | 0.5 | De-essing strength (0-1) |
| `compressor_threshold_offset` | float | -6.0 | Threshold offset from signal RMS (dB) |
| `compressor_ratio` | float | 4.0 | Compression ratio |
| `target_loudness` | float | -16.0 | Target LUFS for normalization |

### A/B Preview

The Gradio UI (`SkyrimNet-LuxTTS.py`) provides an A/B preview feature:
- **Processed Output**: Audio with full post-processing chain
- **Raw Output**: Unprocessed model output (for comparison)

Enable via "Save Raw (A/B Preview)" checkbox in the UI.

<details>
<summary><strong>Vocalization Audio Tags</strong></summary>

The server supports ElevenLabs-style `[bracket]` audio tags for non-speech vocalizations embedded within dialogue text. This enables expressive speech output with sighs, gasps, whispers, and more — all generated from the same voice used for speech.

### Supported Tags (14 total)

| Category | Tags |
|----------|------|
| **Human Reactions** | `[sighs]`, `[groans]`, `[moans]` |
| **Emotional** | `[gasps]`, `[screams]`, `[shouts]`, `[laughs]`, `[sobs]`, `[whimpers]` |
| **Breath/Air** | `[breathes heavily]`, `[clears throat]`, `[coughs]` |
| **Delivery** | `[whispers]` (modifies next speech segment), `[pause]` (0.3s silence) |

### Usage Examples

```
[sighs] I can't believe we made it.
Stop! [gasps] How did you find me?
[whispers] Don't make a sound.
[screams] Get away from me!
[breathes heavily] We need to keep moving.
Hello [pause] my friend.
```

Just include tags in your text input — the server handles parsing, generation, and crossfade stitching automatically.

<details>
<summary>How It Works</summary>

1. **Tag Parser** — Scans text for `[bracket]` tags, splits into speech/vocalization segments
2. **Vocalization Generator** — For each vocalization tag, generates TTS audio from a phonetic template, then applies a DSP effect chain (pitch shift, filters, breath noise, etc.)
3. **Whisper Mode** — `[whispers]` applies DSP effects to the following speech segment instead of generating standalone audio
4. **Stitcher** — Crossfades all segments together with 50ms overlaps

</details>

<details>
<summary>Performance</summary>

- **Zero overhead** when no tags are present (single regex scan)
- **With tags**: ~100–300ms extra DSP processing per vocalization segment
- All DSP uses numpy/scipy — no GPU required

</details>

<details>
<summary>Extending with Custom Tags</summary>

Tag recipes are defined in `utilities/vocalization/recipes.json`. Each recipe maps a tag name to TTS text, speed, max duration, and a DSP effect chain. Add new tags by editing this file — no code changes required.

</details>

</details>

---

<details>
<summary><strong>LuxTTS — Model Usage & API</strong></summary>

### The main features are
- Voice cloning: SOTA voice cloning on par with models 10x larger.
- Clarity: Clear 48khz speech generation unlike most TTS models which are limited to 24khz.
- Speed: Reaches speeds of 150x realtime on a single GPU and faster then realtime on CPU's as well.
- Efficiency: Fits within 1gb vram meaning it can fit in any local gpu.

https://github.com/user-attachments/assets/a3b57152-8d97-43ce-bd99-26dc9a145c29

## Usage
You can try it locally, colab, or spaces.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cDaxtbSDLRmu6tRV_781Of_GSjHSo1Cu?usp=sharing)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YatharthS/LuxTTS)

#### Simple installation:
```
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt
```

#### Load model:
```python
from zipvoice.luxvoice import LuxTTS

# load model on GPU
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')

# load model on CPU
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=2)

# load model on MPS for macs
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')
```

#### Simple inference
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, rms=0.01)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```

#### Inference with sampling params:
```python
import soundfile as sf
from IPython.display import Audio

text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

rms = 0.01 ## higher makes it sound louder(0.01 or so recommended)
t_shift = 0.9 ## sampling param, higher can sound better but worse WER
num_steps = 4 ## sampling param, higher sounds better but takes longer(3-4 is best for efficiency)
speed = 1.0 ## sampling param, controls speed of audio(lower=slower)
return_smooth = False ## sampling param, makes it sound smoother possibly but less cleaner
ref_duration = 5 ## Setting it lower can speedup inference, set to 1000 if you find artifacts.

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, duration=ref_duration, rms=rms)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=num_steps, t_shift=t_shift, speed=speed, return_smooth=return_smooth)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech
if display is not None:
  display(Audio(final_wav, rate=48000))
```
## Tips
- Please use at minimum a 3 second audio file for voice cloning.
- You can use return_smooth = True if you hear metallic sounds.
- Lower t_shift for less possible pronunciation errors but worse quality and vice versa.

</details>

<details>
<summary><strong>Info, Community & Acknowledgments</strong></summary>

## Info

Q: How is this different from ZipVoice?

A: LuxTTS uses the same architecture but distilled to 4 steps with an improved sampling technique. It also uses a custom 48khz vocoder instead of the default 24khz version.

Q: Can it be even faster?

A: Yes, currently it uses float32. Float16 should be significantly faster(almost 2x).

## Community
- [Lux-TTS-Gradio](https://github.com/NidAll/LuxTTS-Gradio): A gradio app to use LuxTTS.
- [OptiSpeech](https://github.com/ycharfi09/OptiClone): Clean UI app to use LuxTTS.
- [LuxTTS-Comfyui](https://github.com/DragonDiffusionbyBoyo/BoyoLuxTTS-Comfyui.git): Nodes to use LuxTTS in comfyui.

Thanks to all community contributions!

## Roadmap

- [x] Release model and code
- [x] Huggingface spaces demo
- [x] Release MPS support (thanks to @builtbybasit)
- [ ] Release LuxTTS v1.5
- [ ] Release code for float16 inference

## Acknowledgments

- [ZipVoice](https://github.com/k2-fsa/ZipVoice) for their excellent code and model.
- [Vocos](https://github.com/gemelo-ai/vocos.git) for their great vocoder.

## Final Notes

The model and code are licensed under the Apache-2.0 license. See LICENSE for details.

Stars/Likes would be appreciated, thank you.

Email: yatharthsharma350@gmail.com

</details>
