import json
from typing import Optional

import numpy as np
import torch
import librosa
from transformers import pipeline
from huggingface_hub import snapshot_download

from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm, chunk_tokens_punctuation, cross_fade_concat

from dataclasses import dataclass

from linacodec.vocoder.vocos import Vocos
from zipvoice.onnx_modeling import OnnxModel
from torch.nn.utils import parametrize

# Duration prediction can underestimate actual speech length, causing
# end-of-utterance truncation. This safety margin slows the internal speed
# prediction so the model allocates more mel frames; silence is trimmed after vocoding.
_DURATION_MARGIN_RATIO = 0.20   # 20% extra frames over predicted length
_DURATION_MARGIN_MIN_FRAMES = 30  # at least ~320ms of headroom
_DURATION_MAX_FRAMES_PER_TEXT_TOKEN = 10  # cap over-prediction from unreliable ratios
_TAIL_DECAY_FRAMES = 5           # linear fade-out at boundary (mel frames)
_TRIM_TOP_DB = 30                # librosa.effects.trim sensitivity
_TRAIL_SILENCE_MS = 200          # silence appended after trim


def _sanitize_transcription(text):
    """Collapse Whisper hallucination patterns from prompt transcription.

    Whisper produces long runs of repeated characters for non-speech audio
    (e.g. dragon roars -> "SUUUUUUUU..."). These explode the prompt token
    count and collapse the frames-per-token ratio used for duration prediction.
    """
    import re
    cleaned = re.sub(r'(.)\1{2,}', r'\1\1', text)
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    if len(cleaned.strip()) < 5:
        cleaned = " speech"
    return cleaned


def _apply_duration_margin(speed, prompt_features_lens, prompt_tokens, tokens):
    """Reduce effective speed so ratio-based duration prediction allocates extra frames.

    The model predicts gen_frames = ceil(prompt_frames / prompt_tokens * text_tokens / speed).
    Lowering speed increases predicted frames. We compute how much to slow down
    so the model gets the desired margin, keeping the duration='predict' path intact.
    """
    device = prompt_features_lens.device
    prompt_tokens_lens = torch.tensor(
        [len(t) for t in prompt_tokens], dtype=torch.int64, device=device
    )
    tokens_lens = torch.tensor(
        [len(t) for t in tokens], dtype=torch.int64, device=device
    )
    # Base predicted frames (matching forward_text_inference_ratio_duration)
    gen_frames = torch.ceil(
        (prompt_features_lens / prompt_tokens_lens * tokens_lens).float() / speed
    ).to(dtype=torch.int64)
    margin = torch.max(
        torch.ceil(gen_frames.float() * _DURATION_MARGIN_RATIO).to(dtype=torch.int64),
        torch.tensor(_DURATION_MARGIN_MIN_FRAMES, dtype=torch.int64, device=device),
    )
    # Adjusted speed: same formula solved for speed with (gen_frames + margin)
    target_frames = gen_frames.float() + margin.float()
    # Cap over-prediction from unreliable prompt ratios (e.g. after Whisper hallucination cleanup)
    max_frames = tokens_lens.float() * _DURATION_MAX_FRAMES_PER_TEXT_TOKEN
    target_frames = torch.min(target_frames, max_frames)
    adjusted_speed = (prompt_features_lens.float() / prompt_tokens_lens.float()
                      * tokens_lens.float() / target_frames)
    return adjusted_speed.item()


@dataclass
class LuxTTSConfig:
    # Model Setup
    model_dir: Optional[str] = None
    checkpoint_name: str = "model.pt"
    vocoder_path: Optional[str] = None
    trt_engine_path: Optional[str] = None

    # Tokenizer & Language
    tokenizer: str = "emilia"  # choices: ["emilia", "libritts", "espeak", "simple"]
    lang: str = "en-us"


@torch.inference_mode
def process_audio(audio, transcriber, tokenizer, feature_extractor, device, target_rms=0.1, duration=4, feat_scale=0.1):
    prompt_wav, sr = librosa.load(audio, sr=24000, duration=duration)
    prompt_wav2, sr = librosa.load(audio, sr=16000, duration=duration)

    # Transcribe BEFORE trimming so Whisper sees full, unaltered audio
    prompt_text = transcriber(prompt_wav2)["text"]
    prompt_text = _sanitize_transcription(prompt_text)
    print(prompt_text)

    # Trim silence from feature extraction audio only to prevent prompt leaking
    prompt_wav, _ = librosa.effects.trim(prompt_wav, top_db=30)

    # Add 200ms trailing silence to seal prompt boundary
    trail_samples = int(0.2 * sr)
    prompt_wav = np.append(prompt_wav, np.zeros(trail_samples, dtype=np.float32))

    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=24000
    ).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    return prompt_tokens, prompt_features_lens, prompt_features, prompt_rms

def _compute_inflated_gen_frames(prompt_features_lens, prompt_tokens, tokens, speed):
    """Compute generation frame count with safety margin to prevent truncation.

    Replicates the ratio-based duration prediction from
    ZipVoice.forward_text_inference_ratio_duration and adds a margin so the
    model has room to finish the utterance naturally.
    """
    device = prompt_features_lens.device
    prompt_tokens_lens = torch.tensor(
        [len(t) for t in prompt_tokens], dtype=torch.int64, device=device
    )
    tokens_lens = torch.tensor(
        [len(t) for t in tokens], dtype=torch.int64, device=device
    )
    gen_frames = torch.ceil(
        prompt_features_lens.float() / prompt_tokens_lens.float() * tokens_lens.float() / speed
    ).to(dtype=torch.int64)

    margin = torch.max(
        torch.ceil(gen_frames.float() * _DURATION_MARGIN_RATIO).to(dtype=torch.int64),
        torch.tensor(_DURATION_MARGIN_MIN_FRAMES, dtype=torch.int64, device=device),
    )
    return gen_frames + margin


def _trim_and_pad_wav(wav, sample_rate=48000):
    """Trim edge silence from vocoded audio and add trailing silence."""
    wav_np = wav.cpu().numpy().squeeze()
    wav_np, _ = librosa.effects.trim(wav_np, top_db=_TRIM_TOP_DB)
    trail = np.zeros(int(_TRAIL_SILENCE_MS * sample_rate / 1000), dtype=np.float32)
    wav_np = np.append(wav_np, trail)
    return torch.from_numpy(wav_np).unsqueeze(0)


def generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1):
    CHUNK_CHAR_THRESHOLD = 120
    if len(text) > CHUNK_CHAR_THRESHOLD:
        return _generate_chunked(
            prompt_tokens, prompt_features_lens, prompt_features, prompt_rms,
            text, model, vocoder, tokenizer,
            num_step, guidance_scale, speed, t_shift, target_rms,
            chunk_char_threshold=CHUNK_CHAR_THRESHOLD,
        )
    tokens = tokenizer.texts_to_token_ids([text])

    speed = speed * 1.3
    speed = _apply_duration_margin(speed, prompt_features_lens, prompt_tokens, tokens)

    with torch.inference_mode():
        (pred_features, _, _, pred_lens) = model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration='predict',
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

    # Convert to waveform
    pred_features = pred_features.permute(0, 2, 1) / 0.1

    # pred_features is x1_wo_prompt from sample() — already separated from prompt
    # and sized to x1_wo_prompt_lens. pred_lens is actually prompt_features_lens
    # (returned by sample()), NOT the generation length. Use the full mel output.
    actual_len = pred_features.size(2)
    last_frame = pred_features[:, :, actual_len - 1:actual_len]
    decay = torch.linspace(1.0, 0.0, _TAIL_DECAY_FRAMES).to(pred_features.device).view(1, 1, -1)
    tail = last_frame * decay
    pred_features = torch.cat([pred_features[:, :, :actual_len], tail], dim=2)

    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Volume matching
    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)

    wav = _trim_and_pad_wav(wav)
    return wav

def _generate_chunked(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1, chunk_char_threshold=120):
    """Generate speech for longer texts by chunking at punctuation boundaries."""
    speed_internal = speed * 1.3

    # Tokenize to string tokens for chunking
    tokens_str = tokenizer.texts_to_tokens([text])[0]

    # Estimate max_tokens per chunk targeting ~25s total (prompt + generated)
    prompt_duration_s = prompt_features.size(1) * 0.01
    token_duration_s = prompt_duration_s / max(len(tokens_str), 1) / speed
    max_tokens = int(max((25 - prompt_duration_s) / max(token_duration_s, 0.01), 20))
    max_tokens = min(max_tokens, 150)

    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

    if len(chunked_tokens_str) <= 1:
        # Cannot split further — inline single-pass generation to avoid
        # re-entering generate()'s len(text) > threshold gate (infinite recursion)
        chunk_token_ids = tokenizer.texts_to_token_ids([text])
        chunk_speed = _apply_duration_margin(
            speed_internal, prompt_features_lens, prompt_tokens, chunk_token_ids
        )
        with torch.inference_mode():
            (pred_features, _, _, pred_lens) = model.sample(
                tokens=chunk_token_ids,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed=chunk_speed,
                t_shift=t_shift,
                duration='predict',
                num_step=num_step,
                guidance_scale=guidance_scale,
            )
        pred_features = pred_features.permute(0, 2, 1) / 0.1
        actual_len = pred_features.size(2)
        last_frame = pred_features[:, :, actual_len - 1:actual_len]
        decay = torch.linspace(1.0, 0.0, _TAIL_DECAY_FRAMES).to(pred_features.device).view(1, 1, -1)
        tail = last_frame * decay
        pred_features = torch.cat([pred_features[:, :, :actual_len], tail], dim=2)
        wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
        if prompt_rms < target_rms:
            wav = wav * (prompt_rms / target_rms)
        wav = _trim_and_pad_wav(wav)
        return wav

    # Generate each chunk
    chunk_wavs = []
    with torch.inference_mode():
        for chunk_str_tokens in chunked_tokens_str:
            chunk_token_ids = tokenizer.tokens_to_token_ids([chunk_str_tokens])

            chunk_speed = _apply_duration_margin(
                speed_internal, prompt_features_lens, prompt_tokens, chunk_token_ids
            )

            (pred_features, _, _, pred_lens) = model.sample(
                tokens=chunk_token_ids,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                speed=chunk_speed,
                t_shift=t_shift,
                duration='predict',
                num_step=num_step,
                guidance_scale=guidance_scale,
            )

            pred_features = pred_features.permute(0, 2, 1) / 0.1
            actual_len = pred_features.size(2)
            last_frame = pred_features[:, :, actual_len - 1:actual_len]
            decay = torch.linspace(1.0, 0.0, _TAIL_DECAY_FRAMES).to(pred_features.device).view(1, 1, -1)
            tail = last_frame * decay
            pred_features = torch.cat([pred_features[:, :, :actual_len], tail], dim=2)

            wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
            if prompt_rms < target_rms:
                wav = wav * (prompt_rms / target_rms)
            wav = _trim_and_pad_wav(wav)
            chunk_wavs.append(wav)

    final_wav = cross_fade_concat(chunk_wavs, fade_duration=0.1, sample_rate=48000)
    return final_wav

def load_models_gpu(model_path=None, device="cuda"):
    params = LuxTTSConfig()
    if model_path is None:
        model_path = snapshot_download("YatharthS/LuxTTS")

    token_file = f"{model_path}/tokens.txt"
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = ZipVoiceDistill(
        **model_config["model"],
        **tokenizer_config,
    )
    load_checkpoint(filename=model_ckpt, model=model, strict=True)
    params.device = torch.device(device, 0)

    model = model.to(params.device).eval()
    feature_extractor = VocosFbank()

    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').to(device)
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=params.device))

    params.sampling_rate = model_config["feature"]["sampling_rate"]
    return model, feature_extractor, vocos, tokenizer, transcriber

def load_models_cpu(model_path = None, num_thread=2):
    params = LuxTTSConfig()
    params.seed = 42

    model_path = snapshot_download('YatharthS/LuxTTS')

    token_file = f"{model_path}/tokens.txt"
    text_encoder_path = f"{model_path}/text_encoder.onnx"
    fm_decoder_path = f"{model_path}/fm_decoder.onnx"
    model_config  = f"{model_path}/config.json"

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device='cpu')

    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=num_thread)

    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').eval()
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=torch.device('cpu')))

    feature_extractor = VocosFbank()

    params.sampling_rate = model_config["feature"]["sampling_rate"]
    params.onnx_int8 = True
    return model, feature_extractor, vocos, tokenizer, transcriber
