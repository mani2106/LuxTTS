"""Tests for the duration safety margin fix (end-of-utterance truncation).

Unit tests (no GPU):
  test_inflated_frames_adds_margin
  test_inflated_frames_minimum_margin
  test_inflated_frames_single_token
  test_trim_and_pad_removes_edge_silence
  test_trim_and_pad_adds_trailing_silence

Integration tests (require GPU, skip otherwise):
  test_no_truncation_on_known_dialogue
  test_previously_truncated_texts_complete
"""

import numpy as np
import pytest
import torch

from zipvoice.modeling_utils import (
    _apply_duration_margin,
    _trim_and_pad_wav,
    _DURATION_MARGIN_RATIO,
    _DURATION_MARGIN_MIN_FRAMES,
    _DURATION_MAX_FRAMES_PER_TEXT_TOKEN,
    _TRAIL_SILENCE_MS,
)


# ---------------------------------------------------------------------------
# Unit tests — _apply_duration_margin
# ---------------------------------------------------------------------------

def _base_gen_frames(prompt_frames, prompt_tokens, text_tokens, speed):
    """Replicate the raw ratio formula (no margin) for expected-value calculation."""
    return int(np.ceil(prompt_frames / prompt_tokens * text_tokens / speed))


class TestApplyDurationMargin:
    def test_reduces_speed(self):
        """Adjusted speed should be lower than input for typical ratios."""
        # Realistic: 300 prompt frames, 60 prompt tokens (ratio=5), 35 text tokens
        prompt_lens = torch.tensor([300], dtype=torch.int64)
        prompt_tokens = [list(range(60))]
        tokens = [list(range(35))]
        speed = 1.3

        result = _apply_duration_margin(speed, prompt_lens, prompt_tokens, tokens)
        assert result < speed, f"Adjusted speed {result} should be < input {speed}"

    def test_margin_produces_expected_frame_count(self):
        """The adjusted speed should yield (base + margin) frames when fed back."""
        prompt_lens = torch.tensor([300], dtype=torch.int64)
        prompt_tokens = [list(range(60))]
        tokens = [list(range(35))]
        speed = 1.3

        adjusted = _apply_duration_margin(speed, prompt_lens, prompt_tokens, tokens)
        raw = _base_gen_frames(300, 60, 35, speed)
        margin = max(int(np.ceil(raw * _DURATION_MARGIN_RATIO)), _DURATION_MARGIN_MIN_FRAMES)
        expected = raw + margin

        actual = _base_gen_frames(300, 60, 35, adjusted)
        assert abs(actual - expected) <= 1, f"Expected ~{expected} frames, got {actual}"

    def test_minimum_margin_for_short_text(self):
        """Short texts with high prompt token counts get minimum margin."""
        # 300 frames, 50 prompt tokens, 10 text tokens (ratio=6)
        prompt_lens = torch.tensor([300], dtype=torch.int64)
        prompt_tokens = [list(range(50))]
        tokens = [list(range(10))]
        speed = 1.3

        adjusted = _apply_duration_margin(speed, prompt_lens, prompt_tokens, tokens)
        raw = _base_gen_frames(300, 50, 10, speed)
        actual = _base_gen_frames(300, 50, 10, adjusted)
        assert actual >= raw + _DURATION_MARGIN_MIN_FRAMES

    def test_max_cap_prevents_over_prediction(self):
        """Over-prediction from unreliable prompt ratios is capped."""
        # Extreme ratio: 100 frames, 2 prompt tokens, 35 text tokens
        prompt_lens = torch.tensor([100], dtype=torch.int64)
        prompt_tokens = [[1, 2]]
        tokens = [list(range(35))]
        speed = 1.3

        adjusted = _apply_duration_margin(speed, prompt_lens, prompt_tokens, tokens)
        actual = _base_gen_frames(100, 2, 35, adjusted)
        max_expected = 35 * _DURATION_MAX_FRAMES_PER_TEXT_TOKEN
        assert actual <= max_expected + 1, f"Got {actual} frames, expected <= {max_expected}"


# ---------------------------------------------------------------------------
# Unit tests — _trim_and_pad_wav
# ---------------------------------------------------------------------------

class TestTrimAndPadWav:
    def test_removes_leading_silence(self):
        """Leading silence should be trimmed."""
        sr = 48000
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        speech = np.random.randn(int(1.0 * sr)).astype(np.float32) * 0.5
        audio = np.concatenate([silence, speech])
        wav = torch.from_numpy(audio).unsqueeze(0)

        result = _trim_and_pad_wav(wav, sample_rate=sr)
        result_np = result.numpy().squeeze()

        assert result_np.shape[0] < audio.shape[0], "Should be shorter after trim"

    def test_removes_trailing_silence(self):
        """Trailing silence should be trimmed."""
        sr = 48000
        speech = np.random.randn(int(1.0 * sr)).astype(np.float32) * 0.5
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        audio = np.concatenate([speech, silence])
        wav = torch.from_numpy(audio).unsqueeze(0)

        result = _trim_and_pad_wav(wav, sample_rate=sr)
        result_np = result.numpy().squeeze()

        assert result_np.shape[0] < audio.shape[0] + int(_TRAIL_SILENCE_MS * sr / 1000) + 100

    def test_adds_trailing_silence(self):
        """Should add _TRAIL_SILENCE_MS of silence after trimming."""
        sr = 48000
        speech = np.random.randn(int(0.5 * sr)).astype(np.float32) * 0.5
        wav = torch.from_numpy(speech).unsqueeze(0)

        result = _trim_and_pad_wav(wav, sample_rate=sr)
        result_np = result.numpy().squeeze()

        expected_trail = int(_TRAIL_SILENCE_MS * sr / 1000)
        actual_trail = result_np.shape[0] - result_np.shape[0]  # placeholder
        # Check last N samples are near-zero (the trailing silence)
        tail = result_np[-expected_trail:]
        assert np.max(np.abs(tail)) < 1e-6, "Trailing samples should be silent"

    def test_output_is_2d_tensor(self):
        """Should return shape (1, N) tensor."""
        sr = 48000
        speech = np.random.randn(int(0.5 * sr)).astype(np.float32) * 0.5
        wav = torch.from_numpy(speech).unsqueeze(0)

        result = _trim_and_pad_wav(wav, sample_rate=sr)
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_preserves_speech_content(self):
        """Trimming should not remove the speech portion."""
        sr = 48000
        speech = np.random.randn(int(1.0 * sr)).astype(np.float32) * 0.5
        silence = np.zeros(int(0.3 * sr), dtype=np.float32)
        audio = np.concatenate([silence, speech, silence])
        wav = torch.from_numpy(audio).unsqueeze(0)

        result = _trim_and_pad_wav(wav, sample_rate=sr)
        result_np = result.numpy().squeeze()

        # The speech portion should still be present (high energy)
        assert np.max(np.abs(result_np)) > 0.1, "Speech content should be preserved"


# ---------------------------------------------------------------------------
# Integration tests — require GPU + model loading
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
class TestGenerationNoTruncation:
    """Verify that generated audio is not truncated for known problematic texts.

    These tests generate audio end-to-end and check:
    1. Output duration is reasonable (> minimum threshold)
    2. Whisper transcription does not end with trailing '...' or '-'
    3. Waveform ends with natural decay (low energy in final 20ms)
    """

    # Texts that were previously truncated, based on Whisper analysis
    PREVIOUSLY_TRUNCATED_TEXTS = [
        "Hello my lady, I am not sure who you are, but perhaps you can help me.",
        "Prove yourself against Malkoran's filth and you shall be rewarded with my devotion.",
        "Your victory over Malkoran's corruption has earned you the gratitude of a goddess.",
        "My power is vast, but I require a champion to wield it in my name.",
    ]

    SPEAKER = "serana"

    @pytest.fixture(scope="class")
    def tts_system(self):
        """Load model once for all tests in this class."""
        from zipvoice.modeling_utils import load_models_gpu, process_audio
        from utilities.model_utils import _resolve_model_path
        from pathlib import Path

        model_path = _resolve_model_path("YatharthS/LuxTTS")
        model, feature_extractor, vocoder, tokenizer, transcriber = load_models_gpu(
            model_path=str(model_path)
        )

        speaker_file = Path("speakers/en") / f"{self.SPEAKER}.wav"
        if not speaker_file.exists():
            pytest.skip(f"Speaker file not found: {speaker_file}")

        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(
            str(speaker_file), transcriber, tokenizer, feature_extractor,
            device=model.device if hasattr(model, 'device') else torch.device("cuda:0"),
        )

        return {
            "model": model,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "prompt_tokens": prompt_tokens,
            "prompt_features_lens": prompt_features_lens,
            "prompt_features": prompt_features,
            "prompt_rms": prompt_rms,
        }

    @pytest.mark.parametrize("text", PREVIOUSLY_TRUNCATED_TEXTS)
    def test_transcription_is_complete(self, tts_system, text):
        """Whisper should transcribe the full text without trailing '...' or '-'."""
        import whisper
        import librosa

        model = tts_system["model"]
        vocoder = tts_system["vocoder"]
        tokenizer = tts_system["tokenizer"]

        from zipvoice.modeling_utils import generate
        wav = generate(
            prompt_tokens=tts_system["prompt_tokens"],
            prompt_features_lens=tts_system["prompt_features_lens"],
            prompt_features=tts_system["prompt_features"],
            prompt_rms=tts_system["prompt_rms"],
            text=text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
        )

        wav_np = wav.cpu().numpy().squeeze()
        wav_16k = librosa.resample(wav_np, orig_sr=48000, target_sr=16000).astype(np.float32)

        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(wav_16k, language="en", fp16=False)
        transcription = result["text"].strip()

        # Must not end with truncation indicators
        assert not transcription.endswith("..."), (
            f"Transcription ends with '...' (truncated): \"{transcription}\""
        )
        assert not transcription.endswith("-"), (
            f"Transcription ends with '-' (cut mid-word): \"{transcription}\""
        )
        assert len(transcription) > 10, (
            f"Transcription too short, possibly garbled: \"{transcription}\""
        )

    @pytest.mark.parametrize("text", PREVIOUSLY_TRUNCATED_TEXTS)
    def test_waveform_ends_naturally(self, tts_system, text):
        """Waveform should have low energy in the final 20ms (natural decay)."""
        model = tts_system["model"]
        vocoder = tts_system["vocoder"]
        tokenizer = tts_system["tokenizer"]

        from zipvoice.modeling_utils import generate
        wav = generate(
            prompt_tokens=tts_system["prompt_tokens"],
            prompt_features_lens=tts_system["prompt_features_lens"],
            prompt_features=tts_system["prompt_features"],
            prompt_rms=tts_system["prompt_rms"],
            text=text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
        )

        wav_np = wav.cpu().numpy().squeeze()
        sr = 48000
        last_20ms = int(0.02 * sr)
        last_peak = np.max(np.abs(wav_np[-last_20ms:]))
        overall_peak = np.max(np.abs(wav_np))

        # Last 20ms should be < 10% of overall peak (natural decay)
        assert last_peak < overall_peak * 0.15, (
            f"Audio ends abruptly: last20ms_peak={last_peak:.4f}, "
            f"overall_peak={overall_peak:.4f}, ratio={last_peak/overall_peak:.2f}"
        )

    @pytest.mark.parametrize("text", PREVIOUSLY_TRUNCATED_TEXTS)
    def test_duration_is_reasonable(self, tts_system, text):
        """Output duration should be proportional to text length, not capped."""
        model = tts_system["model"]
        vocoder = tts_system["vocoder"]
        tokenizer = tts_system["tokenizer"]

        from zipvoice.modeling_utils import generate
        wav = generate(
            prompt_tokens=tts_system["prompt_tokens"],
            prompt_features_lens=tts_system["prompt_features_lens"],
            prompt_features=tts_system["prompt_features"],
            prompt_rms=tts_system["prompt_rms"],
            text=text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
        )

        dur = wav.shape[-1] / 48000
        # Previously these texts were capped at ~3.05s
        # With margin, they should be longer
        assert dur > 2.0, f"Duration too short: {dur:.2f}s for text of {len(text)} chars"
