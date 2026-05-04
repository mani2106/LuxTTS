import numpy as np

from utilities.vocalization.distinctness_check import check_vocalization_distinctness


def _make_speech_like(sr=48000, duration=1.0):
    """Generate speech-like audio: fundamental + harmonics around 200-600Hz."""
    t = np.linspace(0, duration, int(sr * duration))
    return (0.3 * np.sin(2 * np.pi * 200 * t) + 0.15 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)


def _make_whisper_like(sr=48000, duration=1.0):
    """Generate whisper-like audio: band-limited noise in 600-4000Hz."""
    n = int(sr * duration)
    noise = np.random.randn(n).astype(np.float32) * 0.1
    b, a = __import__('scipy').signal.butter(4, [600 / (sr / 2), 4000 / (sr / 2)], btype='band')
    return __import__('scipy').signal.filtfilt(b, a, noise).astype(np.float32)


def test_distinct_vocalization_passes():
    """A whisper should be distinct enough from speech baseline."""
    speech = _make_speech_like()
    vocalization = _make_whisper_like()

    result = check_vocalization_distinctness(vocalization, speech, sr=48000)

    assert result['is_distinct'] is True


def test_identical_to_speech_fails():
    """Audio identical to speech should fail distinctness check."""
    speech = _make_speech_like()

    result = check_vocalization_distinctness(speech, speech, sr=48000)

    assert result['is_distinct'] is False


def test_similar_to_speech_fails():
    """Audio very similar to speech should fail."""
    speech = _make_speech_like()
    # Add tiny difference - use small noise to keep it spectrally similar
    np.random.seed(42)
    similar = speech + 0.00005 * np.random.randn(len(speech)).astype(np.float32)

    result = check_vocalization_distinctness(similar, speech, sr=48000)

    assert result['is_distinct'] is False


def test_silence_handled():
    """Silent vocalization should not crash."""
    speech = _make_speech_like()
    silence = np.zeros(48000, dtype=np.float32)

    result = check_vocalization_distinctness(silence, speech, sr=48000)

    assert 'is_distinct' in result
    assert 'centroid_diff_hz' in result
    assert 'energy_ratio_db' in result
