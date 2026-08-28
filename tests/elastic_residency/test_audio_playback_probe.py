import wave

import numpy as np
import pytest

from experiments.elastic_residency.audio_playback_probe import (
    _choose_output_device,
    _load_pcm16_wav,
    _negotiate_output_format,
    probe_wav,
)


class _FakeSoundDevice:
    default = type("Default", (), {"device": (1, 2)})()

    @staticmethod
    def query_devices():
        return [
            {"name": "input", "max_output_channels": 0, "hostapi": 0},
            {"name": "mme", "max_output_channels": 2, "hostapi": 0},
            {"name": "wasapi", "max_output_channels": 2, "hostapi": 1},
        ]

    @staticmethod
    def query_hostapis():
        return [
            {"name": "MME", "default_output_device": 1},
            {"name": "Windows WASAPI", "default_output_device": 2},
        ]

    @staticmethod
    def check_output_settings(*, channels, samplerate, **_kwargs):
        if channels != 2 or samplerate != 48000:
            raise RuntimeError("unsupported")


def test_choose_output_device_prefers_wasapi() -> None:
    device_id, device, host_api = _choose_output_device(
        _FakeSoundDevice,
        None,
        prefer_wasapi=True,
    )
    assert device_id == 2
    assert device["name"] == "wasapi"
    assert host_api == "Windows WASAPI"


def test_pcm16_wav_loader_preserves_shape(tmp_path) -> None:
    wav_path = tmp_path / "silence.wav"
    samples = np.zeros((240, 1), dtype="<i2")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(samples.tobytes())

    audio, sample_rate, channels = _load_pcm16_wav(wav_path)
    assert audio.shape == (240, 1)
    assert sample_rate == 24000
    assert channels == 1


def test_output_format_negotiation_resamples_and_expands_mono() -> None:
    audio = np.zeros((240, 1), dtype=np.float32)
    converted, sample_rate, channels = _negotiate_output_format(
        _FakeSoundDevice,
        audio,
        24000,
        1,
        2,
        {
            "default_samplerate": 48000,
            "max_output_channels": 2,
        },
    )
    assert converted.shape == (480, 2)
    assert sample_rate == 48000
    assert channels == 2


def test_probe_rejects_missing_wav(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_wav(tmp_path / "missing.wav", backend="native")
