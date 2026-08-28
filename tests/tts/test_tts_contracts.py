from pathlib import Path

import pytest

from data.adapters.qwen3_tts_adapter import Qwen3TTSAdapter
from domain.models.tts import TTSChunkResult, VoiceCloneProfile
from services.tts.worker_protocol import TTSWorkerConfig


def test_icl_profile_requires_accurate_reference_text() -> None:
    with pytest.raises(ValueError, match="ref_text"):
        VoiceCloneProfile(
            profile_id="calm",
            ref_audio_path="ref.wav",
            ref_text="",
            language="Chinese",
        )


def test_x_vector_profile_is_explicit_ablation() -> None:
    profile = VoiceCloneProfile(
        profile_id="xvector-ablation",
        ref_audio_path="ref.wav",
        ref_text="",
        language="Chinese",
        x_vector_only_mode=True,
    )
    assert profile.x_vector_only_mode is True


def test_profile_audio_hash_and_result_record_excludes_bytes(tmp_path: Path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"reference")
    profile = VoiceCloneProfile(
        profile_id="calm",
        ref_audio_path=str(audio),
        ref_text="准确转写",
        language="Chinese",
    ).validate_audio()
    assert len(profile.audio_sha256) == 64

    record = TTSChunkResult(
        request_id="r",
        request_epoch=0,
        chunk_id=0,
        raw_text="你好",
        synthesis_text="你好",
        audio_bytes=b"audio",
    ).to_record()
    assert "audio_bytes" not in record


def test_production_adapter_rejects_fake_backend(tmp_path: Path) -> None:
    profile = VoiceCloneProfile(
        profile_id="fake",
        ref_audio_path="unused.wav",
        ref_text="",
        language="Chinese",
        x_vector_only_mode=True,
    )
    config = TTSWorkerConfig(
        model_path="unused",
        device="cpu",
        voice_profiles=(profile,),
        output_dir=str(tmp_path),
        backend_kind="test_fake_tts",
    )
    with pytest.raises(ValueError, match="real_qwen3_tts"):
        Qwen3TTSAdapter(config)
