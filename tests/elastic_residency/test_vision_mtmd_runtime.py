"""libmtmd 运行后端与 chunk 执行的离线契约测试（假绑定，不接触真实模型）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.elastic_residency.vision_mtmd_fakes import (  # noqa: E402
    CHUNK_AUDIO,
    CHUNK_IMAGE,
    CHUNK_TEXT,
    FakeLlama,
    FakeMtmd,
    default_chunks,
    install_fake_llama_cpp,
)

install_fake_llama_cpp()

from experiments.elastic_residency.vision.mtmd_eval import (  # noqa: E402
    evaluate_chunks,
    generate_fixed_tokens,
)
from experiments.elastic_residency.vision.mtmd_runtime import (  # noqa: E402
    MtmdSessionConfig,
    MtmdUnavailableError,
    MtmdVisionSession,
    probe_mmproj_caps,
)


@pytest.fixture()
def mmproj_file(tmp_path: Path) -> Path:
    path = tmp_path / "mmproj.gguf"
    path.write_bytes(b"GGUF-placeholder")
    return path


def _session(mmproj: Path, mtmd: FakeMtmd, *, use_gpu: bool = True) -> MtmdVisionSession:
    return MtmdVisionSession(
        MtmdSessionConfig(
            mmproj_path=mmproj,
            use_gpu=use_gpu,
            n_threads=14,
            flash_attn_enabled=True,
            image_max_tokens=256,
        ),
        mtmd=mtmd,
    )


class TestSessionLifecycle:
    def test_open_records_capabilities_and_applies_params(
        self, mmproj_file: Path
    ) -> None:
        mtmd = FakeMtmd()
        session = _session(mmproj_file, mtmd)
        evidence = session.open(FakeLlama())
        assert evidence["residency_state"] == "VISION_WARM_GPU"
        assert evidence["supports_vision"] is True
        assert evidence["uses_mrope"] is True
        assert evidence["media_marker"] == "<__media__>"
        assert evidence["projector_load_s"] >= 0.0
        assert evidence["projector_warmup_enabled"] is True
        # non-causal 是 per-chunk 属性，不得出现在 session 级证据里
        assert "uses_non_causal_encode" not in evidence
        assert "use_non_causal" not in mtmd.calls
        assert mtmd.init_params.use_gpu is True
        assert mtmd.init_params.n_threads == 14
        assert mtmd.init_params.image_max_tokens == 256
        assert mtmd.init_params.flash_attn_type == 1
        release = session.close()
        assert release["projector_freed"] is True
        assert mtmd.ctx_freed is True

    def test_cpu_projector_state_name(self, mmproj_file: Path) -> None:
        mtmd = FakeMtmd()
        session = _session(mmproj_file, mtmd, use_gpu=False)
        assert session.open(FakeLlama())["residency_state"] == "VISION_WARM_CPU"
        assert mtmd.init_params.use_gpu is False

    def test_init_failure_and_missing_vision_are_explicit(
        self, mmproj_file: Path
    ) -> None:
        with pytest.raises(MtmdUnavailableError):
            _session(mmproj_file, FakeMtmd(init_returns_none=True)).open(FakeLlama())
        mtmd = FakeMtmd(supports_vision=False)
        with pytest.raises(MtmdUnavailableError):
            _session(mmproj_file, mtmd).open(FakeLlama())
        assert mtmd.ctx_freed is True

    def test_double_open_rejected(self, mmproj_file: Path) -> None:
        session = _session(mmproj_file, FakeMtmd())
        session.open(FakeLlama())
        with pytest.raises(MtmdUnavailableError):
            session.open(FakeLlama())

    def test_missing_mmproj_file_raises(self, tmp_path: Path) -> None:
        session = _session(tmp_path / "absent.gguf", FakeMtmd())
        with pytest.raises(FileNotFoundError):
            session.open(FakeLlama())

    def test_probe_caps_reads_declared_inputs(self, mmproj_file: Path) -> None:
        caps = probe_mmproj_caps(mmproj_file, mtmd=FakeMtmd())
        assert caps["declares_vision_input"] is True
        assert caps["declares_audio_input"] is False
        assert caps["default_media_marker"] == "<__media__>"


class TestBitmapAndTokenize:
    def test_bitmap_and_chunk_descriptors(self, mmproj_file: Path, tmp_path: Path) -> None:
        image = tmp_path / "fixed.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        mtmd = FakeMtmd()
        session = _session(mmproj_file, mtmd)
        session.open(FakeLlama())
        bitmap = session.bitmap_from_file(image)
        described = bitmap.describe()
        assert described["decoded_width_px"] == 448
        assert described["is_audio"] is False
        chunks = session.tokenize("prefix <__media__> suffix", [bitmap])
        descriptors = chunks.describe()
        assert [item["type_name"] for item in descriptors] == ["text", "image", "text"]
        assert descriptors[1]["image_tokens"]["n_tokens"] == 196
        assert chunks.text_token_ids()[:3] == [0, 1, 2]
        assert mtmd.tokenized_bitmap_count == 1
        chunks.free()
        bitmap.free()
        assert mtmd.freed_bitmaps == [1001]

    def test_bitmap_decode_failure_is_explicit(
        self, mmproj_file: Path, tmp_path: Path
    ) -> None:
        image = tmp_path / "broken.png"
        image.write_bytes(b"not-an-image")
        session = _session(mmproj_file, FakeMtmd(bitmap_available=False))
        session.open(FakeLlama())
        with pytest.raises(MtmdUnavailableError):
            session.bitmap_from_file(image)

    def test_tokenize_failure_frees_chunks(self, mmproj_file: Path) -> None:
        mtmd = FakeMtmd(tokenize_code=-3)
        session = _session(mmproj_file, mtmd)
        session.open(FakeLlama())
        with pytest.raises(MtmdUnavailableError):
            session.tokenize("text <__media__>", [])
        assert mtmd.freed_chunk_handles == [7777]

    def test_require_ctx_before_open(self, mmproj_file: Path) -> None:
        with pytest.raises(MtmdUnavailableError):
            _session(mmproj_file, FakeMtmd()).bitmap_from_file(mmproj_file)


class TestChunkExecution:
    def _prepare(self, mmproj_file: Path, mtmd: FakeMtmd):
        session = _session(mmproj_file, mtmd)
        llm = FakeLlama()
        session.open(llm)
        chunks = session.tokenize("text <__media__> tail", [])
        return llm, session, chunks

    def test_split_encode_isolates_projector_and_syncs_state(
        self, mmproj_file: Path
    ) -> None:
        mtmd = FakeMtmd()
        llm, session, chunks = self._prepare(mmproj_file, mtmd)
        evidence = evaluate_chunks(
            llm=llm, session=session, chunks=chunks, mode="split_encode"
        )
        assert evidence["mode"] == "split_encode"
        assert evidence["visual_token_total"] == 196
        assert evidence["visual_position_total"] == 196
        assert evidence["text_token_total"] == 20
        assert evidence["n_past_final"] == 216
        assert evidence["visual_embedding_available"] is True
        assert evidence["projector_encode_total_s"] is not None
        assert evidence["image_decode_total_s"] is not None
        assert "encode" in mtmd.calls and "decode_image" in mtmd.calls
        assert llm._ctx.cleared == 1
        image_records = [
            item for item in evidence["chunks"] if item["type_name"] == "image"
        ]
        assert [item["uses_non_causal_encode"] for item in image_records] == [True]
        assert mtmd.calls.count("use_non_causal") == 1

    def test_chunk_single_mode_uses_upstream_helper(self, mmproj_file: Path) -> None:
        mtmd = FakeMtmd()
        llm, session, chunks = self._prepare(mmproj_file, mtmd)
        evidence = evaluate_chunks(
            llm=llm, session=session, chunks=chunks, mode="chunk_single"
        )
        assert "eval_chunk_single" in mtmd.calls
        assert "encode" not in mtmd.calls
        assert evidence["visual_embedding_available"] is None
        assert evidence["projector_encode_total_s"] is None
        assert evidence["n_past_final"] == 216

    def test_missing_visual_embedding_fails(self, mmproj_file: Path) -> None:
        mtmd = FakeMtmd(embeddings_available=False)
        llm, session, chunks = self._prepare(mmproj_file, mtmd)
        with pytest.raises(MtmdUnavailableError):
            evaluate_chunks(
                llm=llm, session=session, chunks=chunks, mode="split_encode"
            )

    def test_non_zero_encode_and_decode_codes_fail(self, mmproj_file: Path) -> None:
        for kwargs in ({"encode_code": -1}, {"decode_code": -2}):
            mtmd = FakeMtmd(**kwargs)
            llm, session, chunks = self._prepare(mmproj_file, mtmd)
            with pytest.raises(MtmdUnavailableError):
                evaluate_chunks(
                    llm=llm, session=session, chunks=chunks, mode="split_encode"
                )

    def test_context_overflow_is_rejected(self, mmproj_file: Path) -> None:
        mtmd = FakeMtmd()
        session = _session(mmproj_file, mtmd)
        llm = FakeLlama(n_ctx=100)
        session.open(llm)
        chunks = session.tokenize("text <__media__>", [])
        with pytest.raises(MtmdUnavailableError):
            evaluate_chunks(
                llm=llm, session=session, chunks=chunks, mode="split_encode"
            )

    def test_audio_chunk_rejected_and_unknown_mode(self, mmproj_file: Path) -> None:
        audio_chunks = [{"type": CHUNK_AUDIO, "n_tokens": 4, "n_pos": 4}]
        mtmd = FakeMtmd(chunks=audio_chunks)
        llm, session, chunks = self._prepare(mmproj_file, mtmd)
        with pytest.raises(MtmdUnavailableError):
            evaluate_chunks(
                llm=llm, session=session, chunks=chunks, mode="split_encode"
            )
        with pytest.raises(ValueError):
            evaluate_chunks(llm=llm, session=session, chunks=chunks, mode="auto")

    def test_empty_text_chunk_rejected(self, mmproj_file: Path) -> None:
        broken = [{"type": CHUNK_TEXT, "n_tokens": 0, "n_pos": 0, "tokens": []}]
        mtmd = FakeMtmd(chunks=broken)
        llm, session, chunks = self._prepare(mmproj_file, mtmd)
        with pytest.raises(MtmdUnavailableError):
            evaluate_chunks(
                llm=llm, session=session, chunks=chunks, mode="chunk_single"
            )

    def test_image_chunk_without_image_tokens_rejected(self, mmproj_file: Path) -> None:
        chunks_spec = default_chunks()
        chunks_spec[1] = {"type": CHUNK_IMAGE, "n_tokens": 0, "n_pos": 0}
        mtmd = FakeMtmd(chunks=chunks_spec)
        mtmd.mtmd_input_chunk_get_tokens_image = lambda chunk: None  # type: ignore[method-assign]
        _, session, chunks = self._prepare(mmproj_file, mtmd)
        with pytest.raises(MtmdUnavailableError):
            chunks.describe()


class TestFixedDecode:
    def test_generates_exact_token_count_without_reset(self) -> None:
        llm = FakeLlama()
        llm.n_tokens = 216
        result = generate_fixed_tokens(llm=llm, generation_tokens=8)
        assert result["generated_token_count"] == 8
        assert result["token_gap_ms"]["count"] == 7
        assert llm.resets == 0
        assert llm.n_tokens == 224

    def test_rejects_too_small_generation(self) -> None:
        with pytest.raises(ValueError):
            generate_fixed_tokens(llm=FakeLlama(), generation_tokens=1)
