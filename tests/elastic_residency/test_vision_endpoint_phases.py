"""文本/视觉端点执行阶段与编排计划的离线测试（全部使用假后端）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.elastic_residency.vision_mtmd_fakes import (  # noqa: E402
    FakeLlama,
    FakeMtmd,
    install_fake_llama_cpp,
    install_fake_mtmd_module,
)

install_fake_llama_cpp()

from experiments.elastic_residency.vision.vision_endpoint_phases import (  # noqa: E402
    VisionPhaseConfig,
    run_text_phase,
    run_vision_phase,
)
from experiments.elastic_residency.vision.vision_endpoint_smoke import (  # noqa: E402
    build_plan,
)
from experiments.elastic_residency.vision.vision_evidence import (  # noqa: E402
    RESIDENCY_TEXT_ONLY,
    RESIDENCY_TEXT_ONLY_CHAT,
    build_prompt_evidence,
)


class _Snapshots:
    """按调用顺序返回递增的整卡显存与 RSS，模拟 projector warm/释放。"""

    def __init__(self, values: list[tuple[float, int]]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> tuple[dict[str, float], int]:
        card, rss = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return {"gpu_card_used_mb": card}, rss


@pytest.fixture()
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "fixed.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


@pytest.fixture()
def mmproj_file(tmp_path: Path) -> Path:
    path = tmp_path / "mmproj.gguf"
    path.write_bytes(b"GGUF-placeholder")
    return path


def _vision_config(mmproj: Path, image: Path, **overrides) -> VisionPhaseConfig:
    values = {
        "mmproj_path": mmproj,
        "image_path": image,
        "question": "请描述这张图片。",
        "prompt_mode": "qwen_im",
        "eval_mode": "split_encode",
        "generation_tokens": 8,
        "use_gpu": True,
        "n_threads": 14,
    }
    values.update(overrides)
    return VisionPhaseConfig(**values)  # type: ignore[arg-type]


class TestTextPhase:
    def test_fixed_token_filler_prompt(self) -> None:
        llm = FakeLlama()
        phase = run_text_phase(
            llm=llm,
            residency_state=RESIDENCY_TEXT_ONLY,
            context_tokens=512,
            generation_tokens=8,
            prompt_evidence=None,
        )
        assert phase["vision"] is None
        assert phase["prompt"]["prompt_construction_mode"] == "fixed_token_filler"
        assert phase["prompt"]["prompt_token_count"] == 512
        assert phase["prompt_position_total"] == 512
        assert phase["inference"]["generated_token_count"] == 8
        assert phase["kv_native_state"]["kv_tokens_used"] == 520
        assert phase["kv_roundtrip"]["succeeded"] is True
        assert phase["output"]["output_text"]
        assert len(phase["output"]["output_text_sha256"]) == 64

    def test_chat_control_prompt_reuses_vision_prompt(self) -> None:
        llm = FakeLlama()
        prompt = build_prompt_evidence(
            mode="qwen_im", question="请描述这张图片。", media_marker=None
        )
        phase = run_text_phase(
            llm=llm,
            residency_state=RESIDENCY_TEXT_ONLY_CHAT,
            context_tokens=512,
            generation_tokens=8,
            prompt_evidence=prompt,
        )
        assert phase["prompt"]["prompt_construction_mode"] == "qwen_im"
        assert phase["prompt"]["prompt_contains_media_marker"] is False
        assert phase["prompt"]["prompt_token_count"] > 0

    def test_missing_chat_prompt_and_unknown_state_rejected(self) -> None:
        with pytest.raises(ValueError):
            run_text_phase(
                llm=FakeLlama(),
                residency_state=RESIDENCY_TEXT_ONLY_CHAT,
                context_tokens=512,
                generation_tokens=8,
                prompt_evidence=None,
            )
        with pytest.raises(ValueError):
            run_text_phase(
                llm=FakeLlama(),
                residency_state="VISION_WARM_GPU",
                context_tokens=512,
                generation_tokens=8,
                prompt_evidence=None,
            )

    def test_context_overflow_rejected(self) -> None:
        with pytest.raises(ValueError):
            run_text_phase(
                llm=FakeLlama(n_ctx=128),
                residency_state=RESIDENCY_TEXT_ONLY,
                context_tokens=512,
                generation_tokens=8,
                prompt_evidence=None,
            )


class TestVisionPhase:
    def test_real_path_records_projector_and_visual_tokens(
        self, mmproj_file: Path, image_file: Path
    ) -> None:
        fake = FakeMtmd()
        install_fake_mtmd_module(fake)
        snapshots = _Snapshots(
            [(3000.0, 1_000_000), (3640.0, 1_200_000), (3640.0, 1_200_000), (3010.0, 1_050_000)]
        )
        phase = run_vision_phase(
            llm=FakeLlama(),
            config=_vision_config(mmproj_file, image_file),
            chat_template=None,
            snapshot=snapshots,
        )
        vision = phase["vision"]
        assert vision["session"]["residency_state"] == "VISION_WARM_GPU"
        assert vision["chunk_execution"]["visual_token_total"] == 196
        assert vision["chunk_execution"]["visual_embedding_available"] is True
        assert vision["projector_warm_delta"]["gpu_card_delta_mb"] == pytest.approx(640.0)
        assert vision["projector_warm_delta"]["rss_delta_bytes"] == 200_000
        assert vision["release"]["projector_freed"] is True
        assert vision["bitmaps"][0]["decoded_width_px"] == 448
        assert phase["prompt"]["prompt_contains_media_marker"] is True
        assert phase["prompt_position_total"] == 216
        assert phase["kv_native_state"]["kv_tokens_used"] == 224
        assert phase["inference"]["ttft_s"] > 0
        # Windows 的 monotonic_ns 分辨率约 15.6 ms，假后端可能落在同一刻度，
        # 因此这里只要求区间单调不减；真实 Linux 运行才有纳秒级区间。
        assert phase["phase_intervals_ns"]["inference"][0] <= (
            phase["phase_intervals_ns"]["inference"][1]
        )
        assert set(phase["phase_intervals_ns"]) == {"prefill", "decode", "inference"}
        assert fake.freed_bitmaps == [1001]
        assert fake.freed_chunk_handles == [7777]
        assert fake.ctx_freed is True

    def test_chunk_single_mode_leaves_encode_evidence_none(
        self, mmproj_file: Path, image_file: Path
    ) -> None:
        fake = FakeMtmd()
        install_fake_mtmd_module(fake)
        phase = run_vision_phase(
            llm=FakeLlama(),
            config=_vision_config(mmproj_file, image_file, eval_mode="chunk_single"),
            chat_template=None,
            snapshot=_Snapshots([(3000.0, 1_000_000)]),
        )
        execution = phase["vision"]["chunk_execution"]
        assert execution["projector_encode_total_s"] is None
        assert execution["visual_embedding_available"] is None

    def test_projector_failure_still_frees_handles(
        self, mmproj_file: Path, image_file: Path
    ) -> None:
        fake = FakeMtmd(encode_code=-1)
        install_fake_mtmd_module(fake)
        with pytest.raises(Exception):
            run_vision_phase(
                llm=FakeLlama(),
                config=_vision_config(mmproj_file, image_file),
                chat_template=None,
                snapshot=_Snapshots([(3000.0, 1_000_000)]),
            )
        assert fake.freed_bitmaps == [1001]
        assert fake.freed_chunk_handles == [7777]

    def test_cpu_projector_reports_cpu_state(
        self, mmproj_file: Path, image_file: Path
    ) -> None:
        install_fake_mtmd_module(FakeMtmd())
        phase = run_vision_phase(
            llm=FakeLlama(),
            config=_vision_config(mmproj_file, image_file, use_gpu=False),
            chat_template=None,
            snapshot=_Snapshots([(3000.0, 1_000_000)]),
        )
        assert phase["vision"]["session"]["residency_state"] == "VISION_WARM_CPU"
        assert phase["vision"]["session"]["projector_use_gpu"] is False


class TestPlan:
    def test_default_matrix_order(self) -> None:
        plan = build_plan(("K0", "FULL"), ("TEXT_ONLY", "VISION_WARM_GPU"), 1)
        assert [(item[0], item[1]) for item in plan] == [
            ("K0", "TEXT_ONLY"),
            ("K0", "VISION_WARM_GPU"),
            ("FULL", "TEXT_ONLY"),
            ("FULL", "VISION_WARM_GPU"),
        ]
        assert [item[3] for item in plan] == [1, 2, 3, 4]

    def test_repeats_expand_per_cell(self) -> None:
        plan = build_plan(("FULL",), ("VISION_WARM_GPU",), 3)
        assert [item[2] for item in plan] == [1, 2, 3]

    def test_invalid_inputs_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_plan((), ("TEXT_ONLY",), 1)
        with pytest.raises(ValueError):
            build_plan(("K0", "K0"), ("TEXT_ONLY",), 1)
        with pytest.raises(ValueError):
            build_plan(("K0",), ("TEXT_ONLY", "TEXT_ONLY"), 1)
        with pytest.raises(ValueError):
            build_plan(("K0",), ("VISION_WARM_TPU",), 1)
        with pytest.raises(ValueError):
            build_plan(("K0",), ("TEXT_ONLY",), 0)
