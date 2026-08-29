"""KV 测量语义、日志解析与原生接口采集的单元测试。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.kv_measurement import (  # noqa: E402
    BYTES_PER_MIB,
    KV_MEASUREMENT_SEMANTICS,
    buffer_report_acceptance,
    capture_llama_kv_state,
    parse_llama_buffer_lines,
    position_span_from_bounds,
)


REQUIRED_SEMANTIC_FIELDS = (
    "full_state_serialized_bytes",
    "kv_sequence_state_bytes",
    "kv_position_min",
    "kv_position_max",
    "kv_tokens_used",
    "kv_gpu_buffer_mb",
    "kv_cpu_buffer_mb",
    "compute_gpu_buffer_mb",
    "compute_cpu_buffer_mb",
    "kv_state_bytes",
    "gpu_card_delta_from_baseline_mb",
)


class TestFieldSemantics:
    def test_all_required_fields_have_semantics(self) -> None:
        for field in REQUIRED_SEMANTIC_FIELDS:
            assert field in KV_MEASUREMENT_SEMANTICS
            assert KV_MEASUREMENT_SEMANTICS[field].strip()

    def test_full_state_semantics_disclaims_pure_kv(self) -> None:
        text = KV_MEASUREMENT_SEMANTICS["full_state_serialized_bytes"]
        assert "logits" in text
        assert "embedding" in text
        assert "不是纯 GPU KV 显存" in text

    def test_sequence_state_semantics_disclaims_gpu_buffer(self) -> None:
        text = KV_MEASUREMENT_SEMANTICS["kv_sequence_state_bytes"]
        assert "不等于 GPU 预分配 KV buffer" in text

    def test_kv_state_bytes_is_compatibility_alias(self) -> None:
        assert "兼容字段" in KV_MEASUREMENT_SEMANTICS["kv_state_bytes"]

    def test_buffer_fields_distinguish_capacity_from_usage(self) -> None:
        assert "容量" in KV_MEASUREMENT_SEMANTICS["kv_gpu_buffer_mb"]
        assert "使用量而不是容量" in KV_MEASUREMENT_SEMANTICS["kv_tokens_used"]


class TestPositionSpanFromBounds:
    def test_contiguous_range(self) -> None:
        assert position_span_from_bounds(0, 511) == 512

    def test_single_token(self) -> None:
        assert position_span_from_bounds(7, 7) == 1

    def test_empty_cache_returns_zero(self) -> None:
        assert position_span_from_bounds(-1, -1) == 0

    def test_none_position_raises(self) -> None:
        with pytest.raises(ValueError, match="不得计算"):
            position_span_from_bounds(None, 5)  # type: ignore[arg-type]


class _FakeApi:
    """模拟 llama_cpp 原生接口。"""

    def __init__(
        self,
        *,
        full_state: int | None,
        seq_state: int | None,
        pos_min: int | None,
        pos_max: int | None,
    ) -> None:
        self._full_state = full_state
        self._seq_state = seq_state
        self._pos_min = pos_min
        self._pos_max = pos_max

    def llama_get_memory(self, ctx: object) -> object:
        return object()

    def llama_state_get_size(self, ctx: object) -> int | None:
        return self._full_state

    def llama_state_seq_get_size(self, ctx: object, seq_id: int) -> int | None:
        return self._seq_state

    def llama_memory_seq_pos_min(self, memory: object, seq_id: int) -> int | None:
        return self._pos_min

    def llama_memory_seq_pos_max(self, memory: object, seq_id: int) -> int | None:
        return self._pos_max


def _fake_llm(*, with_memory: bool = False, n_tokens: int = 0) -> object:
    context = types.SimpleNamespace(ctx=object())
    if with_memory:
        context.memory = object()
    return types.SimpleNamespace(_ctx=context, n_tokens=n_tokens)


class TestCaptureLlamaKvState:
    def test_captures_all_fields(self) -> None:
        api = _FakeApi(full_state=4096, seq_state=2048, pos_min=0, pos_max=511)
        state = capture_llama_kv_state(
            _fake_llm(n_tokens=512), seq_id=0, api=api
        )
        assert state["full_state_serialized_bytes"] == 4096
        assert state["kv_sequence_state_bytes"] == 2048
        assert state["kv_position_min"] == 0
        assert state["kv_position_max"] == 511
        assert state["kv_tokens_used"] == 512
        assert state["kv_tokens_used_source"] == "llama_cpp_python.n_tokens"
        assert state["kv_position_span"] == 512

    def test_uses_context_memory_when_available(self) -> None:
        api = _FakeApi(full_state=10, seq_state=5, pos_min=-1, pos_max=-1)
        state = capture_llama_kv_state(_fake_llm(with_memory=True), api=api)
        assert state["kv_tokens_used"] == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"full_state": None, "seq_state": 5, "pos_min": 0, "pos_max": 1},
            {"full_state": 5, "seq_state": None, "pos_min": 0, "pos_max": 1},
            {"full_state": 5, "seq_state": 5, "pos_min": None, "pos_max": 1},
            {"full_state": 5, "seq_state": 5, "pos_min": 0, "pos_max": None},
        ],
    )
    def test_unavailable_results_raise_instead_of_zero(self, kwargs) -> None:
        api = _FakeApi(**kwargs)
        with pytest.raises(RuntimeError, match="不可用"):
            capture_llama_kv_state(_fake_llm(), api=api)

    def test_missing_context_handle_raises(self) -> None:
        api = _FakeApi(full_state=5, seq_state=5, pos_min=0, pos_max=1)
        with pytest.raises(RuntimeError, match="上下文句柄"):
            capture_llama_kv_state(
                types.SimpleNamespace(_ctx=None), api=api
            )


SAMPLE_LOG_LINES = [
    "llama_kv_cache_unified: CUDA0 KV buffer size = 512.00 MiB",
    "llama_kv_cache_unified: CPU KV buffer size = 64.00 MiB",
    "llama_init: CUDA0 compute buffer size = 128.50 MiB",
    "llama_init: CPU compute buffer size = 8.00 MiB",
    "llama_init: CUDA_Host compute buffer size = 4.00 MiB",
    "llama_model_load_internal: model size = 4096.00 MiB",
]


class TestParseLlamaBufferLines:
    def test_parses_gpu_kv_and_cpu_kv(self) -> None:
        report = parse_llama_buffer_lines(SAMPLE_LOG_LINES)
        assert report["kv_gpu_buffer_mb"] == pytest.approx(512.0)
        assert report["kv_cpu_buffer_mb"] == pytest.approx(64.0)

    def test_gpu_compute_excludes_cuda_host(self) -> None:
        report = parse_llama_buffer_lines(SAMPLE_LOG_LINES)
        assert report["compute_gpu_buffer_mb"] == pytest.approx(128.5)
        assert report["compute_cpu_buffer_mb"] == pytest.approx(12.0)
        assert report["compute_buffer_mb_by_device"]["CUDA_Host"] == pytest.approx(
            4.0
        )

    def test_missing_devices_yield_none_not_zero(self) -> None:
        report = parse_llama_buffer_lines(["no buffer lines here"])
        assert report["kv_gpu_buffer_mb"] is None
        assert report["kv_cpu_buffer_mb"] is None
        assert report["compute_gpu_buffer_mb"] is None
        assert report["compute_cpu_buffer_mb"] is None
        assert report["matched_line_count"] == 0

    def test_unit_conversion_gib_and_bytes(self) -> None:
        report = parse_llama_buffer_lines(
            [
                "CUDA0 KV buffer size = 1.00 GiB",
                "CUDA0 KV buffer size = 1048576 B",
            ]
        )
        expected = (1024.0 + 1.0) * 1024 * 1024 / BYTES_PER_MIB
        assert report["kv_gpu_buffer_mb"] == pytest.approx(expected)

    def test_accumulates_repeated_device_lines(self) -> None:
        report = parse_llama_buffer_lines(
            [
                "CUDA0 KV buffer size = 100.00 MiB",
                "CUDA0 KV buffer size = 50.00 MiB",
            ]
        )
        assert report["kv_gpu_buffer_mb"] == pytest.approx(150.0)
        assert report["matched_line_count"] == 2


class TestBufferReportAcceptance:
    def test_accepts_gpu_kv_only(self) -> None:
        report = parse_llama_buffer_lines(["CUDA0 KV buffer size = 1.00 MiB"])
        checks = buffer_report_acceptance(report)
        assert checks["kv_buffer_size_captured"] is True
        assert checks["compute_buffer_size_captured"] is False

    def test_rejects_missing_report(self) -> None:
        checks = buffer_report_acceptance(None)
        assert checks == {
            "kv_buffer_size_captured": False,
            "compute_buffer_size_captured": False,
        }

    def test_rejects_empty_report(self) -> None:
        checks = buffer_report_acceptance({})
        assert not any(checks.values())


class TestLlamaInitLogCapture:
    def _install_fake_llama_cpp(self, monkeypatch: pytest.MonkeyPatch) -> list:
        calls: list = []

        def fake_log_set(callback, user_data) -> None:
            calls.append(callback)

        def decorator(func):
            return func

        fake_module = types.SimpleNamespace(
            llama_log_set=fake_log_set,
            llama_log_callback=decorator,
        )
        monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
        return calls

    def test_collects_lines_and_parses_buffers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._install_fake_llama_cpp(monkeypatch)

        from experiments.elastic_residency.kv_measurement import (
            LlamaInitLogCapture,
        )

        with LlamaInitLogCapture() as capture:
            callback = calls[-1]
            callback(1, b"CUDA0 KV buffer size = 256.00 MiB\n", None)
            # 非激活阶段（模拟 enter 前）不收集。
        assert calls, "log_set 必须被调用以安装捕获回调"
        report = capture.buffer_report()
        assert report["kv_gpu_buffer_mb"] == pytest.approx(256.0)
        assert capture.lines == ["CUDA0 KV buffer size = 256.00 MiB\n"]

    def test_custom_log_set_restored_on_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_fake_llama_cpp(monkeypatch)
        restored: list = []

        def custom_log_set(callback, user_data) -> None:
            restored.append((callback, user_data))

        from experiments.elastic_residency.kv_measurement import (
            LlamaInitLogCapture,
        )

        with LlamaInitLogCapture(log_set=custom_log_set):
            assert restored, "进入上下文时必须安装捕获回调"
            installed_callback = restored[-1][0]
            assert installed_callback is not None
        assert restored[-1] == (None, None)
