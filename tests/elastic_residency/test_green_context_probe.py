"""Green Context 只读能力探测的单元测试（不加载真实 CUDA）。"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.elastic_residency.sm import green_context_probe as module
from experiments.elastic_residency.sm.green_context_probe import (
    DEVICE_RESOURCE_LAYOUT_REASON,
    REQUIRED_SYMBOLS,
    describe_resource_layout,
    probe_green_context,
)


class _FakeLibrary:
    def __init__(self, symbols: set[str] | None = None) -> None:
        available = symbols if symbols is not None else set(REQUIRED_SYMBOLS)
        for name in REQUIRED_SYMBOLS:
            if name in available:
                setattr(self, name, object())


def test_required_symbols_use_cu_green_ctx_stream_create() -> None:
    """stream 符号是 cuGreenCtxStreamCreate，不是不存在的旧名。"""

    assert "cuGreenCtxStreamCreate" in REQUIRED_SYMBOLS
    assert "cuStreamCreateWithGreenCtx" not in REQUIRED_SYMBOLS
    assert "cuGreenCtxCreate" in REQUIRED_SYMBOLS


def test_missing_cuda_library_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "load_cuda_library", lambda *_a, **_k: (None, None))
    capability = probe_green_context()
    assert capability.supported is False
    assert capability.status == "unsupported_cuda_driver_library_unavailable"
    assert capability.missing_symbols == REQUIRED_SYMBOLS
    assert capability.minimum_partition_sms is None
    assert capability.gate_conditions["device_resource_layout_verified"] is False


def test_missing_symbols_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "load_cuda_library",
        lambda *_a, **_k: (_FakeLibrary({"cuGreenCtxCreate"}), "libcuda.so.1"),
    )
    capability = probe_green_context()
    assert capability.supported is False
    assert set(capability.missing_symbols) == set(REQUIRED_SYMBOLS) - {
        "cuGreenCtxCreate"
    }
    assert "required_driver_api_symbols_missing" in capability.reasons


def test_all_symbols_present_still_not_claiming_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """符号齐全也不能声称可用：创建/绑定/逃逸三项条件未证明。"""

    monkeypatch.setattr(
        module,
        "load_cuda_library",
        lambda *_a, **_k: (_FakeLibrary(), "libcuda.so.1"),
    )
    capability = probe_green_context()
    assert capability.supported is False
    assert capability.gate_conditions["driver_api_symbols"] is True
    assert capability.gate_conditions["green_context_creation"] is False
    assert capability.gate_conditions["framework_stream_binding"] is False
    assert capability.gate_conditions["no_default_stream_escape"] is False
    assert capability.device_resource_query_attempted is False
    assert "device_resource_query_not_attempted_read_only_probe" in capability.reasons
    assert "framework_stream_binding_unproven" in capability.reasons
    assert DEVICE_RESOURCE_LAYOUT_REASON in capability.reasons


def _library_with_device_resource(sm_count: int, *, rc: int = 0) -> Any:
    library = _FakeLibrary()

    def cu_init(_flags: int) -> int:
        return 0

    def cu_device_get(device: Any, _index: int) -> int:
        device._obj.value = 0
        return 0

    def cu_dev_resource(_device: int, resource: Any, _kind: int) -> int:
        if rc != 0:
            return rc
        resource._obj.data.sm.sm_count = sm_count
        return 0

    library.cuInit = cu_init
    library.cuDeviceGet = cu_device_get
    library.cuDeviceGetDevResource = cu_dev_resource
    return library


def test_device_resource_observation_stays_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """查询成功也只能是未验证观测值，不得升级成设备能力结论。"""

    monkeypatch.setattr(
        module,
        "load_cuda_library",
        lambda *_a, **_k: (_library_with_device_resource(36), "libcuda.so.1"),
    )
    capability = probe_green_context(attempt_device_resource=True)
    assert capability.device_resource_query_attempted is True
    assert capability.device_sm_count is None
    assert capability.device_sm_count_unverified == 36
    assert capability.device_resource_layout_verified is False
    assert capability.minimum_partition_sms is None
    assert capability.supported is False
    assert capability.gate_conditions["device_resource_query"] is False
    assert capability.gate_conditions["min_partition_granularity"] is False
    assert DEVICE_RESOURCE_LAYOUT_REASON in capability.reasons


def test_implausible_value_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "load_cuda_library",
        lambda *_a, **_k: (_library_with_device_resource(-7), "libcuda.so.1"),
    )
    capability = probe_green_context(attempt_device_resource=True)
    assert capability.device_sm_count_unverified is None
    assert "device_resource_query_failed" in capability.reasons
    assert capability.supported is False


def test_driver_error_is_recorded_without_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "load_cuda_library",
        lambda *_a, **_k: (_library_with_device_resource(36, rc=201), "libcuda.so.1"),
    )
    capability = probe_green_context(attempt_device_resource=True)
    assert capability.device_sm_count_unverified is None
    assert capability.details["device_resource_query_reason"] == (
        "cuDeviceGetDevResource_rc_201"
    )
    assert capability.supported is False


def test_exception_during_query_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Exploding(_FakeLibrary):
        def cuInit(self, _flags: int) -> int:
            raise RuntimeError("boom")

        def cuDeviceGet(self, *_args: Any) -> int:
            raise RuntimeError("boom")

        def cuDeviceGetDevResource(self, *_args: Any) -> int:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        module, "load_cuda_library", lambda *_a, **_k: (_Exploding(), "libcuda.so.1")
    )
    capability = probe_green_context(attempt_device_resource=True)
    assert capability.device_sm_count_unverified is None
    assert capability.details["device_resource_query_reason"].startswith(
        "device_resource_query_exception:"
    )


def test_resource_layout_report_does_not_claim_verification() -> None:
    layout = describe_resource_layout()
    assert layout["layout_verified"] is False
    assert layout["reason"] == DEVICE_RESOURCE_LAYOUT_REASON
    assert layout["type_offset"] == 0


def test_capability_is_json_serializable() -> None:
    payload = probe_green_context().to_dict()
    assert payload["schema_version"] == "edgesched.green_context_probe.v1"
    assert payload["supported"] is False
    assert sorted(payload["symbols"]) == sorted(REQUIRED_SYMBOLS)
