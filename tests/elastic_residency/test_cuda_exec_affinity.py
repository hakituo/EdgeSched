"""``cuCtxGetExecAffinity`` ABI 的回归测试（fake ctypes，不加载真实 CUDA）。"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.elastic_residency.sm import cuda_exec_affinity as module
from experiments.elastic_residency.sm.cuda_exec_affinity import (
    CU_EXEC_AFFINITY_TYPE_SM_COUNT,
    describe_param_layout,
    query_context_exec_affinity_sm_count,
)


class _FakeCudaLibrary:
    """记录调用参数的最小 CUDA Driver 替身。

    ``query_rc_by_type`` 模拟枚举编号漂移：只有传入的 type 命中表内项时
    使用对应返回码，其余用 ``query_rc``。默认全部成功（等价于真实驱动
    上 SM_COUNT=0 首次即中的行为）。
    """

    def __init__(
        self,
        *,
        query_rc: int = 0,
        num_sms: int = 24,
        query_rc_by_type: dict[int, int] | None = None,
    ) -> None:
        self.query_rc = query_rc
        self.num_sms = num_sms
        self.query_rc_by_type = dict(query_rc_by_type or {})
        self.init_calls: list[Any] = []
        self.query_calls: list[tuple[Any, Any]] = []

        def cu_init(flags: Any) -> int:
            self.init_calls.append(flags)
            return 0

        def cu_ctx_get_exec_affinity(param: Any, second: Any) -> int:
            self.query_calls.append((param, second))
            rc = self.query_rc_by_type.get(second, self.query_rc)
            if rc == 0:
                param._obj.num_sms = self.num_sms
            return rc

        self.cuInit = cu_init
        self.cuCtxGetExecAffinity = cu_ctx_get_exec_affinity


@pytest.fixture
def fake_library(monkeypatch: pytest.MonkeyPatch) -> _FakeCudaLibrary:
    library = _FakeCudaLibrary()

    def _load(*_args: Any, **_kwargs: Any) -> tuple[Any, str]:
        return library, "libcuda.so.1"

    monkeypatch.setattr(module, "load_cuda_library", _load)
    return library


def test_second_argument_is_the_affinity_type_enum(
    fake_library: _FakeCudaLibrary,
) -> None:
    """第二个参数必须是整数枚举值，不能是元素个数指针。"""

    result = query_context_exec_affinity_sm_count()
    assert result["context_effective_sm_count"] == 24
    assert result["cu_query_returncode"] == 0
    assert fake_library.query_calls, "必须真正调用 cuCtxGetExecAffinity"
    _param, second = fake_library.query_calls[0]
    assert isinstance(second, int), "第二个参数必须是 int 枚举，不能是指针/byref"
    assert not isinstance(second, bool)
    # MPS 场景 SM_COUNT 枚举值是 0（DPDK 2021 cuExecAffinityParamType 起），
    # 不是 1；传 1 会被驱动以 UNSUPPORTED_EXEC_AFFINITY 拒绝。
    assert second == CU_EXEC_AFFINITY_TYPE_SM_COUNT == 0
    assert not hasattr(second, "value"), "不得传入 ctypes 包装对象"
    assert result["requested_affinity_type"] == CU_EXEC_AFFINITY_TYPE_SM_COUNT


def test_argtypes_declare_pointer_and_c_int(fake_library: _FakeCudaLibrary) -> None:
    import ctypes

    query_context_exec_affinity_sm_count()
    argtypes = fake_library.cuCtxGetExecAffinity.argtypes
    assert len(argtypes) == 2
    # 注意：Windows 上 ctypes.c_int 与 c_long 是同一类型，不能按名字断言。
    assert argtypes[1] is ctypes.c_int
    assert ctypes.sizeof(argtypes[1]) == 4
    assert fake_library.cuCtxGetExecAffinity.restype is ctypes.c_int


def test_param_layout_offsets_match_c_abi() -> None:
    layout = describe_param_layout()
    assert layout["type_offset"] == 0
    assert layout["num_sms_offset"] == 4
    assert layout["offsets_match_c_abi"] is True
    # 联合体整体大小未与头文件核对，必须显式标注，不得宣称已验证。
    assert layout["union_total_size_verified"] is False
    assert layout["struct_size"] >= 8


def test_nonzero_return_code_is_recorded_not_faked(
    fake_library: _FakeCudaLibrary,
) -> None:
    fake_library.query_rc = 201
    result = query_context_exec_affinity_sm_count()
    assert result["context_effective_sm_count"] is None
    assert result["context_sm_count_source"] is None
    assert result["context_sm_count_unavailable_reason"] == (
        "cuCtxGetExecAffinity_rc_201"
    )
    assert result["cu_query_returncode"] == 201


def test_zero_sm_count_is_not_reported_as_measurement(
    fake_library: _FakeCudaLibrary,
) -> None:
    fake_library.num_sms = 0
    result = query_context_exec_affinity_sm_count()
    assert result["context_effective_sm_count"] is None
    assert result["context_sm_count_unavailable_reason"] == (
        "driver_reported_zero_or_non_numeric_sm_count"
    )


def test_cu_init_failure_short_circuits(fake_library: _FakeCudaLibrary) -> None:
    def failing_init(_flags: Any) -> int:
        return 999

    fake_library.cuInit = failing_init
    result = query_context_exec_affinity_sm_count()
    assert result["context_sm_count_unavailable_reason"] == "cuInit_rc_999"
    assert fake_library.query_calls == []


def test_missing_symbols_are_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoSymbols:
        def cuInit(self, _flags: Any) -> int:
            return 0

    monkeypatch.setattr(
        module, "load_cuda_library", lambda *_a, **_k: (_NoSymbols(), "libcuda.so.1")
    )
    result = query_context_exec_affinity_sm_count()
    assert result["context_sm_count_unavailable_reason"] == (
        "cuCtxGetExecAffinity_symbol_unavailable"
    )
    assert result["context_effective_sm_count"] is None


def test_library_unavailable_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module, "load_cuda_library", lambda *_a, **_k: (None, None)
    )
    result = query_context_exec_affinity_sm_count()
    assert result["context_sm_count_unavailable_reason"] == (
        "cuda_driver_library_unavailable"
    )


def test_enum_drift_retries_next_candidate_and_records_trail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """枚举编号漂移（0 被拒、1 成功）时必须重试下一候选并记录轨迹。"""

    library = _FakeCudaLibrary(query_rc_by_type={0: 224, 1: 0}, num_sms=18)
    monkeypatch.setattr(
        module, "load_cuda_library", lambda *_a, **_k: (library, "libcuda.so.1")
    )
    result = query_context_exec_affinity_sm_count()
    assert result["context_effective_sm_count"] == 18
    assert result["requested_affinity_type"] == 1
    assert result["cu_query_returncode"] == 0
    assert result["affinity_type_candidates_tried"] == [
        {"type": 0, "returncode": 224},
        {"type": 1, "returncode": 0},
    ]


def test_all_candidates_fail_records_last_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全部候选失败时以最后一次返回码作为原因，不得伪造测量值。"""

    library = _FakeCudaLibrary(query_rc=224)
    monkeypatch.setattr(
        module, "load_cuda_library", lambda *_a, **_k: (library, "libcuda.so.1")
    )
    result = query_context_exec_affinity_sm_count()
    assert result["context_effective_sm_count"] is None
    assert result["context_sm_count_unavailable_reason"] == (
        "cuCtxGetExecAffinity_rc_224"
    )
    assert [item["type"] for item in result["affinity_type_candidates_tried"]] == [
        0,
        1,
    ]


def test_successful_query_does_not_retry_other_candidates(
    fake_library: _FakeCudaLibrary,
) -> None:
    """首个候选即成功时不得继续尝试其他枚举值。"""

    result = query_context_exec_affinity_sm_count()
    assert result["context_effective_sm_count"] == 24
    assert [second for _param, second in fake_library.query_calls] == [0]
