"""CUDA Driver API 只读查询：context 有效 SM 数与驱动版本。

只做查询，不创建 context、不创建 Green Context、不启动任何子进程。
查询失败时返回 ``None`` 与明确原因，绝不回退到按百分比推算。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

CUDA_SUCCESS = 0

#: ``CUexecAffinityType``：MPS 场景下检测 context 关联 SM 数的枚举值。
#: **取值是 0，不是 1**——自该 API 为 MPS 引入起（见 DPDK 2021 年
#: cuExecAffinityParamType patch，注释即 "useful for MPS to detect number
#: of SMs associated to a CUDA context"）SM_COUNT 就是 0；本仓库早期版本
#: 误按 1 调用，驱动对 1 一律返回 CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY(224)，
#: 曾把 SM-A 误判为平台不支持。driver 595.91 实测：传 0 成功且会把
#: ``param.type`` 写回 0。个别头文件出现过 INVALID=0/SM_COUNT=1 的编号，
#: 因此查询按候选序尝试并把实际生效值写进证据。
CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0
#: 候选枚举值（按序尝试；第一个成功且返回正数的即为实际生效值）。
CU_EXEC_AFFINITY_TYPE_CANDIDATES = (0, 1)

_LIBRARY_CANDIDATES = ("libcuda.so.1", "nvcuda.dll", "libcuda.dylib")

#: ``CUexecAffinityParam`` 中 ``type`` 与联合体首成员 ``numSms`` 的字节偏移。
#: 这两个偏移由 C ABI 保证（enum 为 4 字节，联合体首成员从偏移 4 开始）。
_TYPE_OFFSET = 0
_NUM_SMS_OFFSET = 4

#: 联合体 ``CUexecAffinityParamValue`` 的真实大小无法在本仓库内与 CUDA 头文件
#: 逐一核对，因此额外保留一段写入余量，避免驱动写越界；只有前 8 字节会被读取。
_UNION_WRITE_MARGIN_BYTES = 256


class _CuExecAffinityParam(ctypes.Structure):
    """``CUexecAffinityParam`` 的读取布局。

    布局依据：``type``（``CUexecAffinityType``，4 字节）之后紧跟联合体，
    联合体首成员 ``CUexecAffinitySmCount{ unsigned int numSms; }`` 位于偏移 4。
    联合体尾部大小未核对，以 ``padding`` 作为写入余量并在结果中显式标注
    ``layout`` 信息，不据此宣称结构体整体布局已验证。
    """

    _fields_ = [
        ("type", ctypes.c_uint),
        ("num_sms", ctypes.c_uint),
        ("padding", ctypes.c_char * _UNION_WRITE_MARGIN_BYTES),
    ]


def describe_param_layout() -> dict[str, Any]:
    """返回结构体布局自检结果，写入证据而不是口头声称可信。"""

    return {
        "struct_size": ctypes.sizeof(_CuExecAffinityParam),
        "type_offset": _CuExecAffinityParam.type.offset,
        "num_sms_offset": _CuExecAffinityParam.num_sms.offset,
        "offsets_match_c_abi": (
            _CuExecAffinityParam.type.offset == _TYPE_OFFSET
            and _CuExecAffinityParam.num_sms.offset == _NUM_SMS_OFFSET
        ),
        "union_total_size_verified": False,
        "union_write_margin_bytes": _UNION_WRITE_MARGIN_BYTES,
        "note": (
            "type/numSms 偏移由 C ABI 保证；联合体整体大小未与 CUDA 头文件核对，"
            "padding 仅作为写入余量"
        ),
    }


def load_cuda_library(
    candidates: tuple[str, ...] = _LIBRARY_CANDIDATES,
) -> tuple[Any | None, str | None]:
    """按顺序尝试加载 CUDA Driver 库；失败时返回原因而不是抛异常。"""

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError:
            continue
    return None, None


def query_context_exec_affinity_sm_count() -> dict[str, Any]:
    """通过 ``cuCtxGetExecAffinity`` 查询当前 context 的有效 SM 数。

    CUDA Driver API 声明：

    .. code-block:: c

        CUresult cuCtxGetExecAffinity(
            CUexecAffinityParam *pExecAffinity,
            CUexecAffinityType type
        );

    第二个参数是 ``CUexecAffinityType`` 枚举值（按 C ABI 以 ``int`` 传递），
    **不是** 入/出元素个数指针。该枚举在不同驱动/头文件代际间编号有漂移
    （MPS 场景自引入起 SM_COUNT=0，个别新头文件出现过 1），因此按
    ``CU_EXEC_AFFINITY_TYPE_CANDIDATES`` 顺序尝试，第一个返回
    CUDA_SUCCESS 且读出正 numSms 的候选即实际生效值，完整尝试轨迹写入证据。
    """

    result: dict[str, Any] = {
        "context_effective_sm_count": None,
        "context_sm_count_source": None,
        "context_sm_count_unavailable_reason": None,
        "library": None,
        "cu_init_returncode": None,
        "cu_query_returncode": None,
        "requested_affinity_type": None,
        "affinity_type_candidates_tried": [],
        "layout": describe_param_layout(),
    }
    library, library_name = load_cuda_library()
    if library is None:
        result["context_sm_count_unavailable_reason"] = (
            "cuda_driver_library_unavailable"
        )
        return result
    result["library"] = library_name

    cu_init = getattr(library, "cuInit", None)
    cu_query = getattr(library, "cuCtxGetExecAffinity", None)
    if cu_init is None or cu_query is None:
        result["context_sm_count_unavailable_reason"] = (
            "cuCtxGetExecAffinity_symbol_unavailable"
        )
        return result
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    cu_query.argtypes = [
        ctypes.POINTER(_CuExecAffinityParam),
        ctypes.c_int,
    ]
    cu_query.restype = ctypes.c_int

    init_rc = int(cu_init(0))
    result["cu_init_returncode"] = init_rc
    if init_rc != CUDA_SUCCESS:
        result["context_sm_count_unavailable_reason"] = f"cuInit_rc_{init_rc}"
        return result

    for candidate in CU_EXEC_AFFINITY_TYPE_CANDIDATES:
        param = _CuExecAffinityParam()
        param.type = candidate
        query_rc = int(cu_query(ctypes.byref(param), candidate))
        result["affinity_type_candidates_tried"].append(
            {"type": candidate, "returncode": query_rc}
        )
        result["cu_query_returncode"] = query_rc
        result["requested_affinity_type"] = candidate
        if query_rc != CUDA_SUCCESS:
            # 枚举编号不匹配（典型为 224 UNSUPPORTED_EXEC_AFFINITY）时继续
            # 尝试下一候选；全部失败时以最后一次返回码作为原因。
            continue
        if int(param.num_sms) <= 0:
            result["context_sm_count_unavailable_reason"] = (
                "driver_reported_zero_or_non_numeric_sm_count"
            )
            return result
        result["context_effective_sm_count"] = int(param.num_sms)
        result["context_sm_count_source"] = "cuda_driver_cuCtxGetExecAffinity"
        return result
    last_rc = result["cu_query_returncode"]
    result["context_sm_count_unavailable_reason"] = (
        f"cuCtxGetExecAffinity_rc_{last_rc}"
        if last_rc is not None
        else "cuCtxGetExecAffinity_not_attempted"
    )
    return result


def resolve_driver_version() -> tuple[str | None, str | None]:
    """优先用 direct NVML 查询驱动版本，其次只读内核文件。

    WSL2 客体内 NVML 报告的版本对应 Windows 宿主驱动。两条路径都失败时返回
    ``None`` 与原因，绝不启动 ``nvidia-smi`` 子进程。
    """

    try:
        from ..gpu_telemetry import get_system_driver_version
    except Exception as exc:  # noqa: BLE001 - 环境诊断不应中断探针
        return None, f"nvml_import_failed:{exc.__class__.__name__}"
    version, reason = get_system_driver_version()
    if version:
        return version, reason
    return _proc_driver_version()


def _proc_driver_version() -> tuple[str | None, str | None]:
    path = Path("/proc/driver/nvidia/version")
    if not path.is_file():
        return None, "driver_version_not_queryable_without_cli"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, "proc_driver_version_unreadable"
    for line in content.splitlines():
        if "Kernel Module" in line or "NVRM version" in line:
            return line.strip(), "proc_driver_nvidia_version"
    return None, "proc_driver_version_pattern_not_found"
