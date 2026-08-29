"""论文实验使用的低开销 GPU 遥测。

正式路径直接调用 NVIDIA Management Library (NVML)，不启动 ``nvidia-smi``
子进程。CLI 仍可作为人工诊断工具，但不属于本模块的默认或回退路径。
"""

from __future__ import annotations

import ctypes
import threading
from functools import lru_cache
from typing import Any, Protocol


NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1
BYTES_PER_MIB = 1024.0 * 1024.0


class GPUProbe(Protocol):
    backend_name: str

    def sample(self, pid: int) -> dict[str, float | bool | str | None]: ...


class NvmlUnavailableError(RuntimeError):
    """direct NVML 无法初始化或核心查询失败。"""


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlProcessInfoV2(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class NvmlBindings:
    """NVML ctypes 薄绑定；只暴露实验所需的只读查询。"""

    def __init__(
        self,
        device_index: int = 0,
        library_name: str = "libnvidia-ml.so.1",
    ) -> None:
        try:
            self._library = ctypes.CDLL(library_name)
        except OSError as exc:
            raise NvmlUnavailableError(f"无法加载 {library_name}: {exc}") from exc
        self._configure_symbols()
        self._check(self._init(), "nvmlInit_v2")
        self._handle = ctypes.c_void_p()
        self._check(
            self._get_handle(device_index, ctypes.byref(self._handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        self._lock = threading.Lock()

    def _configure_symbols(self) -> None:
        try:
            self._init = self._library.nvmlInit_v2
            self._get_handle = self._library.nvmlDeviceGetHandleByIndex_v2
            self._get_memory = self._library.nvmlDeviceGetMemoryInfo
            self._get_utilization = self._library.nvmlDeviceGetUtilizationRates
            self._get_power = self._library.nvmlDeviceGetPowerUsage
            self._get_temperature = self._library.nvmlDeviceGetTemperature
        except AttributeError as exc:
            raise NvmlUnavailableError(f"NVML 缺少核心符号: {exc}") from exc
        self._get_processes = getattr(
            self._library,
            "nvmlDeviceGetComputeRunningProcesses_v2",
            None,
        )
        self._init.restype = ctypes.c_int
        self._get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self._get_handle.restype = ctypes.c_int
        self._get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
        self._get_memory.restype = ctypes.c_int
        self._get_utilization.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NvmlUtilization),
        ]
        self._get_utilization.restype = ctypes.c_int
        self._get_power.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self._get_power.restype = ctypes.c_int
        self._get_temperature.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self._get_temperature.restype = ctypes.c_int
        if self._get_processes is not None:
            self._get_processes.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(_NvmlProcessInfoV2),
            ]
            self._get_processes.restype = ctypes.c_int

    @staticmethod
    def _check(return_code: int, operation: str) -> None:
        if return_code != NVML_SUCCESS:
            raise NvmlUnavailableError(f"{operation} failed rc={return_code}")

    def device_metrics(self) -> dict[str, float]:
        with self._lock:
            memory = _NvmlMemory()
            self._check(
                self._get_memory(self._handle, ctypes.byref(memory)),
                "nvmlDeviceGetMemoryInfo",
            )
            utilization = _NvmlUtilization()
            utilization_rc = self._get_utilization(
                self._handle,
                ctypes.byref(utilization),
            )
            power_mw = ctypes.c_uint()
            power_rc = self._get_power(self._handle, ctypes.byref(power_mw))
            temperature = ctypes.c_uint()
            temperature_rc = self._get_temperature(
                self._handle,
                0,
                ctypes.byref(temperature),
            )
        return {
            "card_used_mb": memory.used / BYTES_PER_MIB,
            "card_total_mb": memory.total / BYTES_PER_MIB,
            "utilization_percent": (
                float(utilization.gpu) if utilization_rc == NVML_SUCCESS else 0.0
            ),
            "power_w": (
                power_mw.value / 1000.0 if power_rc == NVML_SUCCESS else 0.0
            ),
            "temperature_c": (
                float(temperature.value)
                if temperature_rc == NVML_SUCCESS
                else 0.0
            ),
        }

    def process_memory(self, pid: int) -> tuple[float | None, str | None]:
        if self._get_processes is None:
            return None, "nvml_process_query_symbol_unavailable"
        count = ctypes.c_uint(0)
        with self._lock:
            first_rc = self._get_processes(self._handle, ctypes.byref(count), None)
            if first_rc == NVML_SUCCESS and count.value == 0:
                return None, "no_compute_process_rows"
            if first_rc not in (NVML_SUCCESS, NVML_ERROR_INSUFFICIENT_SIZE):
                return None, f"nvml_process_query_failed_rc_{first_rc}"
            capacity = max(1, count.value)
            rows = (_NvmlProcessInfoV2 * capacity)()
            second_rc = self._get_processes(
                self._handle,
                ctypes.byref(count),
                rows,
            )
        if second_rc != NVML_SUCCESS:
            return None, f"nvml_process_query_failed_rc_{second_rc}"
        for row in rows[: count.value]:
            if int(row.pid) != int(pid):
                continue
            if int(row.used_gpu_memory) == NVML_VALUE_NOT_AVAILABLE:
                return None, "driver_reported_non_numeric_process_memory"
            return row.used_gpu_memory / BYTES_PER_MIB, None
        return None, "current_pid_not_reported"


class DirectNvmlProbe:
    """直接调用 NVML 的 GPU 探针，不创建外部进程。"""

    backend_name = "direct_nvml"

    def __init__(
        self,
        *,
        include_process_memory: bool = False,
        bindings: NvmlBindings | Any | None = None,
    ) -> None:
        self.include_process_memory = include_process_memory
        self._bindings = bindings or NvmlBindings()

    def sample(self, pid: int) -> dict[str, float | bool | str | None]:
        try:
            result: dict[str, float | bool | str | None] = {
                "available": True,
                "backend": self.backend_name,
                **self._bindings.device_metrics(),
            }
        except NvmlUnavailableError as exc:
            return {
                "available": False,
                "backend": self.backend_name,
                "unavailable_reason": str(exc),
            }
        process_used_mb: float | None = None
        process_reason: str | None = "process_query_disabled"
        if self.include_process_memory:
            process_used_mb, process_reason = self._bindings.process_memory(pid)
        result.update(
            process_used_mb=process_used_mb,
            process_memory_queried=self.include_process_memory,
            process_memory_available=process_used_mb is not None,
            process_memory_unavailable_reason=process_reason,
        )
        return result


@lru_cache(maxsize=2)
def get_direct_nvml_probe(include_process_memory: bool = False) -> DirectNvmlProbe:
    return DirectNvmlProbe(include_process_memory=include_process_memory)
