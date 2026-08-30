"""GPU 计算份额（SM）管理能力：数据契约、MPS 生命周期、客户端探针与 SM-A 入口。

本子包只服务于 ``experiments/elastic_residency`` 的 P1-C / SM-A 实验，
不接入任何在线调度路径。语义边界：

- ``MPS_CAP`` 是 ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` 客户端执行资源上限，
  不是独占 SM 分区，不参与“SM 数之和 <= 物理 SM”的准入判断；
- 只有 ``MPS_STATIC`` / ``GREEN_CONTEXT`` 且真实验证成功后才按空间分区处理；
- ``effective_sm_count`` 只接受客户端 CUDA context 实测，比例推算只能作为请求值；
- 后端不可用或份额无法验证时输出 ``UNSUPPORTED``，禁止静默退回普通 CUDA。

模块分组：

- 数据契约：``sm_profiles``、``sm_share_profile``、``sm_partition_rules``
- MPS 生命周期：``mps_runtime``、``mps_daemon``
- 客户端探针：``sm_client_probe``（子进程入口）、``cuda_exec_affinity``、
  ``sm_client_workload``
- 探针编排：``sm_capability``、``sm_client_process``、``sm_client_runner``、
  ``sm_probe_env``、``sm_probe_records``、``sm_probe_verdict``
- 只读能力探测：``green_context_probe``
"""

from .sm_partition_rules import (
    GpuPartitionCapability,
    GpuPartitionProbeResult,
    assert_exclusive_sm_budget,
    supports_exclusive_sm_budget,
    validate_exclusive_sm_budget,
)
from .sm_profiles import (
    GpuComputeBackend,
    GpuComputeShareCapability,
    SmProfileError,
    backend_capability,
    estimate_requested_sm_count,
)
from .sm_share_profile import GpuShareProfile, unsupported_profile

__all__ = [
    "GpuComputeBackend",
    "GpuComputeShareCapability",
    "GpuPartitionCapability",
    "GpuPartitionProbeResult",
    "GpuShareProfile",
    "SmProfileError",
    "assert_exclusive_sm_budget",
    "backend_capability",
    "estimate_requested_sm_count",
    "supports_exclusive_sm_budget",
    "unsupported_profile",
    "validate_exclusive_sm_budget",
]
