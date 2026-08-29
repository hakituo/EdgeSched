"""弹性驻留实验的共享基础设施。"""

from .contracts import (
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentEvent,
    ExperimentRunConfig,
    FailureRecord,
    TimelineSample,
)
from .cpu_topology import (
    CPUTopology,
    PhysicalCoreAllocator,
    process_affinity_scope,
)
from .gpu_telemetry import DirectNvmlProbe, NvmlUnavailableError
from .resource_sampler import EventLoopJitterProbe, ResourceSampler

__all__ = [
    "BackendEvidence",
    "CPUTopology",
    "DirectNvmlProbe",
    "EventLoopJitterProbe",
    "ExperimentArtifactWriter",
    "ExperimentEvent",
    "ExperimentRunConfig",
    "FailureRecord",
    "NvmlUnavailableError",
    "PhysicalCoreAllocator",
    "ResourceSampler",
    "TimelineSample",
    "process_affinity_scope",
]
