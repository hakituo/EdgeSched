"""物理核、SMT sibling 与可归还 affinity 租约。"""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import psutil


@dataclass(frozen=True, order=True)
class PhysicalCore:
    package_id: int
    core_id: int
    logical_cpus: tuple[int, ...]

    @property
    def representative(self) -> int:
        return self.logical_cpus[0]

    @property
    def key(self) -> tuple[int, int]:
        return self.package_id, self.core_id


@dataclass(frozen=True)
class CPUTopology:
    physical_cores: tuple[PhysicalCore, ...]
    logical_cpu_count: int
    source: str

    @classmethod
    def discover(
        cls,
        sysfs_root: str | Path = "/sys/devices/system/cpu",
    ) -> "CPUTopology":
        root = Path(sysfs_root)
        cores: list[PhysicalCore] = []
        groups: dict[tuple[int, int], list[int]] = {}
        for cpu_dir in sorted(root.glob("cpu[0-9]*")):
            try:
                cpu_id = int(cpu_dir.name[3:])
                topology_dir = cpu_dir / "topology"
                core_id = int((topology_dir / "core_id").read_text().strip())
                package_id = int(
                    (topology_dir / "physical_package_id").read_text().strip()
                )
            except (FileNotFoundError, ValueError):
                continue
            groups.setdefault((package_id, core_id), []).append(cpu_id)
        if groups:
            for (package_id, core_id), logical_cpus in sorted(groups.items()):
                cores.append(
                    PhysicalCore(
                        package_id=package_id,
                        core_id=core_id,
                        logical_cpus=tuple(sorted(logical_cpus)),
                    )
                )
            return cls(
                physical_cores=tuple(cores),
                logical_cpu_count=os.cpu_count() or len(cores),
                source="linux_sysfs",
            )
        logical_count = os.cpu_count() or 1
        return cls.from_sibling_groups(
            [(cpu_id,) for cpu_id in range(logical_count)],
            source="logical_fallback",
        )

    @classmethod
    def from_sibling_groups(
        cls,
        sibling_groups: Sequence[Sequence[int]],
        *,
        source: str = "injected",
    ) -> "CPUTopology":
        cores = tuple(
            PhysicalCore(0, core_id, tuple(sorted(int(cpu) for cpu in cpus)))
            for core_id, cpus in enumerate(sibling_groups)
        )
        if any(not core.logical_cpus for core in cores):
            raise ValueError("每个物理核至少需要一个逻辑 CPU")
        all_cpus = [cpu for core in cores for cpu in core.logical_cpus]
        if len(all_cpus) != len(set(all_cpus)):
            raise ValueError("逻辑 CPU 不能属于多个物理核")
        return cls(
            physical_cores=cores,
            logical_cpu_count=len(all_cpus),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "logical_cpu_count": self.logical_cpu_count,
            "physical_core_count": len(self.physical_cores),
            "physical_core_representatives": [
                core.representative for core in self.physical_cores
            ],
            "sibling_groups": [
                list(core.logical_cpus) for core in self.physical_cores
            ],
        }


@dataclass
class CoreLease:
    lease_id: str
    purpose: str
    physical_cores: tuple[PhysicalCore, ...]
    affinity: tuple[int, ...]
    use_smt: bool
    _allocator: "PhysicalCoreAllocator"
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self._allocator.release(self.lease_id)
            self.released = True

    def __enter__(self) -> "CoreLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class PhysicalCoreAllocator:
    """按物理核分配互斥租约，释放后立即归还。"""

    def __init__(self, topology: CPUTopology) -> None:
        if not topology.physical_cores:
            raise ValueError("没有可分配的物理核")
        self.topology = topology
        self._leases: dict[str, CoreLease] = {}
        self._lock = threading.Lock()

    def reserve_system(self, physical_cores: int) -> CoreLease:
        return self.acquire(
            purpose="system_audio_scheduler",
            physical_cores=physical_cores,
            use_smt=True,
            prefer_high_ids=True,
        )

    def acquire(
        self,
        purpose: str,
        physical_cores: int,
        *,
        use_smt: bool = False,
        prefer_high_ids: bool = False,
    ) -> CoreLease:
        if not purpose.strip():
            raise ValueError("purpose 不能为空")
        if physical_cores < 0:
            raise ValueError("physical_cores 不能为负数")
        with self._lock:
            occupied = {
                core.key
                for lease in self._leases.values()
                for core in lease.physical_cores
                if not lease.released
            }
            available = [
                core
                for core in self.topology.physical_cores
                if core.key not in occupied
            ]
            if prefer_high_ids:
                available.reverse()
            if len(available) < physical_cores:
                raise RuntimeError(
                    f"请求 {physical_cores} 个物理核，但只剩 {len(available)} 个"
                )
            selected = tuple(available[:physical_cores])
            if use_smt:
                affinity = tuple(
                    cpu for core in selected for cpu in core.logical_cpus
                )
            else:
                affinity = tuple(core.representative for core in selected)
            lease_id = f"core_{uuid.uuid4().hex[:12]}"
            lease = CoreLease(
                lease_id=lease_id,
                purpose=purpose,
                physical_cores=selected,
                affinity=tuple(sorted(affinity)),
                use_smt=use_smt,
                _allocator=self,
            )
            self._leases[lease_id] = lease
            return lease

    def release(self, lease_id: str) -> None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise KeyError(f"未知核心租约: {lease_id}")
            lease.released = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = [lease for lease in self._leases.values() if not lease.released]
            occupied = {
                core.key for lease in active for core in lease.physical_cores
            }
            return {
                "physical_core_count": len(self.topology.physical_cores),
                "active_lease_count": len(active),
                "free_physical_core_count": (
                    len(self.topology.physical_cores) - len(occupied)
                ),
                "leases": [
                    {
                        "lease_id": lease.lease_id,
                        "purpose": lease.purpose,
                        "physical_core_keys": [
                            list(core.key) for core in lease.physical_cores
                        ],
                        "affinity": list(lease.affinity),
                        "use_smt": lease.use_smt,
                    }
                    for lease in active
                ],
            }


@contextmanager
def process_affinity_scope(
    cpu_ids: Sequence[int],
    *,
    pid: int | None = None,
) -> Iterator[tuple[int, ...]]:
    """临时绑定进程，退出时恢复原 affinity。"""

    requested = tuple(sorted(set(int(cpu_id) for cpu_id in cpu_ids)))
    if not requested:
        raise ValueError("affinity 至少需要一个逻辑 CPU")
    process = psutil.Process(pid or os.getpid())
    if not hasattr(process, "cpu_affinity"):
        raise RuntimeError("当前平台不支持进程 affinity")
    previous = tuple(process.cpu_affinity())
    process.cpu_affinity(list(requested))
    try:
        actual = tuple(process.cpu_affinity())
        if set(actual) != set(requested):
            raise RuntimeError(
                f"affinity 未生效: requested={requested}, actual={actual}"
            )
        yield actual
    finally:
        process.cpu_affinity(list(previous))
