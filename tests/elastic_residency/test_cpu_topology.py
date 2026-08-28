from pathlib import Path

from experiments.elastic_residency.cpu_topology import (
    CPUTopology,
    PhysicalCoreAllocator,
)


def test_allocator_excludes_smt_siblings_and_returns_cores() -> None:
    topology = CPUTopology.from_sibling_groups(
        [(0, 1), (2, 3), (4, 5), (6, 7)]
    )
    allocator = PhysicalCoreAllocator(topology)
    system = allocator.reserve_system(1)
    worker = allocator.acquire("worker", 2, use_smt=False)

    system_logical = {
        cpu for core in system.physical_cores for cpu in core.logical_cpus
    }
    assert not system_logical.intersection(worker.affinity)
    assert len(worker.affinity) == 2
    assert allocator.snapshot()["free_physical_core_count"] == 1

    worker.release()
    system.release()
    restored = allocator.snapshot()
    assert restored["active_lease_count"] == 0
    assert restored["free_physical_core_count"] == 4


def test_linux_sysfs_discovery_groups_siblings(tmp_path: Path) -> None:
    for cpu_id, core_id in ((0, 0), (1, 0), (2, 1), (3, 1)):
        topology = tmp_path / f"cpu{cpu_id}" / "topology"
        topology.mkdir(parents=True)
        (topology / "core_id").write_text(str(core_id))
        (topology / "physical_package_id").write_text("0")
    discovered = CPUTopology.discover(tmp_path)
    assert discovered.source == "linux_sysfs"
    assert discovered.to_dict()["sibling_groups"] == [[0, 1], [2, 3]]
