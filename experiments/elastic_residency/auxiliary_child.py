"""单配置真实 CPU 测量、线程亲和验证和租约释放。"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import psutil

from .auxiliary_inputs import load_inputs
from .auxiliary_report import cached_evidence_valid
from .auxiliary_workloads import WORKLOADS, output_digest
from .contracts import ExperimentArtifactWriter, ExperimentRunConfig
from .cpu_topology import CPUTopology, PhysicalCoreAllocator, process_affinity_scope
from .resource_sampler import ResourceSampler, _percentile


class CpuOnlyProbe:
    backend_name = "not_applicable_cpu_only"

    def sample(self, pid: int) -> dict:
        return {"available": False, "backend": self.backend_name}


def allowed_topology() -> CPUTopology:
    topology = CPUTopology.discover()
    if topology.source != "linux_sysfs" or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("真实物理核画像必须使用 Linux sysfs；拒绝逻辑核回退")
    allowed = set(os.sched_getaffinity(0))
    cores = tuple(core for core in topology.physical_cores
                  if set(core.logical_cpus).issubset(allowed))
    if not cores:
        raise RuntimeError("没有完整可用的物理核/SMT sibling 组")
    return CPUTopology(cores, sum(len(c.logical_cpus) for c in cores), topology.source)


def verify_threads(worker: tuple, sampler: ResourceSampler, system: tuple) -> dict:
    monitor_ids = {t.native_id for t in (sampler._thread, sampler._gpu_thread) if t}
    masks = {}
    valid = True
    for thread in psutil.Process().threads():
        try:
            actual = sorted(os.sched_getaffinity(thread.id))
        except ProcessLookupError:
            continue
        expected = set(system if thread.id in monitor_ids else worker)
        valid &= set(actual) == expected
        masks[str(thread.id)] = actual
    return {"passed": bool(valid and masks), "thread_masks": masks}


def finish_cpu_window(sampler: ResourceSampler, end_ns: int, timeout_s: float = 2) -> None:
    """短任务也必须有有效 CPU 差分点和测量后的样本；等待不计入任务耗时。"""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if (len(sampler.samples) >= 2
                and sampler.samples[-1].monotonic_ns >= end_ns
                and sampler.samples[-1].cpu_sample_valid):
            return
        time.sleep(.01)
    raise RuntimeError("CPU 时间线未形成测量后有效样本")


def run_child(config: dict, output: Path) -> dict:
    topology = allowed_topology()
    allocator = PhysicalCoreAllocator(topology)
    initial = sorted(os.sched_getaffinity(0))
    writer = ExperimentArtifactWriter(output.parent, output.name)
    writer.initialize(ExperimentRunConfig("auxiliary_core_budget_child", output.name),
                      {"cpu_topology": topology.to_dict(), "parameters": config})
    result = {**config["cell"], "events": [], "warmup_events": [],
              "measurement_valid": False, "error": None, "pid": os.getpid()}
    sampler = None
    system = worker = None
    swap_start = psutil.swap_memory().used
    try:
        data, corpus = load_inputs(Path(config["inputs"]))
        system = allocator.reserve_system(config["cell"]["system_cores"])
        worker = allocator.acquire("auxiliary", config["cell"]["cores"], use_smt=False)
        monitor_affinity = system.affinity or worker.affinity
        result["lease_during"] = allocator.snapshot()
        result["monitor_shares_worker"] = not bool(system.affinity)
        with process_affinity_scope(worker.affinity):
            sampler = ResourceSampler(interval_s=.05, gpu_interval_s=.25,
                                      writer=writer, gpu_probe=CpuOnlyProbe(),
                                      thread_affinity=monitor_affinity)
            sampler.start()
            task = config["cell"]["task"]
            start = time.perf_counter()
            workload = WORKLOADS[task](corpus=corpus, cores=config["cell"]["cores"],
                                       model=Path(config["models"].get(task, ".")),
                                       input_root=Path(config["inputs"]).parent)
            result["load_s"] = time.perf_counter() - start
            prepare = getattr(workload, "prepare", None)
            if prepare is not None:
                start = time.perf_counter()
                result["preparation"] = prepare()
                result["preparation"]["elapsed_s"] = time.perf_counter() - start
            for query in data["queries"]:
                warm = workload.run(query)
                if not warm["output_valid"]:
                    raise RuntimeError("预热输出无效")
                result["warmup_events"].append(output_digest(warm))
            for index, query in enumerate(data["queries"]):
                affinity_before = verify_threads(worker.affinity, sampler, monitor_affinity)
                before = psutil.Process().cpu_times()
                start = time.perf_counter()
                value = workload.run(query)
                elapsed = time.perf_counter() - start
                after = psutil.Process().cpu_times()
                event = {"input_index": index, "elapsed_s": elapsed,
                         "cpu_time_s": after.user + after.system - before.user - before.system,
                         "output": value, "output_sha256": output_digest(value),
                         "threads_before": affinity_before,
                         "threads_after": verify_threads(worker.affinity, sampler,
                                                         monitor_affinity)}
                event["average_cpu_cores_used"] = event["cpu_time_s"] / elapsed
                if task == "whisper":
                    event["rtf"] = elapsed / value["audio_duration_s"]
                result["events"].append(event)
                writer.append_iteration(event)
            finish_cpu_window(sampler, time.monotonic_ns())
            del workload
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if sampler:
            try:
                result["resources"] = sampler.stop()
            except Exception:
                result["error"] = (result["error"] or "") + traceback.format_exc()
        if worker:
            worker.release()
        if system:
            system.release()
        result["lease_after"] = allocator.snapshot()
        result["affinity_restored"] = sorted(os.sched_getaffinity(0)) == initial
        swap_values = [swap_start, psutil.swap_memory().used]
        if sampler:
            swap_values.extend(s.system_swap_used_bytes for s in sampler.samples)
            jitter = [s.scheduler_jitter_ms for s in sampler.samples]
            result["jitter"] = {"sample_count": len(jitter),
                                "p99_ms": _percentile(jitter, .99) if len(jitter) >= 100 else None,
                                "max_ms": max(jitter, default=None)}
        result["swap"] = {"start": swap_start, "max": max(swap_values),
                          "end": swap_values[1], "growth_bytes": max(swap_values) - swap_start}
        result["checks"] = {
            "no_error": result["error"] is None,
            "outputs_valid": bool(result["events"]) and all(
                e["output"]["output_valid"] for e in result["events"]),
            "threads_bound": bool(result["events"]) and all(
                e["threads_before"]["passed"] and e["threads_after"]["passed"]
                for e in result["events"]),
            "affinity_restored": result["affinity_restored"],
            "leases_returned": allocator.snapshot()["active_lease_count"] == 0,
            "swap_no_growth": result["swap"]["growth_bytes"] == 0,
            "timeline_present": bool(sampler and len(sampler.samples) >= 2),
        }
        if config["cell"]["task"] == "bge_cached":
            result["checks"]["cached_candidate_contract"] = cached_evidence_valid(result)
        result["measurement_valid"] = all(result["checks"].values())
        writer.write_summary(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_child(json.loads(args.config.read_text(encoding="utf-8")), args.output)
    raise SystemExit(0 if result["measurement_valid"] else 1)
