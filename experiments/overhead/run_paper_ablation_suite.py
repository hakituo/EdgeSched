import argparse
import asyncio
import importlib.util
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

_COMPREHENSIVE_MOD = None


def _repo_root() -> str:
    curr = os.path.abspath(__file__)
    while True:
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        if os.path.exists(os.path.join(parent, "core")):
            return parent
        curr = parent
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _legacy_mvp_core_dir(repo_root: str) -> str:
    path = os.path.join(repo_root, "legacy", "mvp_core")
    if os.path.exists(path):
        return path
    curr = os.path.abspath(__file__)
    while True:
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        if os.path.basename(parent) == "mvp_core":
            return parent
        curr = parent
    return path


def _results_dir(repo_root: str) -> str:
    return os.path.join(_legacy_mvp_core_dir(repo_root), "experiment_results")


def _add_paths(repo_root: str) -> None:
    legacy_mvp_core = _legacy_mvp_core_dir(repo_root)
    legacy_root = os.path.dirname(legacy_mvp_core)
    legacy_experiments = os.path.join(legacy_mvp_core, "experiments")
    for p in (repo_root, legacy_root, legacy_mvp_core, legacy_experiments):
        if p not in sys.path:
            sys.path.append(p)


def _load_comprehensive_experiment_module(repo_root: str):
    global _COMPREHENSIVE_MOD
    if _COMPREHENSIVE_MOD is not None:
        return _COMPREHENSIVE_MOD
    legacy_mvp = _legacy_mvp_core_dir(repo_root)
    module_path = os.path.join(legacy_mvp, "experiments", "overhead", "comprehensive_experiment.py")
    spec = importlib.util.spec_from_file_location("legacy_comprehensive_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment script: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _COMPREHENSIVE_MOD = mod
    return _COMPREHENSIVE_MOD


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


class _BackplaneOverride:
    def __init__(self, disable: bool):
        self.disable = disable
        self._orig = None

    def __enter__(self):
        if not self.disable:
            return self
        import mvp_core.services.task_scheduler as task_scheduler

        self._orig = getattr(task_scheduler, "get_state_backplane", None)
        task_scheduler.get_state_backplane = lambda: None
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.disable:
            return False
        import mvp_core.services.task_scheduler as task_scheduler

        if self._orig is not None:
            task_scheduler.get_state_backplane = self._orig
        return False


def _configure_scheduler(use_cpp: bool, use_cpp_image: bool) -> None:
    from mvp_core.config import get_settings

    settings = get_settings()
    try:
        settings.scheduler.backend = "cpp" if use_cpp else "python"
        settings.scheduler.use_cpp = bool(use_cpp)
        settings.scheduler.use_cpp_for_llm = bool(use_cpp)
        settings.scheduler.llm_backend = "python"
        settings.scheduler.preload_llm = False
        settings.scheduler.use_cpp_for_image = bool(use_cpp) and bool(use_cpp_image)
    except Exception:
        pass


async def _run_once(mode: str, workload: str, use_cpp: bool, use_cpp_image: bool, concurrencies: List[int], n_requests: int, disable_backplane: bool) -> Dict[str, Any]:
    repo_root = _repo_root()
    _add_paths(repo_root)
    mod = _load_comprehensive_experiment_module(repo_root)
    ExperimentContext = getattr(mod, "ExperimentContext")
    ExperimentRunner = getattr(mod, "ExperimentRunner")

    _configure_scheduler(use_cpp=use_cpp, use_cpp_image=use_cpp_image)

    ctx = ExperimentContext(mode, workload)
    runner = ExperimentRunner(ctx)
    await ctx.setup()
    try:
        await runner.run_experiment_1_concurrency(concurrencies=concurrencies)
        await runner.run_experiment_2_blocking()
        await runner.run_experiment_3_distribution(n_requests=int(n_requests))
        results = dict(runner.results)
    finally:
        await ctx.teardown()
    return results


def _aggregate_exp1(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_c = {}
    for run in runs:
        for entry in run.get("exp1", []) or []:
            c = int(entry.get("concurrency", 0))
            bucket = by_c.setdefault(c, {"rps": [], "avg_ms": [], "p95_ms": [], "p99_ms": []})
            bucket["rps"].append(float(entry.get("rps", 0.0)))
            metrics = entry.get("metrics", {}) or {}
            bucket["avg_ms"].append(float(metrics.get("avg_ms", 0.0)))
            bucket["p95_ms"].append(float(metrics.get("p95_ms", 0.0)))
            bucket["p99_ms"].append(float(metrics.get("p99_ms", 0.0)))

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    results = []
    for c in sorted(by_c.keys()):
        bucket = by_c[c]
        results.append(
            {
                "concurrency": c,
                "rps_avg": avg(bucket["rps"]),
                "avg_ms": avg(bucket["avg_ms"]),
                "p95_ms": avg(bucket["p95_ms"]),
                "p99_ms": avg(bucket["p99_ms"]),
            }
        )
    return results


def _aggregate_exp3_metrics(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    values = {"avg_ms": [], "p50_ms": [], "p95_ms": [], "p99_ms": []}
    for run in runs:
        metrics = run.get("exp3_metrics", {}) or {}
        for k in values.keys():
            if k in metrics:
                values[k].append(float(metrics.get(k, 0.0)))

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {k: avg(v) for k, v in values.items()}


def _aggregate_exp2_metrics(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    values = {"avg_lag_ms": [], "max_lag_ms": []}
    for run in runs:
        metrics = run.get("exp2", {}) or {}
        values["avg_lag_ms"].append(float(metrics.get("avg_lag", 0.0)))
        values["max_lag_ms"].append(float(metrics.get("max_lag", 0.0)))

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {k: avg(v) for k, v in values.items()}


async def run_suite(workload: str, concurrencies: List[int], n_requests: int, repeats: int, output_path: str) -> Dict[str, Any]:
    scenarios = [
        {
            "name": "baseline_python",
            "use_cpp": False,
            "disable_backplane": False,
            "use_cpp_image": False,
        },
        {
            "name": "cpp_scheduler",
            "use_cpp": True,
            "disable_backplane": False,
            "use_cpp_image": False,
        },
        {
            "name": "cpp_scheduler_backplane_off",
            "use_cpp": True,
            "disable_backplane": True,
            "use_cpp_image": False,
        },
    ]

    payload = {
        "name": "paper_ablation_suite",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workload": workload,
        "concurrencies": concurrencies,
        "n_requests": int(n_requests),
        "repeats": int(repeats),
        "scenarios": [],
    }

    for scenario in scenarios:
        runs = []
        actual_use_cpp = bool(scenario["use_cpp"])
        actual_use_cpp_image = bool(scenario["use_cpp_image"])
        if workload == "mock":
            actual_use_cpp = False
            actual_use_cpp_image = False
        for _ in range(int(repeats)):
            with _BackplaneOverride(disable=scenario["disable_backplane"]):
                result = await _run_once(
                    mode="xy_core",
                    workload=workload,
                    use_cpp=bool(actual_use_cpp),
                    use_cpp_image=bool(actual_use_cpp_image),
                    concurrencies=concurrencies,
                    n_requests=int(n_requests),
                    disable_backplane=bool(scenario["disable_backplane"]),
                )
            runs.append(result)

        payload["scenarios"].append(
            {
                "name": scenario["name"],
                "use_cpp": bool(actual_use_cpp),
                "disable_backplane": bool(scenario["disable_backplane"]),
                "exp1_aggregate": _aggregate_exp1(runs),
                "exp2_metrics_aggregate": _aggregate_exp2_metrics(runs),
                "exp3_metrics_aggregate": _aggregate_exp3_metrics(runs),
                "paper_experiments": {
                    "exp1_resource_isolation": _aggregate_exp1(runs),
                    "exp2_event_loop": _aggregate_exp2_metrics(runs),
                    "exp3_end_to_end": _aggregate_exp3_metrics(runs),
                },
                "runs": runs,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _parse_concurrencies(raw: str) -> List[int]:
    if not raw:
        return [1, 2, 4]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    repo_root = _repo_root()
    _add_paths(repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="mock", choices=["mock", "real"])
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--n_requests", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    concurrencies = _parse_concurrencies(args.concurrencies)
    _set_seed(int(args.seed) if args.seed is not None else None)

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(_results_dir(repo_root), "paper_ablation_results.json")

    asyncio.run(
        run_suite(
            workload=str(args.workload),
            concurrencies=concurrencies,
            n_requests=int(args.n_requests),
            repeats=int(args.repeats),
            output_path=output_path,
        )
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
