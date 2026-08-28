"""
EdgeSched 论文实验统一运行脚本

支持所有实验的运行、多次迭代（N=5）、原始迭代数据保存和聚合统计。
输出结构对齐 paper/experiment_data/ 目录：
  01_paper_main/     - 聚合后的论文主数据
  02_auxiliary/      - 辅助数据
  03_raw_iterations/ - 原始每次迭代数据

用法：
  python experiments/run_all_experiments.py --experiments all --iterations 5
  python experiments/run_all_experiments.py --experiments migration --iterations 5
  python experiments/run_all_experiments.py --experiments cold_start,lru,crash_recovery --iterations 1
"""

import asyncio
import argparse
import gc
import json
import os
import sys
import time
import shutil
import subprocess
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework import setup_paths, MVP_CORE_DIR, RESULTS_DIR
from experiments.migration.resource_migration_experiment import ResourceMigrationExperiment
from experiments.migration.symmetric_migration_experiment import SymmetricMigrationExperiment

PAPER_DATA_DIR = MVP_CORE_DIR / "paper" / "experiment_data"

ALL_EXPERIMENTS = [
    "edgesched_migration",
    "symmetric_migration",
    "fast_down_naive_up",
    "naive_down_optimized_up",
    "cold_start",
    "lru_thrashing",
    "crash_recovery",
    "stress_test",
    "baseline_comparison",
    "real_workload",
]

EXPERIMENTS_NEED_FORGE = [
    "edgesched_migration",
    "symmetric_migration",
    "fast_down_naive_up",
    "naive_down_optimized_up",
    "real_workload",
]

EXPERIMENTS_NEED_IMAGE_MODEL = [
    "lru_thrashing",
]

EXPERIMENTS_NEED_LLM = [
    "edgesched_migration",
    "symmetric_migration",
    "fast_down_naive_up",
    "naive_down_optimized_up",
    "cold_start",
    "lru_thrashing",
    "real_workload",
]


def get_env_vars() -> Dict[str, str]:
    env = os.environ.copy()
    env["XIAOYOU_LLM_MODEL_PATH"] = env.get(
        "XIAOYOU_LLM_MODEL_PATH",
        str(MVP_CORE_DIR / "models" / "llm" / "qwen2.5-7b-instruct-q4_k_m.gguf"),
    )
    env["XIAOYOU_FORGE_BASE_URL"] = env.get("XIAOYOU_FORGE_BASE_URL", "http://127.0.0.1:7860")
    build_bin = str(MVP_CORE_DIR / "cpp_scheduler" / "build" / "bin")
    existing = env.get("LD_LIBRARY_PATH", "")
    if build_bin not in existing:
        env["LD_LIBRARY_PATH"] = build_bin + (":" + existing if existing else "")
    return env


def ensure_output_dirs():
    for subdir in ["01_paper_main/migration", "02_auxiliary", "03_raw_iterations/migration"]:
        (PAPER_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def save_raw_iteration(experiment_name: str, iteration: int, data: Any):
    raw_dir = PAPER_DATA_DIR / "03_raw_iterations"
    if experiment_name in ["edgesched_migration", "symmetric_migration",
                           "fast_down_naive_up", "naive_down_optimized_up"]:
        raw_dir = raw_dir / "migration"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filepath = raw_dir / f"{experiment_name}_iter{iteration}.json"
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **data}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  原始迭代数据已保存: {filepath}")


def save_aggregated(experiment_name: str, data: Any, subdir: str = "01_paper_main"):
    agg_dir = PAPER_DATA_DIR / subdir
    if experiment_name in ["edgesched_migration", "symmetric_migration",
                           "fast_down_naive_up", "naive_down_optimized_up"]:
        agg_dir = agg_dir / "migration"
    agg_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        "edgesched_migration": "edgesched_aggregated.json",
        "symmetric_migration": "symmetric_aggregated.json",
        "fast_down_naive_up": "fast_down_naive_up_aggregated.json",
        "naive_down_optimized_up": "naive_down_optimized_up_aggregated.json",
    }
    filename = name_map.get(experiment_name, f"{experiment_name}.json")
    filepath = agg_dir / filename
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **data}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  聚合数据已保存: {filepath}")


def aggregate_metrics(all_iterations: List[Dict], metrics_keys: List[str]) -> Dict:
    aggregated = {"metrics": {}, "iterations": len(all_iterations)}
    for key in metrics_keys:
        values = []
        for r in all_iterations:
            v = r.get(key)
            if v is not None and v != 0.0:
                values.append(float(v))
        if values:
            aggregated["metrics"][key] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
                "n_valid": len(values),
                "n_total": len(all_iterations),
            }
    return aggregated


def run_subprocess(cmd: List[str], env: Dict[str, str], timeout: int = 600) -> Optional[Dict]:
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(MVP_CORE_DIR)
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            print(f"  命令超时 ({timeout}s)")
            return None

        if stdout:
            for line in stdout.splitlines()[-50:]:
                print(f"  {line}")
        if stderr:
            for line in stderr.splitlines()[-10:]:
                if "error" in line.lower() or "traceback" in line.lower():
                    print(f"  [STDERR] {line}")
        if proc.returncode != 0:
            print(f"  命令返回非零退出码: {proc.returncode}")
            return None
        return {"returncode": proc.returncode}
    except Exception as e:
        print(f"  命令执行异常: {e}")
        return None


def run_migration_experiment_direct(
    variant: str, iterations: int, llm_requests: int = 10
) -> Dict:
    """直接调用实验类，避免子进程文件覆盖问题"""
    import asyncio

    metrics_keys = [
        "migration_stutter_ms", "coexistence_cpu_ttft_ms", "restore_time_ms",
        "restore_window_max_gap_ms", "llm_gpu_ttft_ms",
    ]

    all_iterations = []
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  [{variant}] 迭代 {i+1}/{iterations}")
        print(f"{'='*60}")

        async def _run_one_iter():
            if variant == "edgesched_migration":
                exp = ResourceMigrationExperiment(
                    llm_requests=llm_requests, warmup_requests=3, seed=42 + i
                )
                await exp.setup()
                return await exp.test_integrated_lifecycle()
            else:
                exp = SymmetricMigrationExperiment(
                    llm_requests=llm_requests, warmup_requests=3, seed=42 + i
                )
                await exp.setup()
                method_map = {
                    "symmetric_migration": "run_symmetric_lifecycle",
                    "fast_down_naive_up": "run_fast_down_naive_up",
                    "naive_down_optimized_up": "run_naive_down_optimized_up",
                }
                return await getattr(exp, method_map[variant])()

        result = asyncio.run(_run_one_iter())

        data = {
            "migration_stutter_ms": result.migration_stutter_ms,
            "coexistence_cpu_ttft_ms": result.coexistence_cpu_ttft_ms,
            "restore_time_ms": result.restore_time_ms,
            "restore_window_max_gap_ms": result.restore_window_max_gap_ms,
            "llm_gpu_ttft_ms": result.llm_gpu_ttft_ms,
            "migration_time_ms": getattr(result, "migration_time_ms", 0.0),
            "failed_restore_count": getattr(result, "failed_restore_count", 0),
            "oom_count": getattr(result, "oom_count", 0),
            "test_passed": getattr(result, "test_passed", False),
            "llm_still_working": getattr(result, "llm_still_working", False),
            "variant": variant,
            "iteration": i + 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        all_iterations.append(data)
        save_raw_iteration(variant, i + 1, data)

        if i < iterations - 1:
            print("  等待系统冷却 (30s)...")
            time.sleep(30)

    aggregated = aggregate_metrics(all_iterations, metrics_keys)
    aggregated["experiment_name"] = variant
    save_aggregated(variant, aggregated)
    return aggregated


def run_cold_start_experiment(iterations: int, env: Dict[str, str]) -> Dict:
    all_iterations = []
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  [cold_start] 迭代 {i+1}/{iterations}")
        print(f"{'='*60}")

        try:
            import psutil
            mem_pct = psutil.virtual_memory().percent
            print(f"  [资源] 系统内存使用: {mem_pct:.0f}%")
        except ImportError:
            pass

        cmd = [sys.executable, "experiments/baselines/cold_start_experiment.py"]
        run_subprocess(cmd, env, timeout=600)

        result_path = RESULTS_DIR / "baseline_cold_start_results.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            all_iterations.append(data)
            save_raw_iteration("cold_start", i + 1, data)
        else:
            print(f"  警告: 未找到结果文件")

        if i < iterations - 1:
            cooldown = 30
            print(f"  等待系统冷却 ({cooldown}s)...")
            time.sleep(cooldown)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

    if iterations == 1:
        save_aggregated("baseline_cold_start_results", all_iterations[0])
        return all_iterations[0]

    context_sizes = ["512", "1024", "2048", "3072"]
    aggregated = {"cold_start": {}, "hot_swap": {}, "iterations": iterations}
    for ctx in context_sizes:
        for mode in ["cold_start", "hot_swap"]:
            if mode in all_iterations[0] and ctx in all_iterations[0][mode]:
                time_key = "total_recovery_time_s" if mode == "cold_start" else "user_interruption_time_s"
                values = []
                for it in all_iterations:
                    if mode in it and ctx in it[mode] and time_key in it[mode][ctx]:
                        values.append(float(it[mode][ctx][time_key]))
                if values:
                    aggregated[mode][ctx] = {
                        time_key: {
                            "mean": statistics.mean(values),
                            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                            "values": values,
                        }
                    }
    save_aggregated("baseline_cold_start_results", aggregated)
    return aggregated


def _force_inter_experiment_cleanup():
    """实验间强制清理：释放 GPU 显存、回收内存、等待系统稳定"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  [清理] RAM 可用: {mem.available / (1024**3):.1f}/{mem.total / (1024**3):.1f}GB")
    except ImportError:
        pass
    try:
        import subprocess
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True
        )
        parts = output.strip().split(",")
        print(f"  [清理] VRAM: {parts[0].strip()}/{parts[1].strip()} MB")
    except Exception:
        pass
    print("  [清理] 等待系统稳定 (15s)...")
    time.sleep(15)


def run_lru_experiment(iterations: int, env: Dict[str, str]) -> Dict:
    all_iterations = []
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  [lru_thrashing] 迭代 {i+1}/{iterations}")
        print(f"{'='*60}")

        cmd = [sys.executable, "experiments/baselines/lru_thrashing_experiment.py"]
        run_subprocess(cmd, env, timeout=300)

        result_path = RESULTS_DIR / "baseline_lru_thrashing_results.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            all_iterations.append(data)
            save_raw_iteration("lru_thrashing", i + 1, data)
        else:
            print(f"  警告: 未找到结果文件")

        if i < iterations - 1:
            _force_inter_experiment_cleanup()

    if all_iterations:
        save_aggregated("baseline_lru_thrashing_results", all_iterations[-1])
        return all_iterations[-1]
    return {}


def run_crash_recovery_experiment(iterations: int, env: Dict[str, str]) -> Dict:
    all_iterations = []
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  [crash_recovery] 迭代 {i+1}/{iterations}")
        print(f"{'='*60}")

        cmd = [sys.executable, "experiments/recovery/crash_recovery_experiment.py", "--comparison"]
        run_subprocess(cmd, env, timeout=120)

        result_path = RESULTS_DIR / "supplementary" / "crash_recovery_comparison.json"
        if not result_path.exists():
            for candidate in RESULTS_DIR.rglob("crash_recovery_comparison*.json"):
                result_path = candidate
                break

        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            all_iterations.append(data)
            save_raw_iteration("crash_recovery", i + 1, data)
        else:
            print(f"  警告: 未找到结果文件")

        if i < iterations - 1:
            time.sleep(10)

    if all_iterations:
        save_aggregated("crash_recovery_comparison", all_iterations[-1])
        return all_iterations[-1]
    return {}


def run_baseline_comparison_experiment(iterations: int, env: Dict[str, str]) -> Dict:
    all_iterations = []
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  [baseline_comparison] 迭代 {i+1}/{iterations}")
        print(f"{'='*60}")

        cmd = [sys.executable, "experiments/overhead/baseline_comparison.py"]
        run_subprocess(cmd, env, timeout=300)

        result_path = RESULTS_DIR / "supplementary" / "baseline_comparison.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            all_iterations.append(data)
            save_raw_iteration("baseline_comparison", i + 1, data)
        else:
            print(f"  警告: 未找到结果文件")

        if i < iterations - 1:
            time.sleep(10)

    if all_iterations:
        save_aggregated("baseline_comparison", all_iterations[-1], subdir="02_auxiliary")
        return all_iterations[-1]
    return {}


def run_stress_test_experiment(iterations: int, env: Dict[str, str]) -> Dict:
    all_iterations = []
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  [stress_test] 迭代 {i+1}/{iterations}")
        print(f"{'='*60}")

        cmd = [sys.executable, "experiments/stress/large_scale_stress_test.py"]
        run_subprocess(cmd, env, timeout=1800)

        result_path = RESULTS_DIR / "supplementary" / "large_scale_stress_test.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            all_iterations.append(data)
            save_raw_iteration("stress_test", i + 1, data)
        else:
            print(f"  警告: 未找到结果文件")

        if i < iterations - 1:
            time.sleep(30)

    if all_iterations:
        save_aggregated("large_scale_stress_test", all_iterations[-1])
        return all_iterations[-1]
    return {}


def run_real_workload_experiment(iterations: int, env: Dict[str, str]) -> Dict:
    all_iterations = []
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"  [real_workload] 迭代 {i+1}/{iterations}")
        print(f"{'='*60}")

        cmd = [sys.executable, "experiments/workload/real_workload_experiment.py"]
        run_subprocess(cmd, env, timeout=600)

        result_path = RESULTS_DIR / "real_workload_trace.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            all_iterations.append(data)
            save_raw_iteration("real_workload", i + 1, data)
        else:
            print(f"  警告: 未找到结果文件")

        if i < iterations - 1:
            time.sleep(30)

    if all_iterations:
        save_aggregated("real_workload_trace", all_iterations[-1])
        return all_iterations[-1]
    return {}


def check_forge_available() -> bool:
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:7860/sdapi/v1/sd-models", timeout=5)
        return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="EdgeSched 论文实验统一运行脚本")
    parser.add_argument(
        "--experiments", type=str, default="all",
        help=f"要运行的实验，逗号分隔。可选: {', '.join(ALL_EXPERIMENTS + ['all', 'migration', 'core'])}"
    )
    parser.add_argument("--iterations", type=int, default=5, help="每个实验的迭代次数 (默认5)")
    parser.add_argument("--llm-requests", type=int, default=10, help="LLM 请求数 (默认10)")
    parser.add_argument("--skip-forge-check", action="store_true", help="跳过 Forge 可用性检查")

    args = parser.parse_args()

    if args.experiments == "all":
        selected = list(ALL_EXPERIMENTS)
    elif args.experiments == "migration":
        selected = ["edgesched_migration", "symmetric_migration",
                    "fast_down_naive_up", "naive_down_optimized_up"]
    elif args.experiments == "core":
        selected = ["edgesched_migration", "cold_start", "crash_recovery"]
    else:
        selected = [e.strip() for e in args.experiments.split(",")]

    for exp in selected:
        if exp not in ALL_EXPERIMENTS:
            print(f"错误: 未知实验 '{exp}'，可选: {ALL_EXPERIMENTS}")
            sys.exit(1)

    print("=" * 60)
    print("EdgeSched 论文实验统一运行脚本")
    print("=" * 60)
    print(f"选中实验: {selected}")
    print(f"迭代次数: {args.iterations}")
    print(f"输出目录: {PAPER_DATA_DIR}")
    print("=" * 60)

    env = get_env_vars()
    setup_paths()
    ensure_output_dirs()

    forge_needed = any(e in EXPERIMENTS_NEED_FORGE for e in selected)
    if forge_needed and not args.skip_forge_check:
        print("\n检查 Forge 服务...")
        if check_forge_available():
            print("  ✅ Forge 服务可用 (http://127.0.0.1:7860)")
        else:
            print("  ⚠️  Forge 服务不可用！需要 Forge 的实验将回退到 Mock 模式。")
            print("  启动命令: wsl -d Ubuntu-24.04-E -e bash -c 'cd /home/leslie/mvp_core/stable-diffusion-webui-forge-main && source /home/leslie/.venvs/forge-cu128/bin/activate && python launch.py --api --listen --port 7860 --skip-version-check --skip-python-version-check'")

    results_summary = {}
    start_time = time.time()

    for exp in selected:
        print(f"\n{'#'*60}")
        print(f"# 开始实验: {exp}")
        print(f"{'#'*60}")

        exp_start = time.time()
        try:
            if exp in ["edgesched_migration", "symmetric_migration", "fast_down_naive_up", "naive_down_optimized_up"]:
                result = run_migration_experiment_direct(exp, args.iterations, args.llm_requests)
            elif exp == "cold_start":
                result = run_cold_start_experiment(args.iterations, env)
            elif exp == "lru_thrashing":
                result = run_lru_experiment(args.iterations, env)
            elif exp == "crash_recovery":
                result = run_crash_recovery_experiment(args.iterations, env)
            elif exp == "stress_test":
                result = run_stress_test_experiment(args.iterations, env)
            elif exp == "baseline_comparison":
                result = run_baseline_comparison_experiment(args.iterations, env)
            elif exp == "real_workload":
                result = run_real_workload_experiment(args.iterations, env)
            else:
                result = {"error": f"未知实验: {exp}"}

            exp_elapsed = time.time() - exp_start
            results_summary[exp] = {
                "status": "completed",
                "elapsed_s": round(exp_elapsed, 1),
            }
            print(f"\n  ✅ {exp} 完成 (耗时 {exp_elapsed:.1f}s)")

            _force_inter_experiment_cleanup()

        except Exception as e:
            import traceback
            exp_elapsed = time.time() - exp_start
            results_summary[exp] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "elapsed_s": round(exp_elapsed, 1),
            }
            print(f"\n  ❌ {exp} 失败: {e}")
            traceback.print_exc()

    total_elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("实验运行总结")
    print("=" * 60)
    for exp, info in results_summary.items():
        status = "✅" if info["status"] == "completed" else "❌"
        print(f"  {status} {exp}: {info['elapsed_s']}s")
        if "error" in info:
            print(f"     错误: {info['error']}")
    print(f"\n总耗时: {total_elapsed:.1f}s")
    print(f"数据输出目录: {PAPER_DATA_DIR}")

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_elapsed_s": round(total_elapsed, 1),
        "iterations": args.iterations,
        "results": results_summary,
    }
    manifest_path = PAPER_DATA_DIR / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"运行清单已保存: {manifest_path}")


if __name__ == "__main__":
    main()
