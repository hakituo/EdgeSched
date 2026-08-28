"""
EdgeSched 论文实验 - 不需要 Forge 服务的实验

Forge 服务不需要启动，这些实验独立运行，不依赖图像生成 API。
包含: lru_thrashing, crash_recovery, stress_test, baseline_comparison

注意: lru_thrashing 虽然涉及图像模型，但通过直接加载 safetensors 文件实现，
不需要 Forge 进程。确保 Forge 已关闭以释放显存。

用法:
  python experiments/run_no_forge_experiments.py --iterations 5
  python experiments/run_no_forge_experiments.py --experiments lru_thrashing,crash_recovery --iterations 5
"""

import argparse
import gc
import json
import os
import sys
import time
import subprocess
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework import setup_paths, MVP_CORE_DIR, RESULTS_DIR

PAPER_DATA_DIR = MVP_CORE_DIR / "paper" / "experiment_data"

NO_FORGE_EXPERIMENTS = [
    "lru_thrashing",
    "crash_recovery",
    "stress_test",
    "baseline_comparison",
]


def get_env_vars() -> Dict[str, str]:
    env = os.environ.copy()
    env["XIAOYOU_LLM_MODEL_PATH"] = env.get(
        "XIAOYOU_LLM_MODEL_PATH",
        str(MVP_CORE_DIR / "models" / "llm" / "qwen2.5-7b-instruct-q4_k_m.gguf"),
    )
    build_bin = str(MVP_CORE_DIR / "cpp_scheduler" / "build" / "bin")
    existing = env.get("LD_LIBRARY_PATH", "")
    if build_bin not in existing:
        env["LD_LIBRARY_PATH"] = build_bin + (":" + existing if existing else "")
    return env


def ensure_output_dirs():
    for subdir in ["01_paper_main", "02_auxiliary", "03_raw_iterations"]:
        (PAPER_DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def save_raw_iteration(experiment_name: str, iteration: int, data: Any):
    raw_dir = PAPER_DATA_DIR / "03_raw_iterations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filepath = raw_dir / f"{experiment_name}_iter{iteration}.json"
    if isinstance(data, dict):
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **data}
    else:
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "data": data}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  原始迭代数据已保存: {filepath}")


def save_aggregated(experiment_name: str, data: Any, subdir: str = "01_paper_main"):
    agg_dir = PAPER_DATA_DIR / subdir
    agg_dir.mkdir(parents=True, exist_ok=True)
    filepath = agg_dir / f"{experiment_name}.json"
    if isinstance(data, dict):
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **data}
    else:
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "data": data}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  聚合数据已保存: {filepath}")


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


def _force_inter_experiment_cleanup():
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


def _aggregate_lru_results(all_iterations: list, cache_mode: str) -> Dict:
    success_iters = [it for it in all_iterations if it.get("status") == "success"]
    timeout_iters = [it for it in all_iterations if "Timeout" in it.get("status", "")]

    aggregated = {
        "cache_mode": cache_mode,
        "iterations": len(all_iterations),
        "success_count": len(success_iters),
        "timeout_count": len(timeout_iters),
    }

    for key in ["llm_only_latency_s", "image_only_latency_s",
                "concurrent_llm_latency_s", "concurrent_image_latency_s"]:
        values = [it.get(key, 0) for it in success_iters if it.get(key, 0) > 0]
        if values:
            aggregated[key] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
                "n_valid": len(values),
            }

    if success_iters:
        llm_only = aggregated.get("llm_only_latency_s", {}).get("mean", 0)
        conc_llm = aggregated.get("concurrent_llm_latency_s", {}).get("mean", 0)
        if llm_only > 0 and conc_llm > 0:
            aggregated["thrashing_ratio"] = {
                "concurrent_llm_vs_llm_only": round(conc_llm / llm_only, 1),
            }

    if timeout_iters:
        aggregated["status"] = "mixed"
        aggregated["status_detail"] = f"{len(timeout_iters)}/{len(all_iterations)} iterations timed out"
    else:
        aggregated["status"] = "success"

    return aggregated


def run_lru_experiment(iterations: int, env: Dict[str, str]) -> Dict:
    all_results = {}

    for cache_mode in ["cold", "warm"]:
        print(f"\n{'#'*60}")
        print(f"# LRU Thrashing - {cache_mode.upper()} 模式")
        print(f"{'#'*60}")

        mode_iterations = []
        for i in range(iterations):
            print(f"\n{'='*60}")
            print(f"  [lru_thrashing/{cache_mode}] 迭代 {i+1}/{iterations}")
            print(f"{'='*60}")

            cmd = [sys.executable, "experiments/baselines/lru_thrashing_experiment.py", "--cache-mode", cache_mode]
            run_subprocess(cmd, env, timeout=300)

            result_path = RESULTS_DIR / f"baseline_lru_thrashing_{cache_mode}_results.json"
            if result_path.exists():
                with open(result_path, "r") as f:
                    data = json.load(f)
                data["cache_mode"] = cache_mode
                data["iteration_index"] = i + 1
                mode_iterations.append(data)
                save_raw_iteration(f"lru_thrashing_{cache_mode}", i + 1, data)
            else:
                print(f"  警告: 未找到结果文件 {result_path}")

            if i < iterations - 1:
                _force_inter_experiment_cleanup()

        if mode_iterations:
            aggregated = _aggregate_lru_results(mode_iterations, cache_mode)
            save_aggregated(f"baseline_lru_thrashing_{cache_mode}_results", aggregated)
            all_results[cache_mode] = aggregated

    if all_results:
        combined = {
            "cold_start": all_results.get("cold", {}),
            "warm_cache": all_results.get("warm", {}),
            "iterations": iterations,
        }
        save_aggregated("baseline_lru_thrashing_results", combined)
        return combined
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
    actual_iters = min(iterations, 1)
    if iterations > 1:
        print(f"  注意: Mock stress test 是确定性测试，只需跑 1 次（请求 {iterations} 次已自动缩减为 1）")

    all_iterations = []
    for i in range(actual_iters):
        print(f"\n{'='*60}")
        print(f"  [stress_test/mock] 迭代 {i+1}/{actual_iters}")
        print(f"{'='*60}")

        cmd = [sys.executable, "experiments/stress/large_scale_stress_test.py"]
        run_subprocess(cmd, env, timeout=1800)

        result_path = RESULTS_DIR / "supplementary" / "large_scale_stress_test.json"
        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            all_iterations.append(data)
            save_raw_iteration("stress_test_mock", i + 1, data)
        else:
            print(f"  警告: 未找到结果文件")

    if all_iterations:
        save_aggregated("large_scale_stress_test", all_iterations[-1])
        return all_iterations[-1]
    return {}


def check_forge_not_running():
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:7860/sdapi/v1/sd-models", timeout=3)
        if resp.status == 200:
            print("  ⚠️  检测到 Forge 服务正在运行！")
            print("  建议: 关闭 Forge 释放显存，避免影响实验结果。")
            print("  如果确定要继续运行，请忽略此警告。")
            return False
    except Exception:
        pass
    print("  ✅ Forge 服务未运行 (显存可独占使用)")
    return True


def main():
    parser = argparse.ArgumentParser(description="EdgeSched 论文实验 - 不需要 Forge 服务的实验")
    parser.add_argument(
        "--experiments", type=str, default="all",
        help=f"要运行的实验，逗号分隔。可选: {', '.join(NO_FORGE_EXPERIMENTS + ['all'])}"
    )
    parser.add_argument("--iterations", type=int, default=5, help="每个实验的迭代次数 (默认5)")

    args = parser.parse_args()

    if args.experiments == "all":
        selected = list(NO_FORGE_EXPERIMENTS)
    else:
        selected = [e.strip() for e in args.experiments.split(",")]

    for exp in selected:
        if exp not in NO_FORGE_EXPERIMENTS:
            print(f"错误: 未知实验 '{exp}'，可选: {NO_FORGE_EXPERIMENTS}")
            sys.exit(1)

    print("=" * 60)
    print("EdgeSched 论文实验 - 不需要 Forge 服务")
    print("=" * 60)
    print(f"选中实验: {selected}")
    print(f"迭代次数: {args.iterations}")
    print(f"输出目录: {PAPER_DATA_DIR}")
    print("=" * 60)

    print("\n检查 Forge 服务状态...")
    check_forge_not_running()

    env = get_env_vars()
    setup_paths()
    ensure_output_dirs()

    results_summary = {}
    start_time = time.time()

    for exp in selected:
        print(f"\n{'#'*60}")
        print(f"# 开始实验: {exp}")
        print(f"{'#'*60}")

        exp_start = time.time()
        try:
            if exp == "lru_thrashing":
                result = run_lru_experiment(args.iterations, env)
            elif exp == "crash_recovery":
                result = run_crash_recovery_experiment(args.iterations, env)
            elif exp == "stress_test":
                result = run_stress_test_experiment(args.iterations, env)
            elif exp == "baseline_comparison":
                result = run_baseline_comparison_experiment(args.iterations, env)
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
    manifest_path = PAPER_DATA_DIR / "run_manifest_no_forge.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"运行清单已保存: {manifest_path}")


if __name__ == "__main__":
    main()
