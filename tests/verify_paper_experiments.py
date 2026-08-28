import json
import os
import subprocess
import sys
import time
import tempfile
import shutil


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


def _mvp_root(repo_root: str) -> str:
    return os.path.join(repo_root, "legacy", "mvp_core")


def _check_payload(payload: dict) -> list[str]:
    errors = []
    if payload.get("name") != "paper_ablation_suite":
        errors.append("name 不匹配")
    scenarios = payload.get("scenarios") or []
    if not scenarios:
        errors.append("scenarios 为空")
    for scenario in scenarios:
        if "exp1_aggregate" not in scenario:
            errors.append(f"{scenario.get('name', 'unknown')} 缺少 exp1_aggregate")
        if "exp2_metrics_aggregate" not in scenario:
            errors.append(f"{scenario.get('name', 'unknown')} 缺少 exp2_metrics_aggregate")
        if "exp3_metrics_aggregate" not in scenario:
            errors.append(f"{scenario.get('name', 'unknown')} 缺少 exp3_metrics_aggregate")
    return errors


def _check_performance_payload(path: str) -> list[str]:
    errors = []
    if not os.path.exists(path):
        return [f"结果文件不存在: {path}"]
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return [f"结果格式不正确: {path}"]
    results = payload.get("results")
    if not isinstance(results, dict):
        errors.append(f"结果缺少 results: {path}")
        return errors
    if "exp1" not in results:
        errors.append(f"结果缺少 exp1: {path}")
    if "exp3_metrics" not in results:
        errors.append(f"结果缺少 exp3_metrics: {path}")
    return errors


def _check_backpressure_payload(path: str) -> list[str]:
    errors = []
    if not os.path.exists(path):
        return [f"结果文件不存在: {path}"]
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("name") != "image_backpressure_microbench":
        errors.append(f"name 不匹配: {path}")
    results = payload.get("results")
    if not isinstance(results, dict):
        errors.append(f"results 为空: {path}")
        return errors
    for key in ("accepted", "rejected", "finished_total"):
        if key not in results:
            errors.append(f"results 缺少 {key}: {path}")
    return errors


def _run_with_live_output(cmd: list[str], timeout_s: int) -> int:
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                print(line.rstrip())
            if proc.poll() is not None:
                break
            if time.time() - start > timeout_s:
                proc.kill()
                return 124
    finally:
        if proc.stdout:
            for line in proc.stdout:
                print(line.rstrip())
    return int(proc.returncode or 0)


def main() -> None:
    repo_root = _repo_root()
    mvp_root = _mvp_root(repo_root)
    experiments_dir = os.path.join(mvp_root, "experiments")
    results_dir = os.path.join(mvp_root, "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    keep_results = str(os.getenv("XIAOYOU_VERIFY_KEEP_RESULTS", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    run_dir = tempfile.mkdtemp(prefix="verify_paper_", dir=results_dir)

    ablation_output_path = os.path.join(run_dir, "paper_ablation_results_verify.json")
    ablation_script_path = os.path.join(
        experiments_dir,
        "overhead",
        "run_paper_ablation_suite.py",
    )

    cmd = [
        sys.executable,
        ablation_script_path,
        "--workload",
        "mock",
        "--concurrencies",
        "1,2",
        "--n_requests",
        "3",
        "--repeats",
        "1",
        "--output",
        ablation_output_path,
    ]

    print("开始执行论文实验验证...")
    exit_code = _run_with_live_output(cmd, timeout_s=180)
    if exit_code != 0:
        raise SystemExit(1)

    if not os.path.exists(ablation_output_path):
        raise SystemExit(1)

    with open(ablation_output_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    errors = _check_payload(payload)
    if errors:
        for err in errors:
            print(err)
        raise SystemExit(1)

    perf_results_dir = os.path.join(run_dir, "verify_performance")
    perf_script_path = os.path.join(experiments_dir, "run_performance_suite.py")
    perf_cmd = [
        sys.executable,
        perf_script_path,
        "--workload",
        "mock",
        "--results_dir",
        perf_results_dir,
        "--concurrency",
        "1,2",
        "--exp3_n",
        "3",
        "--stability_duration",
        "3",
        "--stability_workers",
        "1",
        "--backpressure_n",
        "6",
    ]
    exit_code = _run_with_live_output(perf_cmd, timeout_s=180)
    if exit_code != 0:
        raise SystemExit(1)

    perf_errors = []
    bp_path = os.path.join(perf_results_dir, "image_backpressure_mock.json")
    py_path = os.path.join(perf_results_dir, "xy_core_mock.json")
    cpp_path = os.path.join(perf_results_dir, "xy_core_cpp_sched_mock.json")
    perf_errors.extend(_check_backpressure_payload(bp_path))
    perf_errors.extend(_check_performance_payload(py_path))
    perf_errors.extend(_check_performance_payload(cpp_path))
    if perf_errors:
        for err in perf_errors:
            print(err)
        raise SystemExit(1)

    if keep_results:
        print({"ok": True, "output": ablation_output_path, "perf_dir": perf_results_dir, "run_dir": run_dir})
        return

    print({"ok": True})
    try:
        shutil.rmtree(run_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
