import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
import subprocess
import threading
import csv

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
    module_path = os.path.join(
        legacy_mvp,
        "experiments",
        "overhead",
        "comprehensive_experiment.py",
    )
    spec = importlib.util.spec_from_file_location("legacy_comprehensive_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment script: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _COMPREHENSIVE_MOD = mod
    return _COMPREHENSIVE_MOD

# --- VRAM Sampler ---
class VRAMSampler:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.samples = []
        self.running = False
        self.start_time = 0.0
        self.thread = None

    def _read_vram_gb(self) -> float | None:
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            ).strip()
            first_line = out.splitlines()[0].strip()
            return float(first_line) / 1024.0
        except Exception:
            return None

    def _loop(self):
        while self.running:
            mem = self._read_vram_gb()
            if mem is not None:
                self.samples.append((time.time() - self.start_time, mem))
            time.sleep(0.5)

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.samples = []
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        
        # Save to CSV
        if self.samples:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t_s", "mem_gb"])
                for t_s, mem_gb in self.samples:
                    writer.writerow([f"{float(t_s):.3f}", f"{float(mem_gb):.6f}"])
            print(f"Saved VRAM samples to {self.output_path}")

# --- Helper Metrics ---
def _calc_metrics_seconds(samples_s: list[float]) -> dict:
    if not samples_s:
        return {"n": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "std_dev": 0.0}
    
    import statistics
    xs = sorted(float(x) for x in samples_s)
    n = len(xs)
    avg = sum(xs) / n
    std_dev = statistics.stdev(xs) if n > 1 else 0.0

    def pick(p: float) -> float:
        if n == 1:
            return xs[0]
        # 使用线性插值计算更精准的分位数
        idx = (n - 1) * p
        low = int(idx)
        high = min(n - 1, low + 1)
        weight = idx - low
        return xs[low] * (1 - weight) + xs[high] * weight

    return {
        "n": n,
        "avg_ms": avg * 1000.0,
        "p50_ms": pick(0.50) * 1000.0,
        "p95_ms": pick(0.95) * 1000.0,
        "p99_ms": pick(0.99) * 1000.0,
        "min_ms": xs[0] * 1000.0,
        "max_ms": xs[-1] * 1000.0,
        "std_dev_ms": std_dev * 1000.0,
        "tail_latency_ratio": (pick(0.99) / avg) if avg > 0 else 1.0 # 长尾系数
    }

# --- Experiment Functions ---
async def run_stability_deadline(runner, duration_s: int, workers: int) -> dict:
    start = time.time()
    deadline = start + float(duration_s)
    completed_in_window = 0
    completed_after_deadline = 0
    errors = 0
    latencies_in_window: list[float] = []
    completed_total = 0

    async def worker_loop(wid: int):
        nonlocal completed_in_window, completed_after_deadline, errors, completed_total
        while True:
            if time.time() >= deadline:
                return
            try:
                dur = await runner._run_mixed_task(wid)
                t1 = time.time()
                completed_total += 1
                if t1 <= deadline:
                    completed_in_window += 1
                    latencies_in_window.append(float(dur))
                else:
                    completed_after_deadline += 1
            except Exception:
                errors += 1

    await asyncio.gather(*[worker_loop(i) for i in range(int(workers))])
    end = time.time()
    wall_time = max(0.001, end - start)
    metrics = _calc_metrics_seconds(latencies_in_window)
    return {
        "duration_s": int(duration_s),
        "wall_time_s": float(wall_time),
        "workers": int(workers),
        "completed_in_window": int(completed_in_window),
        "completed_after_deadline": int(completed_after_deadline),
        "completed_total": int(completed_total),
        "errors": int(errors),
        "throughput_in_window": float(completed_in_window) / float(duration_s) if duration_s > 0 else 0.0,
        "throughput_wall": float(completed_total) / float(wall_time),
        "metrics": metrics,
    }

async def run_image_backpressure_microbench(repo_root: str, out_path: str, n: int = 20):
    from mvp_core.config import get_settings
    from mvp_core.presentation.websocket.handler import WebSocketHandler

    settings = get_settings()
    scheduler_settings = getattr(settings, "scheduler", None)
    if scheduler_settings:
        setattr(scheduler_settings, "image_queue_capacity", 2)
        setattr(scheduler_settings, "image_max_concurrent_tasks", 1)

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

    handler = WebSocketHandler(chat_service=None)  # type: ignore
    await handler._ensure_image_workers_started()

    async def slow_run_image_generation(websocket, request_id: str, prompt: str):
        await handler._safe_send_json(websocket, {"type": "image_status", "request_id": request_id, "data": {"prompt": prompt, "status": "started"}})
        await asyncio.sleep(2.0)
        await handler._safe_send_json(websocket, {"type": "image_status", "request_id": request_id, "data": {"prompt": prompt, "status": "finished"}})

    handler._run_image_generation = slow_run_image_generation
    ws = FakeWebSocket()
    request_id = "bp_test"
    prompts = [f"p{i}" for i in range(int(n))]

    start = time.time()
    results = await asyncio.gather(*[handler._try_enqueue_image_generation(ws, request_id, p) for p in prompts], return_exceptions=True)

    accepted_prompts = []
    rejected = 0
    errors = 0
    for prompt, r in zip(prompts, results):
        if isinstance(r, Exception):
            errors += 1
        elif r:
            accepted_prompts.append(prompt)
        else:
            rejected += 1

    target_finished = len(accepted_prompts)
    
    def finished_prompts() -> set[str]:
        done = set()
        for m in ws.sent:
            if isinstance(m, dict) and m.get("type") == "image_status":
                data = m.get("data", {})
                if data.get("status") == "finished":
                    done.add(data.get("prompt"))
        return done

    timeout_s = max(5.0, 2.5 * float(target_finished))
    while True:
        if len([p for p in accepted_prompts if p in finished_prompts()]) >= target_finished:
            break
        if time.time() - start >= timeout_s:
            break
        await asyncio.sleep(0.05)

    elapsed_s = time.time() - start
    done_all = finished_prompts()
    finished_accepted = sum(1 for p in accepted_prompts if p in done_all)
    finished_rejected = sum(1 for p in prompts if p not in accepted_prompts and p in done_all)

    payload = {
        "name": "image_backpressure_microbench",
        "results": {
            "accepted": len(accepted_prompts),
            "rejected": rejected,
            "errors": errors,
            "finished_accepted": finished_accepted,
            "finished_rejected": finished_rejected,
            "finished_total": len(done_all),
            "elapsed_s": elapsed_s,
        },
    }

    try:
        tasks = list(getattr(handler, "_image_workers", set()) or [])
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved Backpressure to {out_path}")

async def _run_one(mode: str, workload: str, use_cpp_scheduler: bool, use_cpp_image: bool, concurrencies: list[int], exp3_n: int, stability_duration: int, stability_workers: int, out_path: str):
    from mvp_core.config import get_settings
    repo_root = _repo_root()
    mod = _load_comprehensive_experiment_module(repo_root)
    ExperimentContext = getattr(mod, "ExperimentContext")
    ExperimentRunner = getattr(mod, "ExperimentRunner")

    settings = get_settings()
    try:
        settings.scheduler.backend = "cpp" if use_cpp_scheduler else "python"
        settings.scheduler.use_cpp = bool(use_cpp_scheduler)
        settings.scheduler.use_cpp_for_llm = bool(use_cpp_scheduler)
        settings.scheduler.llm_backend = "python"
        settings.scheduler.preload_llm = False
        settings.scheduler.use_cpp_for_image = bool(use_cpp_scheduler) and bool(use_cpp_image)
        settings.scheduler.image_engine_type = "SD15_TURBO"
    except Exception:
        pass

    ctx = ExperimentContext(mode, workload)
    runner = ExperimentRunner(ctx)
    await ctx.setup()
    try:
        print(f"Running Exp1 (Concurrency: {concurrencies})...")
        await runner.run_experiment_1_concurrency(concurrencies=concurrencies)
        print("Running Exp2 (Blocking)...")
        await runner.run_experiment_2_blocking()
        print(f"Running Exp3 (Distribution n={exp3_n})...")
        await runner.run_experiment_3_distribution(n_requests=int(exp3_n))
        print(f"Running Exp4 (Stability {stability_duration}s)...")
        stability = await run_stability_deadline(runner, duration_s=int(stability_duration), workers=int(stability_workers))
        runner.results["exp4"] = stability

        payload = {
            "mode": mode,
            "workload": workload,
            "adapters": {"tts_mode": getattr(ctx, "tts_mode", "unknown")},
            "scheduler": {
                "use_cpp_scheduler": bool(use_cpp_scheduler),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "results": runner.results,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {out_path}")
    finally:
        await ctx.teardown()

async def _run_trace_capture(results_dir: str):
    print("Running Trace Capture (1 request)...")
    repo_root = _repo_root()
    mod = _load_comprehensive_experiment_module(repo_root)
    ExperimentContext = getattr(mod, "ExperimentContext")
    ExperimentRunner = getattr(mod, "ExperimentRunner")
    
    # Configure for C++ Scheduler to get "Hybrid" trace
    from mvp_core.config import get_settings
    settings = get_settings()
    try:
        settings.scheduler.backend = "cpp"
        settings.scheduler.use_cpp = True
        settings.scheduler.use_cpp_for_llm = True
        settings.scheduler.llm_backend = "python"
    except Exception:
        pass

    ctx = ExperimentContext("xy_core", "real")
    runner = ExperimentRunner(ctx)
    await ctx.setup()
    
    vram_path = os.path.join(results_dir, "vram_usage_over_time.csv")
    sampler = VRAMSampler(vram_path)
    sampler.start()
    
    try:
        # Run just 1 request to capture trace
        await runner.run_experiment_3_distribution(n_requests=1)
    finally:
        sampler.stop()
        await ctx.teardown()

    # --- Log Parsing & Trace Generation ---
    import glob
    
    log_dir = os.path.join(repo_root, "logs")
    log_files = glob.glob(os.path.join(log_dir, "benchmark_trace_*.log"))
    
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Parsing log for trace: {latest_log}")
        
        trace_events = []
        def extract_ts(line):
            try:
                ts_str = line.split(']')[0].strip('[')
                if '.' in ts_str:
                    base, ms = ts_str.split('.')
                    dt = time.mktime(time.strptime(base, "%Y-%m-%d %H:%M:%S"))
                    return dt + float(f"0.{ms}")
                return time.mktime(time.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
            except Exception:
                return time.time()

        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                if "executing task" in line:
                    trace_events.append({"ts": extract_ts(line), "event": "task_started", "task_id": line.split("executing task")[-1].strip(), "details": line.strip()})
                elif "Task" in line and "completed" in line:
                    trace_events.append({"ts": extract_ts(line), "event": "task_completed", "task_id": line.split("Task")[-1].split("completed")[0].strip(), "details": line.strip()})
                elif "scheduled - ID:" in line:
                    trace_events.append({"ts": extract_ts(line), "event": "task_scheduled", "task_id": line.split("ID:")[-1].split(",")[0].strip(), "details": line.strip()})
        
        if trace_events:
            events_path = os.path.join(results_dir, "scheduler_trace_events.json")
            with open(events_path, "w", encoding="utf-8") as f:
                json.dump(trace_events, f, indent=2, ensure_ascii=False)
            print(f"Generated {len(trace_events)} trace events.")
            
            # Call Visualization Scripts
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from visualize_trace import visualize_trace
                from generate_memory_plot import generate_memory_plot
                
                visualize_trace(events_path, os.path.join(results_dir, "scheduler_trace_timeline.png"))
                visualize_trace(events_path, os.path.join(results_dir, "scheduler_trace_timeline.pdf"))
                generate_memory_plot(events_path, os.path.join(results_dir, "memory_usage_over_time.pdf"))
                print("Generated Trace & Memory plots.")
            except Exception as e:
                print(f"Visualization failed: {e}")

async def main_async(args):
    repo_root = _repo_root()
    _add_paths(repo_root)
    results_dir = args.results_dir or _results_dir(repo_root)
    os.makedirs(results_dir, exist_ok=True)

    concurrencies = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    workload = args.workload

    # 1. Backpressure
    bp_out = os.path.join(results_dir, f"image_backpressure_{workload}.json")
    await run_image_backpressure_microbench(repo_root, bp_out, n=int(args.backpressure_n))

    if args.only_backpressure:
        return

    # 2. Python Scheduler
    py_out = os.path.join(results_dir, f"xy_core_{workload}.json")
    await _run_one("xy_core", workload, False, False, concurrencies, args.exp3_n, args.stability_duration, args.stability_workers, py_out)

    # 3. C++ Scheduler
    cpp_out = os.path.join(results_dir, f"xy_core_cpp_sched_{workload}.json")
    await _run_one("xy_core", workload, True, False, concurrencies, args.exp3_n, args.stability_duration, args.stability_workers, cpp_out)

    # 4. Trace & VRAM (Only in Real mode)
    if workload == "real":
        await _run_trace_capture(results_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="real", choices=["real", "mock"])
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--concurrency", default="1,5,10")
    parser.add_argument("--exp3_n", type=int, default=20)
    parser.add_argument("--stability_duration", type=int, default=15)
    parser.add_argument("--stability_workers", type=int, default=3)
    parser.add_argument("--backpressure_n", type=int, default=20)
    parser.add_argument("--only_backpressure", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))

if __name__ == "__main__":
    sys.exit(main())
