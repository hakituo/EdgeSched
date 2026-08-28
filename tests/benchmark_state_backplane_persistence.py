import os
import sys
import time
import json
import random
import statistics


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mvp_core.services.state_backplane import StateBackplane


ITERATIONS = 5000
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))


def benchmark_json_io():
    json_path = os.path.join(DATA_DIR, "bench_state.json")

    state = {
        "tasks": [],
    }

    latencies = []

    for i in range(ITERATIONS):
        state["bio"]["dopamine"] = random.random()
        state["tasks"] = [{"id": f"task_{i}", "status": 2}]

        t0 = time.perf_counter()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    return latencies


def benchmark_backplane_mmap():
    backplane_path = os.path.join(DATA_DIR, "bench_state_backplane.dat")
    if os.path.exists(backplane_path):
        os.remove(backplane_path)

    backplane = StateBackplane(backplane_path)
    latencies = []
    task_id = "bench_task_0000"
    backplane.register_task(task_id, priority=1, status=2)

    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        backplane.update_task_status(task_id, 2 if (i % 2 == 0) else 3)
        backplane._mmap.flush()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

    backplane.close()
    return latencies


def run_benchmark():
    print(f"Running Benchmark: {ITERATIONS} Iterations")
    print("-" * 50)

    print("1. Benchmarking JSON (OS fsync)...")
    json_times = benchmark_json_io()
    avg_json = statistics.mean(json_times)
    p99_json = statistics.quantiles(json_times, n=100)[98]
    print(f"   Avg: {avg_json:.4f} ms | P99: {p99_json:.4f} ms")

    print("2. Benchmarking State Backplane (mmap flush)...")
    backplane_times = benchmark_backplane_mmap()
    avg_backplane = statistics.mean(backplane_times)
    p99_backplane = statistics.quantiles(backplane_times, n=100)[98]
    print(f"   Avg: {avg_backplane:.4f} ms | P99: {p99_backplane:.4f} ms")

    speedup = avg_json / avg_backplane if avg_backplane > 0 else 0
    print("-" * 50)
    print(f"Speedup Factor: {speedup:.2f}x")

    print("\nLatency Distribution (ms):")
    print(f"JSON       : {'#' * int(avg_json * 2)} ({avg_json:.2f}ms)")
    print(f"Backplane  : {'#' * int(avg_backplane * 2)} ({avg_backplane:.2f}ms)")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    run_benchmark()
