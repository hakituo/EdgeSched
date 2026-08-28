import os
import sys
import time
import json
import struct
import mmap
import random
import statistics
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mvp_core.services.nvram_manager import NVRAMManager

ITERATIONS = 5000  # Number of write operations
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

def benchmark_json_io():
    """Simulate persistence using standard JSON file I/O"""
    json_path = os.path.join(DATA_DIR, "bench_state.json")
    
    # Initial state
    state = {
        "bio": {"dopamine": 0.5, "energy": 0.5, "mood": 0.5},
        "tasks": []
    }
    
    latencies = []
    
    start_total = time.time()
    for i in range(ITERATIONS):
        # Update state
        state["bio"]["dopamine"] = random.random()
        state["tasks"] = [{"id": f"task_{i}", "status": 2}]
        
        t0 = time.perf_counter()
        with open(json_path, "w") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno()) # Ensure write to disk (fair comparison to mmap flush)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000) # ms
        
    return latencies

def benchmark_nvram_mmap():
    """Simulate persistence using NVRAM Manager (mmap)"""
    nvram_path = os.path.join(DATA_DIR, "bench_nvram.dat")
    # Clean up
    if os.path.exists(nvram_path):
        os.remove(nvram_path)
        
    nvram = NVRAMManager(nvram_path)
    latencies = []
    
    for i in range(ITERATIONS):
        dopamine = random.random()
        
        t0 = time.perf_counter()
        # Direct memory update
        nvram.update_bio_state(dopamine, 0.5, 0.5)
        # Assuming flush happens periodically or via OS, but for "Safety Critical" we can force sync
        # In mmap, writes are memory-speed. The OS handles syncing.
        # To be strictly fair with os.fsync, we should call mmap.flush()
        # But NVRAM advantage IS that you don't always need to block on IO.
        # We will test the "Application Perceived Latency"
        nvram._mmap.flush() 
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000) # ms
        
    nvram.close()
    return latencies

def run_benchmark():
    print(f"Running Benchmark: {ITERATIONS} Iterations")
    print("-" * 50)
    
    print("1. Benchmarking JSON (OS fsync)...")
    json_times = benchmark_json_io()
    avg_json = statistics.mean(json_times)
    p99_json = statistics.quantiles(json_times, n=100)[98]
    print(f"   Avg: {avg_json:.4f} ms | P99: {p99_json:.4f} ms")
    
    print("2. Benchmarking NVRAM (mmap flush)...")
    nvram_times = benchmark_nvram_mmap()
    avg_nvram = statistics.mean(nvram_times)
    p99_nvram = statistics.quantiles(nvram_times, n=100)[98]
    print(f"   Avg: {avg_nvram:.4f} ms | P99: {p99_nvram:.4f} ms")
    
    speedup = avg_json / avg_nvram if avg_nvram > 0 else 0
    print("-" * 50)
    print(f"Speedup Factor: {speedup:.2f}x")
    
    # Simple ASCII Plot
    print("\nLatency Distribution (ms):")
    print(f"JSON  : {'#' * int(avg_json * 2)} ({avg_json:.2f}ms)")
    print(f"NVRAM : {'#' * int(avg_nvram * 2)} (<{avg_nvram:.2f}ms)")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    run_benchmark()
