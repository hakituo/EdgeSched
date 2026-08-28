import asyncio
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Setup directories
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "..", "experiment_results")
os.makedirs(results_dir, exist_ok=True)

def heavy_task(duration):
    """Simulates a CPU-bound task (e.g., model loading/inference)"""
    end_time = time.time() + duration
    while time.time() < end_time:
        _ = 1000 * 1000  # Burn CPU

async def measure_lag(duration, workload_func):
    """
    Measures Event Loop Lag while running a workload.
    """
    lags = []
    start_time = time.time()
    
    # 1. Start Monitor
    async def monitor():
        while time.time() - start_time < duration:
            t0 = time.time()
            await asyncio.sleep(0.05) # Target 50ms heartbeat
            t1 = time.time()
            lag = (t1 - t0 - 0.05) * 1000 # Lag in ms
            lags.append((t1 - start_time, lag))
    
    monitor_task = asyncio.create_task(monitor())
    
    # 2. Run Workload after 1s
    await asyncio.sleep(1.0)
    await workload_func()
    
    # 3. Wait for finish
    await asyncio.sleep(1.0)
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
        
    return lags

async def run_blocking_scenario():
    print("Running Blocking Scenario (Main Thread)...")
    async def blocking_workload():
        # Block main thread for 2 seconds
        heavy_task(2.0)
        
    return await measure_lag(4.0, blocking_workload)

async def run_non_blocking_scenario():
    print("Running Non-Blocking Scenario (Offloaded)...")
    executor = ThreadPoolExecutor(max_workers=1)
    
    async def non_blocking_workload():
        # Offload to thread
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, heavy_task, 2.0)
        
    return await measure_lag(4.0, non_blocking_workload)

def generate_blocking_chart():
    print("Starting Real Blocking Analysis...")
    
    # Run Scenarios
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    lags_blocking = loop.run_until_complete(run_blocking_scenario())
    lags_non_blocking = loop.run_until_complete(run_non_blocking_scenario())
    
    loop.close()
    
    # Extract Data
    t_b, y_b = zip(*lags_blocking) if lags_blocking else ([], [])
    t_nb, y_nb = zip(*lags_non_blocking) if lags_non_blocking else ([], [])
    
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    
    # Plot Python line (Red) - Blocking
    ax.plot(t_b, y_b, color='#d62728', label='Python Scheduler (GIL Blocking)', linewidth=2.2)
    
    # Plot C++ line (Green) - Non-Blocking
    ax.plot(t_nb, y_nb, color='#2ca02c', label='C++ Scheduler (Non-Blocking)', linewidth=2.2)
    
    # Annotations
    ax.axvspan(1.0, 3.0, color='0.7', alpha=0.12, zorder=0)
    
    # Find max lag for annotation
    max_lag_b = max(y_b) if y_b else 0
    max_lag_b_t = 0.0
    if y_b:
        idx_b = int(np.argmax(np.asarray(y_b)))
        max_lag_b_t = float(t_b[idx_b])
    ax.annotate(
        f'Blocking Spike (~{int(max_lag_b)}ms)',
        xy=(max_lag_b_t, max_lag_b),
        xytext=(-190, 10),
        textcoords='offset points',
        ha='left',
        va='bottom',
        arrowprops=dict(arrowstyle='-|>', lw=1.0, color='0.2'),
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.95)
    )
    
    max_lag_nb = max(y_nb) if y_nb else 0
    max_lag_nb_t = 0.0
    if y_nb:
        idx_nb = int(np.argmax(np.asarray(y_nb)))
        max_lag_nb_t = float(t_nb[idx_nb])
    ax.annotate(
        f'Offloaded (~{int(max_lag_nb)}ms)',
        xy=(max_lag_nb_t, max_lag_nb),
        xytext=(10, 14),
        textcoords='offset points',
        ha='left',
        va='bottom',
        arrowprops=dict(arrowstyle='-|>', lw=1.0, color='0.2'),
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8', alpha=0.95)
    )

    # Formatting
    ax.set_title('Main Thread Responsiveness Comparison (Event Loop Lag)', fontsize=14, pad=12)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Event Loop Lag (ms)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.set_axisbelow(True)
    
    # Adjust Y limit dynamically but keep some headroom
    ax.set_ylim(-50, max(2500, max_lag_b + 500))
    ax.set_xlim(0, 4.05)
    y_top = ax.get_ylim()[1]
    ax.text(
        2.0,
        y_top * 0.92,
        'Model Loading Window',
        ha='center',
        va='center',
        fontsize=11,
        color='0.25',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.85', alpha=0.75)
    )
    
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.95,
    )
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    
    # Save
    out_pdf = os.path.join(results_dir, "main_thread_blocking.pdf")
    out_png = os.path.join(results_dir, "main_thread_blocking.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    plt.close(fig)

if __name__ == "__main__":
    generate_blocking_chart()
