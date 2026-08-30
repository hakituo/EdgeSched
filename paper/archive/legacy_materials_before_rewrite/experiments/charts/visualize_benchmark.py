import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def _infer_mode_workload_from_filename(base_name: str):
    name = str(base_name or "")
    stem = name[:-5] if name.lower().endswith(".json") else name
    parts = [p for p in stem.split("_") if p]
    if not parts:
        return "unknown", "unknown"

    workload = "unknown"
    if parts[-1] in ("mock", "real"):
        workload = parts[-1]
        parts = parts[:-1]

    mode = "_".join(parts) if parts else "unknown"
    if mode.startswith("xy_core"):
        mode = "xy_core"
    return mode, workload


def _extract_meta(data: dict, base_name: str) -> dict:
    meta = data.get("meta") if isinstance(data, dict) else None
    if isinstance(meta, dict):
        return meta
    return data if isinstance(data, dict) else {}

def load_results(results_dir):
    """Load all *.json files."""
    pattern = os.path.join(results_dir, "*.json")
    files = glob.glob(pattern)
    
    data_list = []
    kept_by_key = {}
    preferred_names = {
        "single_thread_real.json": 1,
        "naive_async_real.json": 1,
        "xy_core_real.json": 1,
        "xy_core_cpp_image_real.json": 1,
        "xy_core_cpp_real.json": 1,
        "naive_mock.json": 1,
        "xy_core_mock.json": 1,
    }
    print(f"Found {len(files)} result files in {results_dir}")
    
    for f in files:
        base_name = os.path.basename(f)
        if base_name.startswith("verify_"):
            continue
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
                
                # Skip error files
                if 'error' in data or 'results' not in data:
                    print(f"Skipping {base_name} (contains error or no results)")
                    continue

                meta = _extract_meta(data, base_name)
                inferred_mode, inferred_workload = _infer_mode_workload_from_filename(base_name)

                mode = meta.get('mode') or inferred_mode
                workload = meta.get('workload') or inferred_workload
                scheduler_cfg = meta.get('scheduler', {}) or {}
                use_cpp_scheduler = bool(scheduler_cfg.get('use_cpp_scheduler'))
                use_cpp_for_llm = bool(scheduler_cfg.get('use_cpp_for_llm'))
                use_cpp_for_image = bool(scheduler_cfg.get('use_cpp_for_image'))
                llm_backend = str(scheduler_cfg.get('llm_backend', 'python') or 'python').lower()
                
                if base_name == "baseline_serial.json":
                    label = "Baseline (Serial)"
                    order = 1
                elif base_name == "naive_async.json":
                    label = "Naive Async"
                    order = 2
                elif base_name == "xy_core_python.json":
                    label = "Xy-Core (Python)"
                    order = 3
                elif base_name == "xy_core_cpp.json":
                    label = "Xy-Core (C++ Hybrid)"
                    order = 4
                elif mode == 'naive_async' and workload == 'mock':
                    label = "Traditional Async (Mock)"
                    order = 0
                elif mode == 'single_thread' and workload == 'real':
                    label = "Serial Baseline (Real)"
                    order = 1
                elif mode == 'naive_async' and workload == 'real':
                    label = "Traditional Async (Real)"
                    order = 2
                elif mode == 'xy_core' and workload == 'real':
                    if use_cpp_scheduler and use_cpp_for_llm and llm_backend == 'cpp':
                        label = "xy-core (Real, C++ Hybrid)"
                        order = 3
                    elif use_cpp_scheduler and use_cpp_for_image:
                        label = "xy-core (Real, C++ Image)"
                        order = 4
                    else:
                        label = "xy-core (Real, Python)"
                        order = 3
                elif mode == 'xy_core' and workload == 'mock':
                    if use_cpp_scheduler and use_cpp_for_llm and llm_backend == 'cpp':
                        label = "xy-core (Mock, C++ Hybrid)"
                        order = 10
                    elif use_cpp_scheduler and use_cpp_for_image:
                        label = "xy-core (Mock, C++ Image)"
                        order = 11
                    else:
                        label = "xy-core (Mock, Python)"
                else:
                    label = f"{mode} ({workload})"
                    order = 99
                
                data['label'] = label
                data['order'] = order
                data['mode'] = mode
                data['workload'] = workload
                data['scheduler'] = scheduler_cfg

                key = (
                    mode,
                    workload,
                    use_cpp_scheduler,
                    use_cpp_for_llm,
                    use_cpp_for_image,
                    llm_backend,
                )
                existing = kept_by_key.get(key)
                pr = preferred_names.get(base_name, 0)
                if existing is not None:
                    existing_pr = preferred_names.get(existing.get("_source_name", ""), 0)
                    if pr <= existing_pr:
                        continue

                data["_source_name"] = base_name
                kept_by_key[key] = data
                print(f"  Loaded: {label} from {base_name}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    data_list = list(kept_by_key.values())
            
    # Sort by defined order
    data_list.sort(key=lambda x: x.get('order', 99))
    return data_list

def plot_exp1_concurrency(data_list, output_dir, prefix: str = ""):
    """Plot RPS vs Concurrency for Exp 1."""
    if not any('exp1' in d['results'] for d in data_list):
        return

    plt.figure(figsize=(10, 6))
    
    for d in data_list:
        res = d['results'].get('exp1')
        if not res: continue
        
        # Sort by concurrency
        res.sort(key=lambda x: x['concurrency'])
        x = [r['concurrency'] for r in res]
        y = [r['rps'] for r in res]
        
        plt.plot(x, y, marker='o', label=d['label'])
        
    plt.xlabel('Concurrency (Tasks)')
    plt.ylabel('Throughput (RPS)')
    plt.title('Experiment 1: System Throughput vs Concurrency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    out_path = os.path.join(output_dir, f'{prefix}exp1_concurrency_rps.pdf')
    plt.savefig(out_path)
    plt.savefig(out_path.replace('.pdf', '.png'))
    print(f"Saved {out_path}")
    plt.close()

def plot_exp2_blocking(data_list, output_dir, prefix: str = ""):
    """Plot Max Blocking Lag for Exp 2."""
    if not any('exp2' in d['results'] for d in data_list):
        return

    labels = []
    max_lags = []
    avg_lags = []
    
    for d in data_list:
        res = d['results'].get('exp2')
        if not res: continue
        
        labels.append(d['label'])
        max_lags.append(float(res.get('max_lag', 0) or 0))
        avg_lags.append(float(res.get('avg_lag', 0) or 0))
        
    if not labels: return

    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, max_lags, width, label='Max Lag')
    rects2 = ax.bar(x + width/2, avg_lags, width, label='Avg Lag')
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Experiment 2: Main Thread Blocking Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}exp2_blocking_latency.pdf')
    plt.savefig(out_path)
    plt.savefig(out_path.replace('.pdf', '.png'))
    print(f"Saved {out_path}")
    plt.close()

def plot_exp3_distribution(data_list, output_dir, prefix: str = ""):
    """Plot Latency Distribution for Exp 3 with P99 markers."""
    if not any('exp3' in d['results'] for d in data_list):
        return
        
    plt.figure(figsize=(12, 7))
    
    # Use boxplot for distribution comparison
    plot_data = []
    labels = []
    p99_values = []
    
    for d in data_list:
        res = d['results'].get('exp3')
        if not res: continue
        
        # Convert to ms
        latencies_ms = sorted([l * 1000 for l in res])
        if not latencies_ms: continue
        
        plot_data.append(latencies_ms)
        labels.append(d['label'])
        
        # Calculate P99 manually for visual marker
        idx = int(len(latencies_ms) * 0.99)
        idx = max(0, min(len(latencies_ms) - 1, idx))
        p99_values.append(latencies_ms[idx])
        
    if not plot_data: return
    
    # Boxplot shows Median, Q1, Q3
    bp = plt.boxplot(plot_data, tick_labels=labels, patch_artist=True)
    
    # Add P99 dots and labels
    for i, p99 in enumerate(p99_values):
        plt.plot(i + 1, p99, 'ro', markersize=8, label='P99 Latency' if i == 0 else "")
        plt.text(i + 1.1, p99, f'P99: {p99:.1f}ms', verticalalignment='center', color='red', fontweight='bold')
    
    plt.ylabel('Task Latency (ms)')
    plt.title('Experiment 3: Task Latency Distribution (Including Long-tail P99)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}exp3_latency_dist.pdf')
    plt.savefig(out_path)
    plt.savefig(out_path.replace('.pdf', '.png'))
    print(f"Saved {out_path} with P99 markers")
    plt.close()

def plot_exp4_stability(data_list, output_dir, prefix: str = ""):
    """Plot Stability (Completed vs Errors) for Exp 4."""
    if not any('exp4' in d['results'] for d in data_list):
        return

    labels = []
    completed = []
    errors = []
    throughput = []

    for d in data_list:
        res = d['results'].get('exp4')
        if not res: continue
        
        labels.append(d['label'])
        c = res.get('completed', 0)
        e = res.get('errors', 0)
        dur = res.get('duration', 30)
        
        completed.append(c)
        errors.append(e)
        throughput.append(c / dur if dur > 0 else 0)

    if not labels: return

    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Total Tasks & Errors
    rects1 = ax1.bar(x - width/2, completed, width, label='Completed', color='green', alpha=0.7)
    rects2 = ax1.bar(x + width/2, errors, width, label='Errors', color='red', alpha=0.7)
    
    ax1.set_ylabel('Count')
    ax1.set_title('Total Tasks Completed & Errors (Stability Test)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.bar_label(rects1, padding=3)
    ax1.bar_label(rects2, padding=3)

    # Plot 2: Throughput
    rects3 = ax2.bar(x, throughput, width*1.5, label='Throughput', color='blue', alpha=0.6)
    ax2.set_ylabel('Tasks / Second')
    ax2.set_title('System Stability Throughput')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.bar_label(rects3, padding=3, fmt='%.2f')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}exp4_stability.pdf')
    plt.savefig(out_path)
    plt.savefig(out_path.replace('.pdf', '.png'))
    print(f"Saved {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default=None, choices=["mock", "real"])
    args = parser.parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "experiment_results")
    output_dir = os.path.join(script_dir, "..", "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading results from: {results_dir}")
    print(f"Saving charts to: {output_dir}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    data_list = load_results(results_dir)
    if args.workload:
        data_list = [d for d in data_list if str(d.get("workload", "")).lower() == args.workload]
    
    if not data_list:
        print("No data found. Please run comprehensive_experiment.py first.")
        return
        
    # Generate Plots
    prefix = f"{args.workload}_" if args.workload else ""
    plot_exp1_concurrency(data_list, output_dir, prefix=prefix)
    plot_exp2_blocking(data_list, output_dir, prefix=prefix)
    plot_exp3_distribution(data_list, output_dir, prefix=prefix)
    plot_exp4_stability(data_list, output_dir, prefix=prefix)
    
if __name__ == "__main__":
    main()
