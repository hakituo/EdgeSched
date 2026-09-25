import matplotlib.pyplot as plt
import numpy as np
from .common import load_json, save, FIG_BASELINE


def plot_cold_start_comparison():
    data = load_json("baseline_cold_start_results.json")
    if not data:
        print("Warning: baseline_cold_start_results.json not found")
        return

    fig, ax = plt.subplots(figsize=(9, 5.0))

    cold_start = data.get("cold_start", {})
    hot_swap = data.get("hot_swap", {})

    contexts = sorted(cold_start.keys(), key=int)
    x = np.arange(len(contexts))
    width = 0.35

    def _get_val(d, ctx, key):
        v = d.get(ctx, {}).get(key, 0)
        if isinstance(v, dict):
            v = v.get("mean", 0)
        return float(v)

    cold_values = [_get_val(cold_start, c, "total_recovery_time_s") * 1000 for c in contexts]
    hot_values = [_get_val(hot_swap, c, "user_interruption_time_s") * 1000 for c in contexts]
    speedups = [c / h if h > 0 else 0 for c, h in zip(cold_values, hot_values)]

    bars1 = ax.bar(x - width / 2, cold_values, width, label='Cold Start (Restart)',
                   color='#e74c3c', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, hot_values, width, label='Hot-Swap (Atomic)',
                   color='#27ae60', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Recovery Time (ms)', fontsize=10)
    ax.set_xlabel('Context Length (tokens)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    y_max = max(max(cold_values), max(hot_values)) * 1.25
    ax.set_ylim(0, y_max)

    for bar, val in zip(bars1, cold_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.03,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar, val, spd in zip(bars2, hot_values, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.03,
                f'{val:.0f}\n({spd:.1f}x)', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Cold Start vs Atomic Hot-Swap Recovery', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save("chart_cold_start_comparison", FIG_BASELINE)


def plot_lru_thrashing():
    data = load_json("baseline_lru_thrashing_results.json")
    if not data:
        print("Warning: baseline_lru_thrashing_results.json not found")
        return

    def _get_val(d, key):
        v = d.get(key, 0)
        if isinstance(v, dict):
            return v.get("mean", 0)
        return v

    def _get_err(d, key):
        v = d.get(key, 0)
        if isinstance(v, dict):
            return v.get("stdev", 0)
        return 0

    cold_data = data.get("cold_start", data) if isinstance(data, dict) else data
    warm_data = data.get("warm_cache", None) if isinstance(data, dict) else None

    has_dual = warm_data is not None

    if has_dual:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.0))
    else:
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax2 = None

    categories = ['LLM Only', 'Image Only', 'Concurrent\nLLM', 'Concurrent\nImage']
    metric_keys = ["llm_only_latency_s", "image_only_latency_s",
                   "concurrent_llm_latency_s", "concurrent_image_latency_s"]

    def _plot_panel(ax, panel_data, title, panel_label):
        values = [_get_val(panel_data, k) * 1000 for k in metric_keys]
        errors = [_get_err(panel_data, k) * 1000 for k in metric_keys]
        colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12']

        timeout_count = panel_data.get("timeout_count", 0)
        success_count = panel_data.get("success_count", 0)
        iterations = panel_data.get("iterations", 5)

        if timeout_count > 0:
            conc_llm_val = _get_val(panel_data, "concurrent_llm_latency_s")
            if conc_llm_val == 0:
                idx = metric_keys.index("concurrent_llm_latency_s")
                values[idx] = 0

        bars = ax.bar(categories, values, yerr=errors, capsize=4,
                      color=colors, edgecolor='black', linewidth=0.8, width=0.5)
        ax.set_ylabel('Latency (ms)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        y_max = max(values) * 1.2 if max(values) > 0 else 1000
        ax.set_ylim(0, y_max)
        for bar, val, err in zip(bars, values, errors):
            if val > 0:
                label = f'{val:.0f}'
                if err > 0:
                    label += f'\n±{err:.0f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (err if err else 0) + y_max * 0.03,
                        label, ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, y_max * 0.05,
                        'Timeout\n(>60s)', ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color='red')

        status_text = f'N={iterations}: {success_count} ok, {timeout_count} timeout'
        thrashing = panel_data.get("thrashing_ratio", {})
        if thrashing:
            status_text += f' | LLM {thrashing.get("concurrent_llm_vs_llm_only", 0):.1f}x slower'
        ax.text(0.5, -0.15, status_text,
                transform=ax.transAxes, fontsize=9, ha='center', va='top',
                style='italic', color='#e74c3c')

    _plot_panel(ax1, cold_data,
                '(a) Cold-Start: OS LRU May Timeout',
                'cold')

    if has_dual:
        _plot_panel(ax2, warm_data,
                    '(b) Warm-Cache: Still Unacceptably Slow',
                    'warm')

    plt.tight_layout()
    save("chart_lru_thrashing", FIG_BASELINE)


def plot_baseline_comparison():
    data = load_json("baseline_comparison.json")
    if not data:
        print("Warning: baseline_comparison.json not found")
        return

    comparison = data.get("comparison", {})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1 = axes[0]
    methods = ['Naive\nAsync', 'Threadpool', 'Semaphore', 'EdgeSched']
    throughputs = [
        comparison.get('naive_async', {}).get('throughput_tasks_per_s', 0),
        comparison.get('threadpool', {}).get('throughput_tasks_per_s', 0),
        comparison.get('semaphore', {}).get('throughput_tasks_per_s', 0),
        comparison.get('edgesched', {}).get('throughput_tasks_per_s', 0)
    ]
    colors = ['#e74c3c', '#27ae60', '#f39c12', '#3498db']

    bars = ax1.bar(methods, throughputs, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    ax1.set_ylabel('Throughput (tasks/s)', fontsize=10)
    ax1.set_title('(a) Throughput Comparison', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 90)
    for bar, val in zip(bars, throughputs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = axes[1]
    p99_latencies = [
        comparison.get('naive_async', {}).get('p99_latency_ms', 0),
        comparison.get('threadpool', {}).get('p99_latency_ms', 0),
        comparison.get('semaphore', {}).get('p99_latency_ms', 0),
        comparison.get('edgesched', {}).get('p99_latency_ms', 0)
    ]

    bars2 = ax2.bar(methods, p99_latencies, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    ax2.set_ylabel('P99 Latency (ms)', fontsize=10)
    ax2.set_title('(b) P99 Latency Comparison', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 8500)
    for bar, val in zip(bars2, p99_latencies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save("fig_baseline_comparison", FIG_BASELINE)
