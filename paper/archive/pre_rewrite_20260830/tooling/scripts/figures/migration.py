import matplotlib.pyplot as plt
import numpy as np
from .common import load_json, save, FIG_MIGRATION


def plot_migration_timeline():
    symmetric_data = load_json("symmetric_aggregated.json")
    edgesched_data = load_json("edgesched_aggregated.json")

    if not symmetric_data or not edgesched_data:
        print("Warning: migration aggregated data not found, skipping timeline figure")
        return

    sym_metrics = symmetric_data.get("metrics", {})
    edge_metrics = edgesched_data.get("metrics", {})

    sym_stutter = sym_metrics.get("migration_stutter_ms", {}).get("mean", 0)
    sym_cpu_ttft = sym_metrics.get("coexistence_cpu_ttft_ms", {}).get("mean", 0)
    sym_restore_time = sym_metrics.get("restore_time_ms", {}).get("mean", 0)
    sym_restore_gap = sym_metrics.get("restore_window_max_gap_ms", {}).get("mean", 0)

    edge_stutter = edge_metrics.get("migration_stutter_ms", {}).get("mean", 0)
    edge_cpu_ttft = edge_metrics.get("coexistence_cpu_ttft_ms", {}).get("mean", 0)
    edge_restore_time = edge_metrics.get("restore_time_ms", {}).get("mean", 0)
    edge_restore_gap = edge_metrics.get("restore_window_max_gap_ms", {}).get("mean", 0)

    image_task_duration = 20000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.0), gridspec_kw={'hspace': 0.55})

    bar_height = 0.7
    y_cpu = 2.2
    y_gpu = 1.1
    y_img = 0.0

    # ── Panel 1: Symmetric Migration ──
    ax = ax1
    t = 0.0

    gpu_baseline_dur = 3000.0
    ax.barh(y_gpu, gpu_baseline_dur, left=t, height=bar_height,
            color='#3498db', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + gpu_baseline_dur / 2, y_gpu + bar_height / 2 + 0.1, 'GPU',
            ha='center', va='bottom', fontsize=7, fontweight='bold', color='#3498db')
    t += gpu_baseline_dur

    ax.barh(y_gpu, sym_stutter, left=t, height=bar_height,
            color='#e74c3c', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + sym_stutter / 2, y_gpu - bar_height / 2 - 0.1, 'Mig',
            ha='center', va='top', fontsize=7, fontweight='bold', color='#e74c3c')
    t += sym_stutter

    ax.barh(y_cpu, image_task_duration, left=t, height=bar_height,
            color='#f39c12', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + image_task_duration / 2, y_cpu, f'LLM on CPU\n(TTFT~{sym_cpu_ttft:.0f}ms)',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.barh(y_img, image_task_duration, left=t, height=bar_height,
            color='#9b59b6', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + image_task_duration / 2, y_img, 'Image Generation (GPU)',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    t += image_task_duration

    stop_dur = sym_restore_gap
    load_dur = sym_restore_time - sym_restore_gap
    restore_start_t = t

    if stop_dur > 500:
        ax.barh(y_cpu, stop_dur, left=t, height=bar_height,
                color='#c0392b', edgecolor='black', linewidth=0.8, alpha=0.9)
        ax.text(t + stop_dur / 2, y_cpu + bar_height / 2 + 0.1, 'Stop',
                ha='center', va='bottom', fontsize=7, fontweight='bold', color='#c0392b')

    ax.barh(y_gpu, load_dur, left=t, height=bar_height,
            color='#e67e22', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + load_dur / 2, y_gpu, 'Load GPU\nModel',
            ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    t += load_dur

    int_center = restore_start_t + load_dur / 2
    ax.annotate('', xy=(t, y_img - 0.3), xytext=(restore_start_t, y_img - 0.3),
                arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=2.5))
    ax.text(int_center, y_img - 0.55,
            f'Interruption Window\n~ {sym_restore_gap:.0f} ms (service unavailable)',
            ha='center', va='top', fontsize=8, fontweight='bold', color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='none', edgecolor='#c0392b', alpha=0.9,
                      linestyle='--', linewidth=1))

    gpu_restored_dur = 3000.0
    ax.barh(y_gpu, gpu_restored_dur, left=t, height=bar_height,
            color='#27ae60', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + gpu_restored_dur / 2, y_gpu, 'LLM on GPU\n(restored)',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.set_yticks([y_img, y_gpu, y_cpu])
    ax.set_yticklabels(['Image\n(GPU)', 'LLM\n(GPU)', 'LLM\n(CPU)'], fontsize=9)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_title('(a) Symmetric Migration: Stop -> Migrate -> Resume (both directions)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-800, t + gpu_restored_dur + 800)
    ax.set_ylim(-1.2, 3.0)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel 2: Asymmetric Migration (EdgeSched) ──
    ax = ax2
    t = 0.0

    ax.barh(y_gpu, gpu_baseline_dur, left=t, height=bar_height,
            color='#3498db', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + gpu_baseline_dur / 2, y_gpu + bar_height / 2 + 0.1, 'GPU',
            ha='center', va='bottom', fontsize=7, fontweight='bold', color='#3498db')
    t += gpu_baseline_dur

    ax.barh(y_gpu, edge_stutter, left=t, height=bar_height,
            color='#e74c3c', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + edge_stutter / 2, y_gpu - bar_height / 2 - 0.1, 'Off',
            ha='center', va='top', fontsize=7, fontweight='bold', color='#e74c3c')
    t += edge_stutter

    ax.barh(y_cpu, image_task_duration, left=t, height=bar_height,
            color='#f39c12', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + image_task_duration / 2, y_cpu, f'LLM on CPU\n(TTFT~{edge_cpu_ttft:.0f}ms)',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.barh(y_img, image_task_duration, left=t, height=bar_height,
            color='#9b59b6', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + image_task_duration / 2, y_img, 'Image Generation (GPU)',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    preload_dur = edge_restore_time
    ax.barh(y_gpu, preload_dur, left=t, height=bar_height,
            color='#2ecc71', edgecolor='black', linewidth=0.8, alpha=0.7,
            hatch='///')
    ax.text(t + preload_dur / 2, y_gpu, 'Background Preload\nGPU Instance',
            ha='center', va='center', fontsize=7, fontweight='bold', color='#145a32')

    ax.text(t + preload_dur * 0.7, y_cpu + 0.35, 'CPU continues\nserving',
            ha='center', va='bottom', fontsize=7, fontstyle='italic', color='#856404',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='none', edgecolor='#f39c12',
                      alpha=0.8, linestyle=':', linewidth=0.8))

    t += preload_dur

    swap_dur = max(edge_restore_gap, 1)
    ax.barh(y_gpu, swap_dur, left=t, height=bar_height,
            color='#27ae60', edgecolor='black', linewidth=0.8, alpha=0.95)
    ax.text(t + swap_dur / 2, y_gpu + bar_height / 2 + 0.1, 'Swap',
            ha='center', va='bottom', fontsize=7, fontweight='bold', color='#27ae60')

    int_center = t + swap_dur / 2
    ax.annotate('', xy=(t + swap_dur, y_img - 0.25), xytext=(t, y_img - 0.25),
                arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2.5))
    ax.text(int_center, y_img - 0.45,
            f'Interruption ~ {edge_restore_gap:.0f} ms (near-zero)\nservice always available',
            ha='center', va='top', fontsize=8, fontweight='bold', color='#1a7431',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='none', edgecolor='#27ae60', alpha=0.9,
                      linestyle='--', linewidth=1))

    t += swap_dur

    ax.barh(y_gpu, gpu_restored_dur, left=t, height=bar_height,
            color='#2980b9', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.text(t + gpu_restored_dur / 2, y_gpu, 'LLM on GPU\n(restored)',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.set_yticks([y_img, y_gpu, y_cpu])
    ax.set_yticklabels(['Image\n(GPU)', 'LLM\n(GPU)', 'LLM\n(CPU)'], fontsize=9)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_title('(b) Asymmetric Migration (EdgeSched): Urgent Down + Atomic Hot-Swap Up',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-800, t + gpu_restored_dur + 800)
    ax.set_ylim(-1.2, 3.0)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save("fig_migration_timeline", FIG_MIGRATION)


def plot_migration_comparison():
    symmetric_data = load_json("symmetric_aggregated.json")
    edgesched_data = load_json("edgesched_aggregated.json")

    if not symmetric_data or not edgesched_data:
        print("Warning: migration aggregated data not found")
        return

    sym_m = symmetric_data.get("metrics", {})
    edge_m = edgesched_data.get("metrics", {})

    metrics_names = ['Downlink\nStutter (ms)', 'CPU TTFT\n(ms)', 'Restore\nTime (ms)', 'Restore\nGap (ms)']

    sym_values = [sym_m.get(k, {}).get("mean", 0) for k in
                  ["migration_stutter_ms", "coexistence_cpu_ttft_ms", "restore_time_ms", "restore_window_max_gap_ms"]]
    sym_errors = [sym_m.get(k, {}).get("stdev", 0) for k in
                  ["migration_stutter_ms", "coexistence_cpu_ttft_ms", "restore_time_ms", "restore_window_max_gap_ms"]]
    edge_values = [edge_m.get(k, {}).get("mean", 0) for k in
                   ["migration_stutter_ms", "coexistence_cpu_ttft_ms", "restore_time_ms", "restore_window_max_gap_ms"]]
    edge_errors = [edge_m.get(k, {}).get("stdev", 0) for k in
                   ["migration_stutter_ms", "coexistence_cpu_ttft_ms", "restore_time_ms", "restore_window_max_gap_ms"]]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, sym_values, width, yerr=sym_errors,
                   label='Symmetric Migration', color='#e74c3c',
                   edgecolor='black', linewidth=0.8, capsize=4)
    bars2 = ax.bar(x + width / 2, edge_values, width, yerr=edge_errors,
                   label='EdgeSched (Asymmetric)', color='#3498db',
                   edgecolor='black', linewidth=0.8, capsize=4)

    ax.set_ylabel('Time (ms)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=9)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, val, err in zip(bars1, sym_values, sym_errors):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 50,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val, err in zip(bars2, edge_values, edge_errors):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 50,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save("fig_migration_comparison", FIG_MIGRATION)


def plot_ablation_comparison():
    symmetric_data = load_json("symmetric_aggregated.json")
    fdnu_data = load_json("fast_down_naive_up_aggregated.json")
    ndou_data = load_json("naive_down_optimized_up_aggregated.json")
    edgesched_data = load_json("edgesched_aggregated.json")

    if not all([symmetric_data, fdnu_data, ndou_data, edgesched_data]):
        print("Warning: ablation data not found")
        return

    variants = ['Symmetric\nMigration', 'Fast-down\n+ Naive-up', 'Naive-down\n+ Optimized-up', 'EdgeSched\n(Full)']
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db']
    all_data = [symmetric_data, fdnu_data, ndou_data, edgesched_data]

    stutter_vals = [d.get("metrics", {}).get("migration_stutter_ms", {}).get("mean", 0) for d in all_data]
    stutter_errs = [d.get("metrics", {}).get("migration_stutter_ms", {}).get("stdev", 0) for d in all_data]
    gap_vals = [d.get("metrics", {}).get("restore_window_max_gap_ms", {}).get("mean", 0) for d in all_data]
    gap_errs = [d.get("metrics", {}).get("restore_window_max_gap_ms", {}).get("stdev", 0) for d in all_data]
    rt_vals = [d.get("metrics", {}).get("restore_time_ms", {}).get("mean", 0) for d in all_data]
    rt_errs = [d.get("metrics", {}).get("restore_time_ms", {}).get("stdev", 0) for d in all_data]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(variants))

    for ax, vals, errs, title in zip(axes,
                                       [stutter_vals, gap_vals, rt_vals],
                                       [stutter_errs, gap_errs, rt_errs],
                                       ['(a) Downlink Stutter', '(b) Restore Gap (Interruption)', '(c) Restore Time']):
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        offset = 50 if 'Stutter' in title else (10 if 'Gap' in title else 100)
        for bar, val, err in zip(bars, vals, errs):
            label = f'{val:.0f}' if val > 0 else '~0'
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height() + err + offset, offset),
                    label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save("fig_ablation_comparison", FIG_MIGRATION)


def plot_resource_migration():
    data = load_json("edgesched_aggregated.json")
    if not data:
        print("Warning: edgesched_aggregated.json not found")
        return

    metrics = data.get("metrics", {})
    gpu_ttft = metrics.get("llm_gpu_ttft_ms", {}).get("mean", 0)
    gpu_ttft_err = metrics.get("llm_gpu_ttft_ms", {}).get("stdev", 0)
    cpu_ttft = metrics.get("coexistence_cpu_ttft_ms", {}).get("mean", 0)
    cpu_ttft_err = metrics.get("coexistence_cpu_ttft_ms", {}).get("stdev", 0)
    stutter = metrics.get("migration_stutter_ms", {}).get("mean", 0)
    stutter_err = metrics.get("migration_stutter_ms", {}).get("stdev", 0)
    restore_time = metrics.get("restore_time_ms", {}).get("mean", 0)
    restore_time_err = metrics.get("restore_time_ms", {}).get("stdev", 0)
    restore_gap = metrics.get("restore_window_max_gap_ms", {}).get("mean", 0)
    restore_gap_err = metrics.get("restore_window_max_gap_ms", {}).get("stdev", 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    ax1 = axes[0]
    names = ['GPU TTFT\n(Baseline)', 'CPU TTFT\n(Coexist)', 'Migration\nStutter']
    values = [gpu_ttft, cpu_ttft, stutter]
    errors = [gpu_ttft_err, cpu_ttft_err, stutter_err]
    colors = ['#27ae60', '#f39c12', '#e74c3c']
    x_pos = np.arange(len(names))

    bars = ax1.bar(x_pos, values, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel('Time (ms)', fontsize=10)
    ax1.set_title('(a) Downlink: Urgent Mode (GPU -> CPU)', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(values) * 1.35)
    for bar, val, err in zip(bars, values, errors):
        label = f'{val:.1f}'
        if err > 0:
            label += f'\n+/-{err:.1f}'
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (err if err else 0) + max(values) * 0.05,
                 label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2 = axes[1]
    names2 = ['Restore\nTime', 'Restore\nGap', 'Post-Restore\nTTFT']
    values2 = [restore_time, restore_gap, gpu_ttft]
    errors2 = [restore_time_err, restore_gap_err, gpu_ttft_err]
    colors2 = ['#9b59b6', '#2ecc71', '#1abc9c']
    x_pos2 = np.arange(len(names2))

    bars2 = ax2.bar(x_pos2, values2, yerr=errors2, capsize=5, color=colors2, edgecolor='black', linewidth=0.8, width=0.6)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(names2, fontsize=9)
    ax2.set_ylabel('Time (ms)', fontsize=10)
    ax2.set_title('(b) Uplink: Atomic Hot-Swap (CPU -> GPU)', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(values2) * 1.35)
    for bar, val, err in zip(bars2, values2, errors2):
        label = f'{val:.1f}'
        if err > 0:
            label += f'\n+/-{err:.1f}'
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (err if err else 0) + max(values2) * 0.05,
                 label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save("fig_resource_migration", FIG_MIGRATION)
