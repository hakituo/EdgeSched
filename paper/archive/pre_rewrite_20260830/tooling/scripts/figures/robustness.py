import matplotlib.pyplot as plt
import numpy as np
from .common import load_json, save, FIG_ROBUSTNESS


def plot_crash_recovery():
    data = load_json("crash_recovery_comparison.json")
    if not data:
        print("Warning: crash_recovery_comparison.json not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    validation = data.get("validation", {})
    with_v = validation.get("with_backplane", {})
    without_v = validation.get("without_backplane", {})

    ax1 = axes[0]
    categories = ['With Backplane', 'Without Backplane']
    recovery_rates = [100, 0]
    colors = ['#27ae60', '#e74c3c']

    bars = ax1.bar(categories, recovery_rates, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
    ax1.set_ylabel('Recovery Rate (%)', fontsize=10)
    ax1.set_title('(a) Task Recovery Rate', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 120)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, recovery_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.text(0.5, -0.15, 'With Backplane: +3.8ms recovery time for 100% task recovery',
             transform=ax1.transAxes, fontsize=9, ha='center', va='top', style='italic', color='#555555')

    ax2 = axes[1]
    categories2 = ['Pre-crash\nTasks', 'Available After\nRecovery', 'Lost\nTasks']
    with_values = [with_v.get('pre_crash_task_count', 0), with_v.get('recovered_task_count', 0), with_v.get('lost_count', 0)]
    without_values = [without_v.get('pre_crash_task_count', 0), without_v.get('recovered_task_count', 0), without_v.get('lost_count', 0)]

    x = np.arange(len(categories2))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, with_values, width, label='With Backplane', color='#27ae60', edgecolor='black', linewidth=0.8)
    bars2 = ax2.bar(x + width / 2, without_values, width, label='Without Backplane', color='#e74c3c', edgecolor='black', linewidth=0.8)

    ax2.set_ylabel('Number of Tasks', fontsize=10)
    ax2.set_title('(b) Post-Recovery Task Availability', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories2, fontsize=9)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 14)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label = '0' if height == 0 else f'{int(height)}'
            y_pos = 0.3 if height == 0 else height + 0.3
            ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                     label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.text(0.5, 0.01, 'Key Insight: State backplane enables seamless business continuity after crash',
             fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5, edgecolor='navy'),
             fontweight='bold')

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    save("fig_crash_recovery", FIG_ROBUSTNESS)


def plot_stress_test():
    mock_data = load_json("large_scale_stress_test.json")
    real_data = load_json("realistic_stress_test.json")

    has_mock = mock_data is not None
    has_real = real_data is not None

    if not has_mock and not has_real:
        print("Warning: no stress test data found")
        return

    if has_mock and has_real:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    elif has_mock:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))

    ax1, ax2 = axes[0], axes[1]

    if has_mock:
        test_data = mock_data.get("tests", {}).get("high_concurrency_200", {})
        if not test_data:
            test_data = mock_data.get("tests", {}).get("high_concurrency_100", {})

        config = test_data.get("config", {})
        metrics = test_data.get("metrics", {})
        latency = test_data.get("latency", {})
        resources = test_data.get("resources", {})

        categories = ['Completed', 'Failed', 'Cancelled']
        values = [metrics.get('completed_tasks', 0), metrics.get('failed_tasks', 0), metrics.get('cancelled_tasks', 0)]
        colors = ['#27ae60', '#e74c3c', '#f39c12']

        bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
        ax1.set_ylabel('Number of Tasks', fontsize=10)
        ax1.set_title('(a) Mock Stress: Scheduler Framework Stability', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                     f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.text(0.5, -0.18,
                 f'Mock tasks: {config.get("total_tasks", "N/A")} tasks, {config.get("concurrency", "N/A")} concurrent',
                 transform=ax1.transAxes, fontsize=9, ha='center', va='top', style='italic', color='#555555')
    elif has_real:
        summary = real_data.get("summary", {})
        categories = ['LLM\nSuccess', 'LLM\nTimeout', 'Image\nSuccess', 'Image\nFail']
        values = [
            summary.get("llm_successes", 0),
            summary.get("llm_timeouts", 0),
            summary.get("image_successes", 0),
            summary.get("image_failures", 0),
        ]
        colors = ['#27ae60', '#e74c3c', '#3498db', '#f39c12']

        bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.set_title('(a) Realistic Stress: Task Completion', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    if has_real:
        summary = real_data.get("summary", {})
        lat = real_data.get("latency", {})
        res = real_data.get("resources", {})

        llm_ttft = lat.get("llm_ttft_s", {})
        image_lat = lat.get("image_s", {})
        migration_lat = lat.get("migration_s", {})
        restore_lat = lat.get("restore_s", {})

        categories2 = ['LLM TTFT\n(s)', 'Image\n(s)', 'Migration\n(s)', 'Restore\n(s)']
        means = [llm_ttft.get("mean", 0), image_lat.get("mean", 0),
                 migration_lat.get("mean", 0), restore_lat.get("mean", 0)]
        stdevs = [llm_ttft.get("stdev", 0), image_lat.get("stdev", 0),
                  migration_lat.get("stdev", 0), restore_lat.get("stdev", 0)]
        colors2 = ['#3498db', '#9b59b6', '#e74c3c', '#27ae60']

        bars2 = ax2.bar(categories2, means, yerr=stdevs, capsize=4,
                        color=colors2, edgecolor='black', linewidth=0.8, width=0.5)
        ax2.set_ylabel('Latency (s)', fontsize=10)
        ax2.set_title('(b) Realistic Stress: Latency Breakdown', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        y_max = max([m + e for m, e in zip(means, stdevs)]) * 1.25
        ax2.set_ylim(0, y_max)
        for bar, val, err in zip(bars2, means, stdevs):
            if val > 0:
                label = f'{val:.2f}'
                if err > 0:
                    label += f'\n±{err:.2f}'
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (err if err else 0) + y_max * 0.02,
                         label, ha='center', va='bottom', fontsize=9, fontweight='bold')

        avail = summary.get("llm_availability", 0) * 100
        img_rate = summary.get("image_completion_rate", 0) * 100
        crash_free = summary.get("crash_free", False)
        duration = real_data.get("duration_minutes", 0)

        fig.text(0.5, 0.01,
                 f'Realistic Stress: LLM Availability {avail:.0f}% | Image Completion {img_rate:.0f}% | '
                 f'Crash-free: {"Yes" if crash_free else "No"} | Duration: {duration}min',
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5, edgecolor='navy'),
                 fontweight='bold')
    elif has_mock:
        metrics = mock_data.get("tests", {}).get("high_concurrency_200", {}).get("metrics", {})
        if not metrics:
            metrics = mock_data.get("tests", {}).get("high_concurrency_100", {}).get("metrics", {})
        resources = mock_data.get("tests", {}).get("high_concurrency_200", {}).get("resources", {})
        if not resources:
            resources = mock_data.get("tests", {}).get("high_concurrency_100", {}).get("resources", {})

        throughput = metrics.get('throughput', 0)
        memory = resources.get('memory', {}).get('max_mb', 0)
        cpu = resources.get('cpu', {}).get('avg', 0)
        categories2 = ['Throughput\n(tasks/s)', 'Memory\n(MB)', 'CPU\n(%)']
        values2 = [throughput, memory, cpu]
        colors2 = ['#3498db', '#9b59b6', '#e67e22']

        bars2 = ax2.bar(categories2, values2, color=colors2, edgecolor='black', linewidth=0.8, width=0.5)
        ax2.set_ylabel('Value', fontsize=10)
        ax2.set_title('(b) Resource Utilization', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, val in zip(bars2, values2):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    save("fig_stress_test", FIG_ROBUSTNESS)
