import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 设置学术界常用的图表字体和样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiment_results")


def _finalize_figure(fig):
    fig.tight_layout()
    return fig

def plot_manual_llama_switch():
    """图1：完整 manual llama.cpp switch 基线"""
    data_file = os.path.join(RESULTS_DIR, "baseline_manual_llama_switch_results.json")
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Skip plotting.")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gpu_ttft = data.get("gpu_phase_ttft_s", [])
    image_tasks = data.get("image_tasks", [])
    img_latencies = [x.get("latency_s", 0.0) for x in image_tasks]

    labels = [
        'GPU TTFT\n(avg)',
        'Manual\nUnload',
        'Image #1\n1024x1024',
        'Image #2\n1024x1024',
        'LLM\nReload',
        'Restored\nTTFT',
    ]
    values = [
        float(np.mean(gpu_ttft)) if gpu_ttft else 0.0,
        data.get("manual_unload_time_s", 0.0),
        img_latencies[0] if len(img_latencies) > 0 else 0.0,
        img_latencies[1] if len(img_latencies) > 1 else 0.0,
        data.get("llm_reload_time_s", 0.0),
        data.get("restored_ttft_s", 0.0),
    ]

    colors = ['#1f77b4', '#7f7f7f', '#2ca02c', '#2ca02c', '#ff7f0e', '#d62728']
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor='black')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Latency (Seconds)')
    ax.set_title('Manual llama.cpp Switch Baseline')
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)

    for rect, val in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)

    out_prefix = os.path.join(RESULTS_DIR, "chart_manual_llama_switch")
    _finalize_figure(fig)
    plt.savefig(out_prefix + ".pdf")
    plt.savefig(out_prefix + ".png")
    plt.close()
    print(f"Generated {out_prefix}.pdf/png")

def plot_cold_start_vs_hot_swap():
    """图1：冷启动与热切换随上下文长度的延迟对比"""
    data_file = os.path.join(RESULTS_DIR, "baseline_cold_start_results.json")
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Skip plotting.")
        return
        
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ctx_sizes = sorted([int(k) for k in data["cold_start"].keys()])
    
    cold_times = [data["cold_start"][str(sz)]["total_recovery_time_s"] for sz in ctx_sizes]
    hot_times = [data["hot_swap"][str(sz)]["user_interruption_time_s"] for sz in ctx_sizes]
    
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    
    # 绘制折线图
    ax.plot(ctx_sizes, cold_times, marker='o', linestyle='-', linewidth=2, color='#d62728', label='Naive Cold-Start (Kill & Restart)')
    ax.plot(ctx_sizes, hot_times, marker='s', linestyle='--', linewidth=2, color='#1f77b4', label='EdgeSched (Atomic Hot-Swap)')
    
    ax.set_xlabel('Context Size (Tokens)')
    ax.set_ylabel('Recovery Latency (Seconds)')
    ax.set_title('Recovery Latency vs Context Size')
    
    ax.grid(True, linestyle=':', alpha=0.7)
    # 把图例移到图外，避免遮挡 500 tokens 的首个高点
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True)
    
    # 设置Y轴为对数坐标更明显展示差距，或者线性坐标展示绝对差距
    # 这里用线性更震撼，因为Cold start是好几秒，Hot swap是零点几秒
    ax.set_ylim(0, max(cold_times) * 1.2)
    ax.set_xlim(min(ctx_sizes) - 120, max(ctx_sizes) + 120)
    ax.annotate(
        'Warm-up affected',
        xy=(ctx_sizes[0], cold_times[0]),
        xytext=(760, cold_times[0] * 1.03),
        arrowprops=dict(arrowstyle='->', color='gray', lw=1),
        fontsize=10,
        color='gray'
    )
    
    out_prefix = os.path.join(RESULTS_DIR, "chart_cold_start_comparison")
    _finalize_figure(fig)
    plt.savefig(out_prefix + ".pdf")
    plt.savefig(out_prefix + ".png")
    plt.close()
    print(f"Generated {out_prefix}.pdf/png")

def plot_lru_thrashing():
    """图2：OS Unified Memory / LRU 失败态基线"""
    data_file = os.path.join(RESULTS_DIR, "baseline_lru_thrashing_results.json")
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Skip plotting.")
        return
        
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # OOM、错误或严重 Thrashing 超时，都按失败态展示
    status = str(data.get("status", ""))
    is_failure = (
        "OOM" in status
        or "Error" in status
        or "Timeout" in status
        or "Thrashing" in status
    )
    
    labels = ['LLM Only', 'Image Only', 'Concurrent (OS LRU/UM)']
    
    # 获取单任务基线
    llm_base = data.get("llm_only_latency_s", 0.5)
    img_base = data.get("image_only_latency_s", 2.5)
    
    # 获取并发（OS LRU）数据，如果崩溃则设为无穷大
    if is_failure:
        llm_os = 0 # 将在图中打上红叉
        img_os = 0
    else:
        llm_os = data.get("concurrent_llm_latency_s", 0)
        img_os = data.get("concurrent_image_latency_s", 0)
        
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    
    llm_bars = [llm_base, 0, llm_os]
    img_bars = [0, img_base, img_os]
    
    rects1 = ax.bar(x - width/2, llm_bars, width, label='LLM Task', color='#ff7f0e', edgecolor='black', hatch='//')
    rects2 = ax.bar(x + width/2, img_bars, width, label='Image Task', color='#2ca02c', edgecolor='black', hatch='\\\\')
    
    ax.set_ylabel('Inference Latency (Seconds)')
    ax.set_title('Task Latency Under Concurrent Execution (8GB VRAM)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='center')
    ax.legend()
    
    # 标注失败态
    if is_failure:
        failure_text = 'Severe Thrashing\n/ OOM / Timeout'
        ax.text(2.0, ax.get_ylim()[1] * 0.58, failure_text, ha='center', va='center',
                color='red', fontweight='bold', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    
    out_prefix = os.path.join(RESULTS_DIR, "chart_lru_thrashing")
    _finalize_figure(fig)
    plt.savefig(out_prefix + ".pdf")
    plt.savefig(out_prefix + ".png")
    plt.close()
    print(f"Generated {out_prefix}.pdf/png")

def plot_real_workload():
    """图3：真实交互工作负载时间线与三阶段 TTFT"""
    data_file = os.path.join(RESULTS_DIR, "real_workload_trace.json")
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Skip plotting.")
        return
        
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    llm_times = []
    llm_latencies = []
    llm_phases = []
    img_times = []
    img_latencies = []
    migration_times = []
    migration_latencies = []
    restore_times = []
    restore_latencies = []
    
    for item in data:
        if item["task_type"] == "LLM":
            llm_times.append(item["timestamp_s"])
            llm_latencies.append(item["latency_s"])
            llm_phases.append(item.get("phase", "unknown"))
        elif item["task_type"] == "IMAGE":
            img_times.append(item["timestamp_s"])
            img_latencies.append(item["latency_s"])
        elif item["task_type"] == "MIGRATION":
            migration_times.append(item["timestamp_s"])
            migration_latencies.append(item["latency_s"])
        elif item["task_type"] == "RESTORE":
            restore_times.append(item["timestamp_s"])
            restore_latencies.append(item["latency_s"])
            
    # 使用双Y轴或者直接在点上标注数值
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))

    phase_styles = {
        "gpu_before_image": ("#1f77b4", "GPU TTFT"),
        "cpu_during_image": ("#d62728", "CPU TTFT"),
        "gpu_after_restore": ("#2ca02c", "Restored GPU TTFT"),
    }
    used_labels = set()
    
    # 绘制连线
    ax1.plot(llm_times, llm_latencies, color='#888888', linewidth=1.5, alpha=0.5, zorder=1)
    
    # 绘制数据点并添加数值标注
    for t, lat, phase in zip(llm_times, llm_latencies, llm_phases):
        color, label = phase_styles.get(phase, ("#7f7f7f", phase))
        plot_label = label if label not in used_labels else None
        used_labels.add(label)
        ax1.scatter([t], [lat], color=color, s=60, label=plot_label, zorder=3, 
                   edgecolors='black', linewidths=1.0, marker='o')
        
        # 为GPU TTFT添加数值标注
        if phase == "gpu_before_image":
            ax1.annotate(f'{lat:.2f}s', xy=(t, lat), xytext=(0, 8),
                        textcoords='offset points', fontsize=8, ha='center',
                        color='#1f77b4', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='#1f77b4', alpha=0.8, linewidth=0.8))

    ax1.set_xlabel('Timeline (Seconds)', fontsize=12)
    ax1.set_ylabel('LCRT TTFT (Seconds)', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # 设置Y轴范围
    max_lat = max(llm_latencies) if llm_latencies else 1
    ax1.set_ylim(0, max_lat * 1.2)
    
    # 设置Y轴主刻度，不使用次刻度
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    
    # 标记迁移窗口
    if migration_times:
        mig_start = migration_times[0] - migration_latencies[0]
        mig_end = migration_times[0]
        ax1.axvspan(mig_start, mig_end, color='orange', alpha=0.25, 
                   label='GPU -> CPU Migration', zorder=0)

    # 标记图片执行窗口
    if img_times:
        burst_start = max(0, min(img_times) - max(img_latencies))
        ax1.axvspan(burst_start, max(img_times), color='red', alpha=0.12, 
                   label='Image Execution Window', zorder=0)
        for idx, (t, lat) in enumerate(zip(img_times, img_latencies), start=1):
            ax1.axvline(t, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
            ax1.text(t, max_lat * 1.12, f'Img {idx}: {lat:.1f}s',
                    fontsize=9, color='red', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='red', alpha=0.9, linewidth=1))

    # 标记恢复线
    if restore_times:
        for rt in restore_times:
            ax1.axvline(rt, color='#2ca02c', linestyle='--', linewidth=2.0, 
                       label='CPU -> GPU Restore', alpha=0.7)
        
    # 图例放在图表下方
    lines1, labels1 = ax1.get_legend_handles_labels()
    merged = []
    seen = set()
    for line, label in list(zip(lines1, labels1)):
        if label and label not in seen:
            merged.append((line, label))
            seen.add(label)
    
    ax1.legend([m[0] for m in merged], [m[1] for m in merged], 
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=True, fontsize=10, framealpha=0.95, edgecolor='gray')
        
    # 网格
    ax1.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)
    ax1.set_title('Real Workload Trace: TTFT Before, During, and After Image Burst', 
                 fontsize=13, pad=12)
    
    out_prefix = os.path.join(RESULTS_DIR, "chart_real_workload_trace")
    plt.savefig(out_prefix + ".pdf", dpi=300, bbox_inches='tight')
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_prefix}.pdf/png")

if __name__ == "__main__":
    print("Generating Academic Charts...")
    plot_manual_llama_switch()
    plot_cold_start_vs_hot_swap()
    plot_lru_thrashing()
    plot_real_workload()
    print("Done!")
