import matplotlib.pyplot as plt
from .common import load_json, save, FIG_AUXILIARY


def plot_real_workload_trace():
    data = load_json("real_workload_trace.json")
    if not data:
        print("Warning: real_workload_trace.json not found")
        return

    if isinstance(data, dict) and "data" in data:
        iterations = data["data"]
        if iterations and isinstance(iterations[-1], dict) and "data" in iterations[-1]:
            events = iterations[-1]["data"]
        elif isinstance(iterations[-1], list):
            events = iterations[-1]
        else:
            events = iterations[-1]
    elif isinstance(data, list):
        if data and isinstance(data[0], dict) and "data" in data[0]:
            events = data[-1]["data"]
        else:
            events = data
    else:
        print("Warning: unexpected real_workload_trace.json format")
        return

    type_config = {
        "LLM": ('#3498db', 'o'),
        "IMAGE": ('#9b59b6', 's'),
        "MIGRATION": ('#e74c3c', '^'),
        "RESTORE": ('#27ae60', 'D'),
    }

    fig, ax = plt.subplots(figsize=(12, 5.5))

    points = []
    for entry in events:
        ts = entry.get("timestamp_s", 0)
        task_type = entry.get("task_type", "")
        latency = entry.get("latency_s", 0) * 1000
        phase = entry.get("phase", "")
        color, marker = type_config.get(task_type, ('#95a5a6', 'x'))
        points.append((ts, latency, phase, color, marker, task_type))

    for ts, lat, _phase, color, marker, _type in points:
        ax.scatter(ts, lat, color=color, marker=marker, s=100,
                   edgecolors='black', linewidth=0.6, zorder=5)

    # 根据 task_type 和 phase 动态查找对应的点，避免硬编码索引出错
    def find_point(task_type, phase=None, index=0):
        matches = [p for p in points if p[5] == task_type and (phase is None or p[2] == phase)]
        if matches and index < len(matches):
            return matches[index]
        return None

    # GPU pre: 第一个 gpu_before_image 的 LLM 点
    gpu_pre = find_point("LLM", "gpu_before_image", 0)
    if gpu_pre:
        ax.annotate("GPU pre", (gpu_pre[0], gpu_pre[1]),
                    textcoords="offset points", xytext=(0, -13),
                    ha='center', va='top', fontsize=7.5, fontweight='bold', color='#3498db')

    # Migrate down: MIGRATION 点
    mig = find_point("MIGRATION", "gpu_to_cpu")
    if mig:
        ax.annotate("Migrate down", (mig[0], mig[1]),
                    textcoords="offset points", xytext=(0, 13),
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#e74c3c')

    # CPU 阶段: 找到所有 cpu_during_image 的 LLM 点，按 latency 排序（高的先，低的后）
    cpu_llms = [p for p in points if p[5] == "LLM" and p[2] == "cpu_during_image"]
    cpu_llms.sort(key=lambda p: p[1], reverse=True)
    if len(cpu_llms) >= 1:
        ax.annotate("CPU 1st req", (cpu_llms[0][0], cpu_llms[0][1]),
                    textcoords="offset points", xytext=(-35, -13),
                    ha='center', va='top', fontsize=7.5, fontweight='bold', color='#3498db')
    if len(cpu_llms) >= 2:
        ax.annotate("CPU 2nd req", (cpu_llms[1][0], cpu_llms[1][1]),
                    textcoords="offset points", xytext=(35, -13),
                    ha='center', va='top', fontsize=7.5, fontweight='bold', color='#3498db')

    # Image done: 两个 IMAGE 点，按 timestamp 排序
    img_points = [p for p in points if p[5] == "IMAGE"]
    img_points.sort(key=lambda p: p[0])
    if len(img_points) >= 1:
        ax.annotate("Image #1 done", (img_points[0][0], img_points[0][1]),
                    textcoords="offset points", xytext=(0, 13),
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#9b59b6')
    if len(img_points) >= 2:
        ax.annotate("Image #2 done", (img_points[1][0], img_points[1][1]),
                    textcoords="offset points", xytext=(0, 13),
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#9b59b6')

    # Restore up: RESTORE 点
    rst = find_point("RESTORE", "cpu_to_gpu")
    if rst:
        ax.annotate("Restore up", (rst[0], rst[1]),
                    textcoords="offset points", xytext=(0, 13),
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#27ae60')

    # GPU post: 第一个 gpu_after_restore 的 LLM 点
    gpu_post = find_point("LLM", "gpu_after_restore", 0)
    if gpu_post:
        ax.annotate("GPU post", (gpu_post[0], gpu_post[1]),
                    textcoords="offset points", xytext=(0, -13),
                    ha='center', va='top', fontsize=7.5, fontweight='bold', color='#3498db')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Real Workload Trace: LLM Chat + Image Generation',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25, linestyle='--')

    # 只扩展必要的边距
    all_ts = [p[0] for p in points]
    all_lat = [p[1] for p in points]
    ax.set_xlim(min(all_ts) - 3, max(all_ts) + 3)
    ax.set_ylim(min(all_lat) - 1800, max(all_lat) + 1800)

    legend_handles = [
        plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
                   markersize=9, markeredgecolor='black', label=l)
        for l, (c, m) in [('LLM Request', type_config["LLM"]),
                           ('Image Task', type_config["IMAGE"]),
                           ('Migration', type_config["MIGRATION"]),
                           ('Restore', type_config["RESTORE"])]
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              frameon=True, fontsize=9)

    plt.tight_layout()
    save("chart_real_workload_trace", FIG_AUXILIARY)
