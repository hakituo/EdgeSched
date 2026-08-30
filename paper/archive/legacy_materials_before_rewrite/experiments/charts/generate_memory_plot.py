import csv
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path


def _configure_matplotlib_fonts():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Source Han Sans SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in candidates if f in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False

def _read_vram_csv(csv_path: Path):
    time_s = []
    mem_gb = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t_key = 't_s' if 't_s' in row else ('Time_s' if 'Time_s' in row else None)
                m_key = 'mem_gb' if 'mem_gb' in row else ('VRAM_GB' if 'VRAM_GB' in row else None)
                if t_key is None or m_key is None:
                    continue
                time_s.append(float(row[t_key]))
                mem_gb.append(float(row[m_key]))
            except Exception:
                continue
    return time_s, mem_gb


def _simulate_vram_from_trace(trace_file: Path):
    with open(trace_file, 'r', encoding='utf-8') as f:
        events = json.load(f)

    events.sort(key=lambda x: x.get('ts', 0.0))
    if not events:
        return [], []

    def classify(details: str) -> str:
        s = (details or '').lower()
        if 'sd' in s or 'image' in s:
            return 'SD'
        if 'llm' in s:
            return 'LLM'
        if 'vl' in s:
            return 'VL'
        return 'Unknown'

    vram_delta_gb = {
        'LLM': 4.5,
        'VL': 2.2,
        'SD': 4.0,
    }

    time_points = [0.0]
    memory_usage = [0.5]
    current_memory = 0.5
    start_time = events[0].get('ts', 0.0)
    active_tasks = {}

    for event in events:
        t = float(event.get('ts', start_time)) - start_time
        task_id = event.get('task_id')
        name = classify(event.get('details', ''))

        time_points.append(t)
        memory_usage.append(current_memory)

        if event.get('event') == 'task_started':
            current_memory += vram_delta_gb.get(name, 0.0)
            if task_id is not None:
                active_tasks[task_id] = name
        elif event.get('event') == 'task_completed':
            task_name = active_tasks.get(task_id, 'Unknown')
            current_memory -= vram_delta_gb.get(task_name, 0.0)

        time_points.append(t + 0.01)
        memory_usage.append(current_memory)

    time_points.append(time_points[-1] + 2.0)
    memory_usage.append(0.5)
    return time_points, memory_usage


def _pick_markers(time_s, mem_gb):
    if not time_s or not mem_gb:
        return None

    baseline = min(mem_gb)
    load_t = None
    unload_t = None

    for i in range(1, len(mem_gb)):
        if mem_gb[i] > baseline + 0.1 and mem_gb[i - 1] <= baseline + 0.1:
            load_t = time_s[i]
            break

    for i in range(len(mem_gb) - 1, 0, -1):
        if mem_gb[i] <= baseline + 0.1 and mem_gb[i - 1] > baseline + 0.1:
            unload_t = time_s[i]
            break

    peak_idx = max(range(len(mem_gb)), key=lambda i: mem_gb[i])
    gen_t = time_s[peak_idx]

    if load_t is None:
        load_t = time_s[0]
    if unload_t is None:
        unload_t = time_s[-1]

    if gen_t <= load_t:
        gen_t = (load_t + unload_t) / 2.0
    return {
        'load': load_t,
        'gen': gen_t,
        'unload': unload_t,
    }


def generate_memory_plot(trace_file: Path, output_path: Path):
    _configure_matplotlib_fonts()
    base_dir = output_path.parent
    vram_csv = base_dir / 'vram_usage_over_time.csv'

    if vram_csv.exists():
        time_s, mem_gb = _read_vram_csv(vram_csv)
    else:
        time_s, mem_gb = _simulate_vram_from_trace(trace_file)

    if not time_s or not mem_gb:
        print('No VRAM data to plot.')
        return

    markers = _pick_markers(time_s, mem_gb)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.plot(time_s, mem_gb, color='#1f77b4', linewidth=2.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Memory Usage over Time')
    ax.grid(True, linestyle='--', alpha=0.6)

    if markers is not None:
        ymax = max(mem_gb)
        # Increase ylim to make room for annotations
        ax.set_ylim(top=ymax * 1.15)

        for key, label in [('load', '加载模型'), ('gen', '生成图像'), ('unload', '卸载模型')]:
            t = markers[key]
            ax.axvline(t, color='black', linestyle='--', linewidth=1.0, alpha=0.85)
            ax.annotate(
                label,
                xy=(t, ymax), # Point to the peak height
                xytext=(0, 10), # Offset upwards
                textcoords='offset points',
                ha='center', # Center alignment
                va='bottom',
                fontsize=10,
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5, linestyle=':'),
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.12)
    if str(output_path).lower().endswith('.pdf'):
        fig.savefig(str(output_path)[:-4] + '.png', dpi=360, bbox_inches='tight', pad_inches=0.12)
    elif str(output_path).lower().endswith('.png'):
        fig.savefig(str(output_path)[:-4] + '.pdf', bbox_inches='tight', pad_inches=0.12)
    print(f"Saved memory plot to {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir.parent / "experiment_results"
    trace_file = results_dir / "scheduler_trace_events.json"
    output_file = results_dir / "memory_usage_over_time.pdf"
    
    if trace_file.exists():
        generate_memory_plot(trace_file, output_file)
    else:
        print(f"Trace file not found: {trace_file}")
