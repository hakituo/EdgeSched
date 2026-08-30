import json
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

def visualize_trace(json_path, output_path):
    with open(json_path, 'r') as f:
        events = json.load(f)

    tasks = {}
    for event in events:
        task_id = event['task_id']
        if task_id not in tasks:
            tasks[task_id] = {'scheduled': None, 'start': None, 'end': None, 'name': 'Unknown', 'worker': 'Unknown'}
        
        if event['event'] == 'task_started':
            tasks[task_id]['start'] = event['ts']
            # Extract worker and task name from details if possible
            # Example: ... Worker worker-0 executing task ...
            if 'Worker' in event['details']:
                try:
                    tasks[task_id]['worker'] = event['details'].split('Worker')[1].split('executing')[0].strip()
                except Exception:
                    pass
        elif event['event'] == 'task_completed':
            tasks[task_id]['end'] = event['ts']
        elif event['event'] == 'task_scheduled':
            tasks[task_id]['scheduled'] = event['ts']
            # Example: ... Name: llm_-1
            if 'Name:' in event['details']:
                try:
                    tasks[task_id]['name'] = event['details'].split('Name:')[1].strip()
                except Exception:
                    pass

    plot_data = []
    min_ts = float('inf')
    
    for task_id, info in tasks.items():
        if info['start'] and info['end']:
            info['task_id'] = task_id
            plot_data.append(info)
            base_ts = info.get('scheduled') or info.get('start')
            if base_ts and base_ts < min_ts:
                min_ts = base_ts

    if not plot_data:
        print("No complete tasks found to plot.")
        return

    for task in plot_data:
        scheduled = task.get('scheduled')
        if scheduled is None:
            scheduled = task.get('start')

        task['scheduled_rel'] = scheduled - min_ts
        task['start_rel'] = task['start'] - min_ts
        task['end_rel'] = task['end'] - min_ts
        task['queue_duration'] = max(0.0, task['start_rel'] - task['scheduled_rel'])
        task['exec_duration'] = max(0.0, task['end_rel'] - task['start_rel'])
        task['duration'] = task['end_rel'] - task['scheduled_rel']

    plot_data.sort(key=lambda x: (x.get('scheduled') or x.get('start') or 0.0, x.get('start') or 0.0))

    for i, task in enumerate(plot_data, start=1):
        task['idx'] = i

    worker_order = []
    worker_to_lane = {}
    for task in plot_data:
        w = task.get('worker') or 'Unknown'
        if w not in worker_to_lane:
            worker_to_lane[w] = len(worker_order)
            worker_order.append(w)

    def normalize_task_name(raw: str) -> str:
        if not raw:
            return "Unknown"
        s = str(raw).strip()
        if "(Backend:" in s:
            s = s.split("(Backend:")[0].strip()
        return s

    fig_height = max(4.5, 1.0 + 0.9 * len(worker_order))
    fig, ax = plt.subplots(figsize=(16, fig_height))

    # Color map for task types
    colors = {'llm': 'skyblue', 'vl': 'lightgreen', 'sd': 'salmon', 'tts': 'orange'}
    
    yticks = list(range(len(worker_order)))
    yticklabels = worker_order

    label_budget = 30
    label_threshold_s = 1.5
    candidates = sorted(
        plot_data,
        key=lambda t: (t.get('exec_duration') or 0.0) + 0.7 * (t.get('queue_duration') or 0.0),
        reverse=True,
    )
    labeled = []
    for t in candidates:
        if len(labeled) >= label_budget:
            break
        if (t.get('exec_duration') or 0.0) < label_threshold_s and (t.get('queue_duration') or 0.0) < label_threshold_s:
            continue
        labeled.append(t)
    labeled_idx = {t['idx'] for t in labeled}

    for task in plot_data:
        scheduled_rel = task['scheduled_rel']
        start_rel = task['start_rel']
        queue_dur = task['queue_duration']
        exec_dur = task['exec_duration']
        worker = task.get('worker') or 'Unknown'
        lane = worker_to_lane.get(worker, 0)
        name = normalize_task_name(task.get('name', 'Unknown'))
        
        # Determine color based on task name prefix
        color = 'gray'
        for key, c in colors.items():
            if key in name.lower():
                color = c
                break
        
        if queue_dur > 0.0:
            ax.barh(
                lane,
                queue_dur,
                left=scheduled_rel,
                height=0.55,
                align='center',
                color='lightgray',
                alpha=0.55,
                edgecolor='black',
                linewidth=0.6,
            )

        ax.barh(
            lane,
            exec_dur,
            left=start_rel,
            height=0.55,
            align='center',
            color=color,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.8,
        )

        if task.get('idx') in labeled_idx:
            idx_text = str(task['idx'])
            x_mid = start_rel + exec_dur / 2
            if exec_dur >= 0.9:
                ax.text(
                    x_mid,
                    lane,
                    idx_text,
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                    clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.72),
                )
            else:
                ax.text(
                    start_rel + exec_dur + 0.25,
                    lane,
                    idx_text,
                    ha='left',
                    va='center',
                    fontsize=8,
                    color='black',
                    clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.72),
                )

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=11)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_title('Scheduler Trace (IDs on selected tasks; see index table)', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    legend_handles = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.55, label='Queued (scheduled→started)'),
        mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.85, label='Executing (started→completed)'),
    ]
    for key, c in colors.items():
        legend_handles.append(mpatches.Patch(facecolor=c, edgecolor='black', alpha=0.85, label=f"Type: {key}"))
    
    # Move legend to outside to avoid occlusion
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9, framealpha=0.9, 
              bbox_to_anchor=(1.01, 1.0))

    ax.margins(x=0.05)
    fig.tight_layout() # Use standard tight_layout with outside legend
    plt.savefig(output_path, dpi=420, bbox_inches='tight', pad_inches=0.15)
    if str(output_path).lower().endswith('.png'):
        plt.savefig(str(output_path)[:-4] + '.pdf', bbox_inches='tight', pad_inches=0.15)
    elif str(output_path).lower().endswith('.pdf'):
        plt.savefig(str(output_path)[:-4] + '.png', dpi=420, bbox_inches='tight', pad_inches=0.15)
    print(f"Saved trace visualization to {output_path}")

    out_path = Path(output_path)
    index_csv_path = out_path.with_name('scheduler_trace_index.csv')
    index_tex_path = out_path.with_name('scheduler_trace_index.tex')

    with open(index_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'idx',
            'task_id',
            'name',
            'worker',
            'scheduled_rel_s',
            'start_rel_s',
            'end_rel_s',
            'queue_s',
            'exec_s',
        ])
        for t in plot_data:
            writer.writerow([
                t.get('idx'),
                t.get('task_id', ''),
                normalize_task_name(t.get('name', 'Unknown')),
                t.get('worker', 'Unknown'),
                f"{t.get('scheduled_rel', 0.0):.6f}",
                f"{t.get('start_rel', 0.0):.6f}",
                f"{t.get('end_rel', 0.0):.6f}",
                f"{t.get('queue_duration', 0.0):.6f}",
                f"{t.get('exec_duration', 0.0):.6f}",
            ])

    def tex_tt(s: str) -> str:
        s = '' if s is None else str(s)
        s = s.replace('\n', ' ').replace('\r', ' ')
        return f"\\texttt{{\\detokenize{{{s}}}}}"

    table_rows = [t for t in plot_data if t.get('idx') in labeled_idx]
    table_rows.sort(key=lambda t: t.get('idx') or 0)

    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{6pt}')
    lines.append('\\begin{tabular}{rllrr}')
    lines.append('\\toprule')
    lines.append('ID & Name & Worker & Queue(s) & Exec(s) \\\\')
    lines.append('\\midrule')
    for t in table_rows:
        idx = t.get('idx')
        name = tex_tt(normalize_task_name(t.get('name', 'Unknown')))
        worker = tex_tt(t.get('worker', 'Unknown'))
        q = f"{t.get('queue_duration', 0.0):.1f}"
        e = f"{t.get('exec_duration', 0.0):.1f}"
        lines.append(f"{idx} & {name} & {worker} & {q} & {e} \\\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\caption{Trace 任务索引（信息分层）。图内仅对少量关键任务标注 ID；完整清单见 \\texttt{scheduler\\_trace\\_index.csv}。}')
    lines.append('\\end{table}')

    with open(index_tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Saved trace index CSV to {index_csv_path}")
    print(f"Saved trace index LaTeX to {index_tex_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        visualize_trace(sys.argv[1], sys.argv[2])
    else:
        experiments_dir = Path(__file__).resolve().parent
        mvp_core_dir = experiments_dir.parent
        visualize_trace(
            str(experiments_dir / "captured_trace.json"),
            str(mvp_core_dir / "experiment_results" / "scheduler_trace_timeline.png"),
        )
