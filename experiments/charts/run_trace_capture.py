import subprocess
import time
import os
import sys
import glob
import json
import re
import threading
import csv
from pathlib import Path

def capture_data():
    print("=== Starting Trace & SMI Capture (Real Workload) ===")

    experiments_dir = Path(__file__).resolve().parent
    mvp_core_dir = experiments_dir.parent
    output_dir = mvp_core_dir / "experiment_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Start Experiment (Async)
    # Run Exp 3 with 1 request to get a clean single trace
    # Force usage of C++ scheduler flags to ensure trace reflects Hybrid Architecture if available
    cmd = [
        sys.executable, 
        "comprehensive_experiment.py", 
        "--workload", "real", 
        "--exp", "3", 
        "--n_requests", "1",
        "--use_cpp_scheduler",
        "--cpp_llm_backend", "python", # Fallback to python backend for stability but routed via C++ Scheduler
        "--use_cpp_image"
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(experiments_dir),
        bufsize=1,
        universal_newlines=True,
    )
    
    def stream_reader(pipe, label, is_stderr=False):
        try:
            for line in iter(pipe.readline, ''):
                try:
                    print(f"[{label}] {line.strip()}")
                except UnicodeEncodeError:
                    # Fallback for Windows consoles that can't handle some chars
                    clean_line = line.strip().encode('ascii', 'ignore').decode('ascii')
                    print(f"[{label}] {clean_line}")
                if is_stderr:
                    # Capture stderr for error checking
                    pass 
        except Exception:
            pass

    # Start threads to stream output
    t_out = threading.Thread(target=stream_reader, args=(process.stdout, "EXP_OUT"))
    t_out.daemon = True
    t_out.start()
    
    t_err = threading.Thread(target=stream_reader, args=(process.stderr, "EXP_ERR", True))
    t_err.daemon = True
    t_err.start()

    vram_samples = []
    vram_sampling_running = True
    vram_start = time.time()

    def read_vram_gb():
        try:
            out = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits',
                ],
                text=True,
            ).strip()
            first_line = out.splitlines()[0].strip()
            mem_mib = float(first_line)
            return mem_mib / 1024.0
        except Exception:
            return None

    def vram_sampler():
        while vram_sampling_running and process.poll() is None:
            mem_gb = read_vram_gb()
            if mem_gb is not None:
                vram_samples.append((time.time() - vram_start, mem_gb))
            time.sleep(1.0)

    t_vram = threading.Thread(target=vram_sampler)
    t_vram.daemon = True
    t_vram.start()

    print("Experiment launched. Waiting for model loading (approx 10s)...")
    
    # Monitor for SMI capture
    # We'll try to capture when we see "Exp 3" in stdout or just after a delay
    smi_captured = False
    
    start_time = time.time()
    while process.poll() is None:
        # Check output (non-blocking read is hard without threads, so we just sleep and snapshot)
        time.sleep(2)
        elapsed = time.time() - start_time
        
        # Capture SMI around 15s mark (assuming models loaded and task running)
        if elapsed > 15 and not smi_captured:
            print("Attempting SMI capture...")
            try:
                smi_out = subprocess.check_output(["nvidia-smi"], text=True)
                with open(output_dir / "nvidia_smi_snapshot.txt", "w", encoding="utf-8") as f:
                    f.write(smi_out)
                print(">>> NVIDIA-SMI Snapshot Captured!")
                smi_captured = True
            except Exception as e:
                print(f"SMI Capture failed: {e}")
        
        if elapsed > 120: # Timeout
            print("Timeout! Killing process.")
            process.kill()
            break
            
    process.wait()
    vram_sampling_running = False
    try:
        t_vram.join(timeout=3)
    except Exception:
        pass
    print("Experiment Process Finished.")

    if vram_samples:
        csv_path = output_dir / 'vram_usage_over_time.csv'
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['t_s', 'mem_gb'])
            writer.writeheader()
            for t_s, mem_gb in vram_samples:
                writer.writerow({'t_s': f"{t_s:.3f}", 'mem_gb': f"{mem_gb:.3f}"})
        print(f"Saved VRAM time series to {csv_path}")
    
    if process.returncode != 0:
        print("Experiment Error (check logs above).")
    else:
        print("Experiment Success.")
        
    # 2. Find and Parse Log
    # repo_root/legacy/mvp_core/experiments -> repo_root
    repo_root = experiments_dir.parent.parent.parent
    log_dir = repo_root / "logs"
    log_files = glob.glob(str(log_dir / "benchmark_trace_*.log"))
    
    trace_events = []
    
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Reading log: {latest_log}")
        
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                # We look for lines related to our task
                # Since logging is text, we need to regex parse relevant lines
                # Format: [Time] [Level] [TraceID] [Logger] Message
                # We want to reconstruct: start, complete events
                
                # Sample: [2025-12-08 06:00:00] [INFO] [TraceID:task_...] [ExperimentRunner] Starting task...
                # We map these to JSON events
                
                # Updated mapping based on actual logs
                if "executing task" in line:
                    ts = extract_ts(line)
                    # Extract task ID
                    # ... Worker worker-0 executing task task_97304782_0
                    task_id = line.split("executing task")[-1].strip()
                    trace_events.append({"ts": ts, "event": "task_started", "task_id": task_id, "details": line.strip()})
                elif "Task" in line and "completed" in line:
                    ts = extract_ts(line)
                    # ... Task task_97304782_0 completed
                    task_id = line.split("Task")[-1].split("completed")[0].strip()
                    trace_events.append({"ts": ts, "event": "task_completed", "task_id": task_id, "details": line.strip()})
                elif "scheduled - ID:" in line:
                     ts = extract_ts(line)
                     # ... Task scheduled - ID: task_97304782_0, Name: llm_-1
                     task_id = line.split("ID:")[-1].split(",")[0].strip()
                     trace_events.append({"ts": ts, "event": "task_scheduled", "task_id": task_id, "details": line.strip()})
                     
        # If we found events, dump them
        if trace_events:
            events_path = output_dir / "scheduler_trace_events.json"
            with open(events_path, "w", encoding="utf-8") as f:
                json.dump(trace_events, f, indent=2, ensure_ascii=False)
            print(f"Generated {len(trace_events)} trace events.")

            try:
                from visualize_trace import visualize_trace
                from generate_memory_plot import generate_memory_plot

                visualize_trace(
                    str(events_path),
                    str(output_dir / "scheduler_trace_timeline.png"),
                )
                visualize_trace(
                    str(events_path),
                    str(output_dir / "scheduler_trace_timeline.pdf"),
                )

                generate_memory_plot(
                    events_path,
                    output_dir / 'memory_usage_over_time.pdf',
                )
            except Exception as e:
                print(f"Trace visualization failed: {e}")
    else:
        print("No log file found.")

def extract_ts(line):
    # Extract timestamp from [2025-12-23 21:40:28] or [2025-12-23 21:40:28.123]
    try:
        ts_str = line.split(']')[0].strip('[')
        # Check if there's a dot for milliseconds
        if '.' in ts_str:
            base, ms = ts_str.split('.')
            dt = time.mktime(time.strptime(base, "%Y-%m-%d %H:%M:%S"))
            return dt + float(f"0.{ms}")
        else:
            dt = time.mktime(time.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
            return dt
    except Exception:
        return time.time()

if __name__ == "__main__":
    capture_data()
