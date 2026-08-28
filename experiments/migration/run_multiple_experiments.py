import os
import json
import subprocess
import statistics
import time
from typing import List, Dict

def run_experiment(iteration: int) -> Dict:
    print(f"\n[Runner] Starting iteration {iteration}/5...")
    
    # 构建命令
    cmd = [
        "python3", "experiments/migration/resource_migration_experiment.py",
        "--test", "1.4",
        "--seed", str(42 + iteration)  # 改变种子以增加随机性
    ]
    
    # 环境变量 (保持与之前一致)
    env = os.environ.copy()
    # 假设这些变量已经在 shell 环境中设置好了，或者在这里硬编码
    # env["XIAOYOU_LLM_MODEL_PATH"] = ... 
    # env["XIAOYOU_FORGE_BASE_URL"] = ...
    
    try:
        subprocess.run(cmd, check=True, env=env)
        
        # 读取结果
        result_path = "/home/leslie/mvp_core/experiment_results/core/resource_migration_lifecycle.json"
        with open(result_path, "r") as f:
            data = json.load(f)
        
        # 为了避免覆盖，重命名结果文件
        backup_path = result_path.replace(".json", f"_iter{iteration}.json")
        os.rename(result_path, backup_path)
        print(f"[Runner] Iteration {iteration} finished. Result saved to {backup_path}")
        
        return data
        
    except subprocess.CalledProcessError as e:
        print(f"[Runner] Iteration {iteration} failed: {e}")
        return None
    except Exception as e:
        print(f"[Runner] Error: {e}")
        return None

def analyze_results(results: List[Dict]):
    metrics = {
        "llm_gpu_ttft_ms": [],
        "migration_stutter_ms": [],
        "coexistence_cpu_ttft_ms": [],
        "restore_time_ms": [],
        "restore_window_max_gap_ms": []
    }
    
    for res in results:
        if not res: continue
        for key in metrics.keys():
            if key in res:
                metrics[key].append(res[key])
    
    print("\n" + "="*60)
    print("Experiment Summary (N=5)")
    print("="*60)
    
    summary = {}
    
    for key, values in metrics.items():
        if not values:
            print(f"{key}: No data")
            continue
            
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean, "stdev": stdev, "values": values}
        
        print(f"{key:<30}: {mean:>8.1f} ± {stdev:>6.1f} ms  (Min: {min(values):.1f}, Max: {max(values):.1f})")
        
    # 保存汇总
    with open("resource_migration_summary_N5.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to resource_migration_summary_N5.json")

def main():
    results = []
    for i in range(1, 6):
        # 每次实验前稍微等待一下，让系统冷却
        if i > 1:
            print("[Runner] Cooling down for 5 seconds...")
            time.sleep(5)
            
        res = run_experiment(i)
        if res:
            results.append(res)
        else:
            print("[Runner] Warning: Skipping failed iteration")
            
    analyze_results(results)

if __name__ == "__main__":
    main()
