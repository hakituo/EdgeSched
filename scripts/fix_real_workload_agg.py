import json
import time
from pathlib import Path

raw_dir = Path("paper/experiment_data/03_raw_iterations")
agg_dir = Path("paper/experiment_data/01_paper_main")
agg_dir.mkdir(parents=True, exist_ok=True)

all_data = []
for i in range(1, 6):
    f = raw_dir / f"real_workload_iter{i}.json"
    if f.exists():
        with open(f) as fp:
            d = json.load(fp)
        all_data.append(d)
        event_count = len(d.get("data", d)) if isinstance(d, dict) else len(d)
        print(f"  读取 iter{i}: {event_count} 条事件")

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "iterations": len(all_data),
    "data": all_data,
}
out = agg_dir / "real_workload_trace.json"
with open(out, "w", encoding="utf-8") as fp:
    json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)
print(f"聚合数据已保存: {out}")
