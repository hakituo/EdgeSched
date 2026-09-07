"""离线重算辅助核数画像，核验子进程原始事件并输出 CSV/可选曲线。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .auxiliary_inputs import load_inputs, sha256
from .auxiliary_report import summarize


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audit(run_root: Path) -> dict:
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    inputs, _ = load_inputs(run_root / "inputs/inputs.json")
    if sha256(run_root / "inputs/inputs.json") != manifest["input_manifest_sha256"]:
        raise ValueError("输入 manifest 哈希不一致")
    snapshot = run_root / Path(manifest["runtime_source"]["path"]).name
    if sha256(snapshot) != manifest["runtime_source"]["sha256"]:
        raise ValueError("运行源码快照哈希不一致")
    records = read_jsonl(run_root / "iterations.jsonl")
    artifacts = set()
    for record in records:
        child = (run_root / record["artifact"]).resolve()
        if run_root.resolve() not in child.parents or child in artifacts:
            raise ValueError("子进程路径越界或被重复使用")
        artifacts.add(child)
        summary = json.loads((child / "summary.json").read_text(encoding="utf-8"))
        events = read_jsonl(child / "iterations.jsonl")
        if record["events"] != events or summary["events"] != events:
            raise ValueError("父/子原始事件不一致")
        if record.get("preparation") != summary.get("preparation"):
            raise ValueError("父/子候选预计算证据不一致")
        if any(record[key] != summary[key] for key in ("task", "cores", "system_cores", "repeat")):
            raise ValueError("父/子配置不一致")
        valid = summary["measurement_valid"] and all(summary["checks"].values())
        if record["measurement_valid"] != (valid and record["child_exit_code"] == 0):
            raise ValueError("父/子有效性或退出码不一致")
        timeline = read_jsonl(child / "timeline.jsonl")
        if valid and (len(timeline) < 2 or timeline[0]["cpu_sample_valid"]
                      or not any(s["cpu_sample_valid"] for s in timeline[1:])):
            raise ValueError("CPU 首样本/差分样本契约无效")
    result = summarize(manifest["plan"], records, len(inputs["queries"]))
    if result != {k: v for k, v in json.loads((run_root / "summary.json").read_text(
            encoding="utf-8")).items() if k != "schema_version"}:
        raise ValueError("离线重算与原 summary 不一致")
    return {**result, "audit": {"passed": True, "child_artifacts_checked": len(artifacts),
                                "raw_events_checked": sum(len(r["events"]) for r in records),
                                "analyzer_sha256": sha256(Path(__file__))}}


def write_csv(result: dict, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in result["groups"] for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, dict)
                          else value for key, value in row.items()} for row in result["groups"])


def plot_curves(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({g["task"] for g in result["groups"]})
    figure, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        groups = [g for g in result["groups"] if g["task"] == task]
        scale = 1 if task == "whisper" else 1000
        for reserve in sorted({g["system_cores"] for g in groups}):
            rows = sorted([g for g in groups if g["system_cores"] == reserve
                           and g["measurement_valid"] and g["p95"] is not None],
                          key=lambda g: g["cores"])
            if rows:
                axis.plot([g["cores"] for g in rows], [g["p95"] * scale for g in rows],
                          marker="o", label=f"R_sys={reserve}")
        axis.axhline(groups[0]["sla"] * scale, color="black", ls="--", label="SLA")
        axis.set(title=task, xlabel="Physical CPU cores",
                 ylabel="p95 RTF" if task == "whisper" else "p95 latency (ms)")
        axis.set_xticks(sorted({g["cores"] for g in groups}))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Isolated auxiliary workloads: reservation sensitivity (not D3)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    write_csv(result, args.output.with_suffix(".csv"))
    if args.plot:
        if args.plot.exists():
            raise FileExistsError(args.plot)
        plot_curves(result, args.plot)
    print(json.dumps(result["audit"]))
