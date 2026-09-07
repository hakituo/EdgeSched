"""双 9B 对比的安全文件准备、计划与离线汇总；不导入模型后端。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import statistics
import tempfile


LARGE_SHA = "423fb3e5fe2c98df616159b7cd1bd04765f67667f5f8712a81e8f63f4e7ee1ef"
SMALL_SHA = "1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab"


def sha256(path: Path) -> str:
    """流式计算文件身份，同时明确预读会影响页缓存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError(f"文件不存在、为软链或 SHA-256 不符，保留原文件：{path}")


def copy_candidate(source: Path, target: Path, expected: str) -> None:
    """同哈希复用；不同哈希绝不覆盖，临时文件校验后独占发布。"""
    verify_file(source, expected)
    if target.exists() or target.is_symlink():
        verify_file(target, expected)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < source.stat().st_size + 1024**3:
        raise ValueError("WSL 剩余空间不足：需模型大小外另留 1 GiB")
    fd, name = tempfile.mkstemp(prefix=".compare-copy-", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        verify_file(temporary, expected)
        temporary.chmod(0o644)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def make_plan(configs: list[str], repeats: int) -> list[dict]:
    if repeats < 1 or not configs or len(set(configs)) != len(configs):
        raise ValueError("重复数须为正数，配置非空且不重复")
    if any(config not in ("FULL", "K0") for config in configs):
        raise ValueError("选型对比仅支持 FULL / K0 端点")
    plan = []
    for round_index in range(1, repeats + 1):
        order = configs if round_index % 2 else list(reversed(configs))
        models = ("large", "small") if round_index % 2 else ("small", "large")
        for config in order:
            for model in models:
                plan.append({"model": model, "config": config, "round": round_index})
    return plan


def measurement(record: dict, timeline: list[dict]) -> dict:
    """只使用有数据的读数；整卡峰值与基线增量分别保存。"""
    if not timeline:
        raise ValueError("子单元缺少资源时间线")
    baseline = record["memory_baseline"]["gpu_card_used_mb"]
    gpu_peak = max(point["gpu_card_used_mb"] for point in timeline)
    return {
        "load_s": record["load_s"],
        "ttft_s": record["inference"]["ttft_s"],
        "prefill_tps": record["inference"]["prefill_tokens_per_s"],
        "decode_tps": record["inference"]["decode_tokens_per_s"],
        "gpu_baseline_mib": baseline,
        "gpu_peak_mib": gpu_peak,
        "gpu_peak_minus_baseline_mib": gpu_peak - baseline,
        "rss_peak_mib": max(point["process_rss_bytes"] for point in timeline) / 2**20,
        "rss_after_inference_mib": record["memory_after_inference"]["rss_bytes"] / 2**20,
        "host_available_min_mib": min(point["system_available_bytes"] for point in timeline) / 2**20,
        "swap_max_mib": max(point["system_swap_used_bytes"] for point in timeline) / 2**20,
        "kv_gpu_mib": record.get("kv_gpu_buffer_mb"),
        "kv_cpu_mib": record.get("kv_cpu_buffer_mb"),
        "compute_gpu_mib": record.get("compute_gpu_buffer_mb"),
        "compute_cpu_mib": record.get("compute_cpu_buffer_mb"),
        "kv_save_s": record["kv_save_s"],
        "kv_restore_s": record["kv_restore_s"],
    }


def summarize(rows: list[dict], plan: list[dict], failures: list[dict]) -> dict:
    """仅汇总通过的记录，单列完整性；三轮不报告可靠 p95 或显著性。"""
    groups = {}
    for model in ("large", "small"):
        for config in dict.fromkeys(unit["config"] for unit in plan):
            selected = [r for r in rows if r["passed"] and r["model"] == model and r["config"] == config]
            metrics = {}
            if selected:
                for key in selected[0]["metrics"]:
                    values = [r["metrics"][key] for r in selected]
                    metrics[key] = None if any(v is None for v in values) else {
                        "mean": statistics.mean(values), "min": min(values), "max": max(values),
                    }
            groups[f"{model}|{config}"] = {"samples": len(selected), "metrics": metrics}
    pairs = []
    for small in rows:
        if small["model"] != "small":
            continue
        large = next((r for r in rows if r["model"] == "large" and
                      (r["config"], r["round"]) == (small["config"], small["round"])), None)
        if large is None or not small["passed"] or not large["passed"]:
            continue
        pairs.append({
            "config": small["config"], "round": small["round"],
            "prompt_tokens_match": small["prompt_tokens_sha256"] == large["prompt_tokens_sha256"],
            "small_minus_large": {
                key: small["metrics"][key] - large["metrics"][key]
                for key in small["metrics"]
                if small["metrics"][key] is not None and large["metrics"][key] is not None
            },
        })
    complete = len(rows) == len(plan) and all(r["passed"] for r in rows) and not failures
    return {
        "passed": complete and len(pairs) * 2 == len(plan) and all(p["prompt_tokens_match"] for p in pairs),
        "expected_records": len(plan), "actual_records": len(rows), "failures": failures,
        "by_model_config": groups, "pairs": pairs, "records": rows,
        "scope": "纯文本端点选型；非完整驻留曲线、视觉/共置或回答质量验收",
    }


def markdown_report(summary: dict) -> str:
    lines = ["# 双 9B 端点对比", "", f"通过：{summary['passed']}；记录：{summary['actual_records']}/{summary['expected_records']}",
             "", "各单元独立进程，加载前会完整读取模型计算哈希；不是磁盘冷启动。", "",
             "| 模型 / 配置 | 有效轮数 | 加载 mean s | TTFT mean s | decode mean tok/s | 整卡峰值 mean MiB | 峰值减基线 mean MiB | RSS峰值 mean MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ("load_s", "ttft_s", "decode_tps", "gpu_peak_mib", "gpu_peak_minus_baseline_mib", "rss_peak_mib")
    for label, group in summary["by_model_config"].items():
        values = [f"{group['metrics'][key]['mean']:.3f}" if group["metrics"].get(key) else "—" for key in keys]
        display_label = label.replace("|", " / ")
        lines.append(f"| {display_label} | {group['samples']} | " + " | ".join(values) + " |")
    lines.extend(["", summary["scope"], "", "不完整或配对 prompt 不一致时不可据表选型。输出为固定填充 prompt 的续写，不是自然问答质量评测。"])
    return "\n".join(lines) + "\n"
