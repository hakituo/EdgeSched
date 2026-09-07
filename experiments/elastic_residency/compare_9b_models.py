"""用户手动启动双 9B 纯文本端点对比，复用现有驻留测量器。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile

from experiments.elastic_residency.compare_9b_support import (
    LARGE_SHA, SMALL_SHA, copy_candidate, make_plan, markdown_report,
    measurement, sha256, summarize, verify_file,
)
from experiments.elastic_residency.vision.vision_artifacts import (
    describe_gguf_artifact, main_model_profile,
)


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--large-model", default="/home/leslie/mvp_core/models/llm/Qwen3.5-9B-q4_k_m.gguf")
    parser.add_argument("--small-source", default="/mnt/d/AI/xiaoyou-core/models/llm/Qwen3.5-9B-Q4_K_M.gguf")
    parser.add_argument("--small-model", default="/home/leslie/mvp_core/models/llm/comparison_1a35ab89/model.gguf")
    parser.add_argument("--configs", nargs="+", choices=("FULL", "K0"), default=["FULL", "K0"])
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument("--child-timeout", type=int, default=1800)
    parser.add_argument("--output-root", default=str(ROOT / "experiment_results/elastic_residency/model_comparison"))
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--source-git-dirty", required=True, choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", required=True, type=int)
    return parser


def prepare_models(args: argparse.Namespace) -> dict:
    large, small = Path(args.large_model), Path(args.small_model)
    for path in (large, small):
        if not path.is_absolute() or path.resolve() != path or str(path).startswith("/mnt/"):
            raise ValueError(f"运行模型须使用 WSL 原生非软链绝对路径：{path}")
    if large == small:
        raise ValueError("大小模型目标路径必须不同")
    print("核验 WSL 大模型哈希……", flush=True)
    verify_file(large, LARGE_SHA)
    print("复制/核验小模型的 WSL 独立副本……", flush=True)
    copy_candidate(Path(args.small_source), small, SMALL_SHA)
    artifacts = {}
    for label, path, digest in (("large", large, LARGE_SHA), ("small", small, SMALL_SHA)):
        artifact = describe_gguf_artifact(path, compute_sha256=False)
        profile = main_model_profile(artifact)
        if profile["architecture"] != "qwen35" or profile["block_count"] != 32:
            raise ValueError(f"模型结构不符：{label}")
        if profile["context_length"] < args.context_tokens + args.generation_tokens:
            raise ValueError("上下文预算超过模型声明")
        artifacts[label] = {"path": str(path), "sha256": digest,
                            "size_bytes": path.stat().st_size, "profile": profile}
    return artifacts


def unit_command(args: argparse.Namespace, root: Path, unit: dict, artifacts: dict) -> list[str]:
    return [sys.executable, "-m", "experiments.elastic_residency.llm_residency_curve",
            "--model", artifacts[unit["model"]]["path"], "--configs", unit["config"],
            "--runs", "1", "--context-tokens", str(args.context_tokens),
            "--generation-tokens", str(args.generation_tokens),
            "--llm-cores", str(args.llm_cores), "--system-cores", str(args.system_cores),
            "--seed", str(42 + unit["round"] - 1), "--jitter-gate-mode", "observe",
            "--child-timeout", str(args.child_timeout), "--output-root", str(root / "units"),
            "--run-id", unit["id"], "--source-git-commit", args.source_git_commit,
            "--source-git-dirty", args.source_git_dirty,
            "--source-git-status-entry-count", str(args.source_git_status_entry_count),
            "--source-git-snapshot", str(root / "source_git_snapshot.tar.gz")]


def execute(command: list[str], log: Path, timeout: int) -> int:
    """中断/超时清理本单元进程组，不影响其他用户进程。"""
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def audit_unit(root: Path, unit: dict, artifact: dict, args: argparse.Namespace) -> dict:
    """不只相信 summary.passed，同时检查 raw child、后端和参数身份。"""
    def read(name: str):
        return json.loads((root / name).read_text(encoding="utf-8"))
    summary = read("summary.json")
    if not summary["passed"] or summary["actual_records"] != 1 or summary["failure_count"]:
        raise ValueError("单元未完整通过")
    cfg = read("run_config.json")
    backend = cfg["backends"][0]
    if backend["model_sha256"] != artifact["sha256"] or not backend["is_real"] or backend["fallback_occurred"]:
        raise ValueError("单元模型身份或真实后端不符")
    record = summary["records"][0]
    raw = [json.loads(line) for line in (root / "iterations.jsonl").read_text(encoding="utf-8").splitlines()]
    if raw != [record] or (root / "failures.jsonl").read_text(encoding="utf-8").strip():
        raise ValueError("原始记录与 summary 不一致")
    child = read(f"C{args.context_tokens}_{unit['config']}_r1_child.json")
    if any(child.get(k) != v for k, v in record.items() if k != "matrix_execution_ordinal"):
        raise ValueError("child 与汇总记录不一致")
    checks = record["acceptance_checks"]
    if not checks or not all(v is True for v in checks.values()):
        raise ValueError("真实性门禁失败")
    for key, expected in (("config", unit["config"]), ("context_tokens", args.context_tokens),
                          ("generation_tokens", args.generation_tokens), ("llm_physical_cores", args.llm_cores),
                          ("system_physical_cores", args.system_cores)):
        if record[key] != expected:
            raise ValueError(f"运行参数不符：{key}")
    source = read("manifest.json")["git"]["source_copy"]
    if not summary["git_provenance_complete"] or sha256(root / source["snapshot"]) != source["snapshot_sha256"]:
        raise ValueError("源码快照核验失败")
    inference = record["inference"]
    # 当前驻留入口只保存前 200 字预览和完整输出哈希，没有全文字段。
    preview = inference.get("output_text_preview")
    output_sha = inference.get("output_text_sha256")
    if not isinstance(preview, str) or not isinstance(output_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", output_sha):
        raise ValueError("单元缺少有效的输出预览或完整输出 SHA-256")
    return {**unit, "passed": True, "metrics": measurement(record, child["timeline"]),
            "prompt_tokens_sha256": record["prompt_tokens_sha256"],
            "performance_target_passed": record["performance_target_passed"],
            "output_text_preview": preview, "output_text_sha256": output_sha,
            "output_text_scope": "preview_only_not_full_text",
            "source_artifact": str(root)}


def run(args: argparse.Namespace) -> Path:
    if sys.platform != "linux":
        raise ValueError("真实对比只允许在 WSL/Linux 执行")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        raise ValueError("run id 格式非法")
    if min(args.context_tokens, args.generation_tokens) < 2 or args.llm_cores < 1 or args.system_cores < 0 or args.child_timeout < 1:
        raise ValueError("token、核数或超时预算非法")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_git_commit):
        raise ValueError("需要完整源提交哈希")
    count = args.source_git_status_entry_count
    if count < 0 or (args.source_git_dirty == "true") != (count > 0):
        raise ValueError("Git dirty 与状态条目数矛盾")
    plan = make_plan(args.configs, args.repeats)
    for i, unit in enumerate(plan, 1):
        unit["id"] = f"{i:02}_{unit['model']}_{unit['config']}_r{unit['round']}"
    root = Path(args.output_root).resolve() / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    rows, failures = [], []
    write_json(root / "plan.json", {"args": vars(args), "plan": plan, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    try:
        artifacts = prepare_models(args)
        with tarfile.open(root / "source_git_snapshot.tar.gz", "w:gz") as archive:
            archive.add(ROOT / "experiments", arcname="experiments", filter=lambda item:
                        None if "__pycache__" in Path(item.name).parts or item.name.endswith(".pyc") else item)
        write_json(root / "models.json", artifacts)
        for ordinal, unit in enumerate(plan, 1):
            print(f"[{ordinal}/{len(plan)}] {unit['model']} {unit['config']} r{unit['round']}", flush=True)
            command = unit_command(args, root, unit, artifacts)
            write_json(root / f"{unit['id']}_command.json", command)
            code = execute(command, root / f"{unit['id']}.log", args.child_timeout + 180)
            if code != 0:
                raise RuntimeError(f"{unit['id']} 退出码 {code}，失败即停")
            rows.append(audit_unit(root / "units" / unit["id"], unit, artifacts[unit["model"]], args))
            current = summarize(rows, plan, failures)
            if any(not pair["prompt_tokens_match"] for pair in current["pairs"]):
                raise ValueError("同轮 prompt token 身份不一致，停止直接对比")
            write_json(root / "summary.json", current)
    except BaseException as exc:
        failures.append({"error": f"{type(exc).__name__}: {exc}", "completed_records": len(rows)})
        raise
    finally:
        summary = summarize(rows, plan, failures)
        write_json(root / "summary.json", summary)
        (root / "comparison.md").write_text(markdown_report(summary), encoding="utf-8")
        print(f"结果：{root}", flush=True)
    return root


if __name__ == "__main__":
    run(build_parser().parse_args())
