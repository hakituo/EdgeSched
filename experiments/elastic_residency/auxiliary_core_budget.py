"""D1 辅助任务物理核曲线与 R_sys 隔离敏感性入口（不等同 D3 并发验收）。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psutil

from .auxiliary_child import allowed_topology
from .auxiliary_inputs import file_identity, load_inputs, sha256
from .auxiliary_report import build_plan, summarize
from .contracts import ExperimentArtifactWriter, ExperimentRunConfig, FailureRecord

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path,
                        default=ROOT / "models/auxiliary/bge-small-zh-v1.5")
    parser.add_argument("--whisper-model", type=Path,
                        default=ROOT / "models/auxiliary/faster-whisper-small")
    parser.add_argument("--tasks", nargs="+", default=["json_rag", "bge", "whisper"])
    parser.add_argument("--system-cores", nargs="+", type=int, default=[2])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "experiment_results/elastic_residency/auxiliary_core_budget")
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--source-git-dirty", choices=["true", "false"], required=True)
    parser.add_argument("--source-git-status-entry-count", type=int, required=True)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def model_files(args: argparse.Namespace) -> dict:
    specs = {"bge": (args.bge_model, ("onnx/model.onnx", "tokenizer.json")),
             "whisper": (args.whisper_model, ("model.bin", "config.json", "tokenizer.json"))}
    specs["bge_cached"] = specs["bge"]
    identities = {}
    for task in args.tasks:
        if task not in specs:
            continue
        root, required = specs[task]
        for name in required:
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
        identities[task] = [file_identity(p) for p in sorted(root.rglob("*"))
                            if p.is_file() and (p.suffix in (".json", ".onnx", ".txt")
                                                or p.name == "model.bin")]
    return identities


def source_snapshot(destination: Path) -> dict:
    """WSL 无 .git；额外冻结本入口及全部实验 Python 依赖的实际运行副本。"""
    snapshot = destination / "runtime_experiment_source.zip"
    with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "experiments").rglob("*.py")):
            archive.write(path, path.relative_to(ROOT).as_posix())
    return file_identity(snapshot)


def run(args: argparse.Namespace) -> dict:
    plan = build_plan(args.tasks, args.system_cores, args.repeats, args.seed)
    if args.timeout_s <= 0:
        raise ValueError("子进程超时必须为正")
    if (len(args.source_git_commit) != 40
            or any(c not in "0123456789abcdef" for c in args.source_git_commit)
            or args.source_git_status_entry_count < 0
            or (args.source_git_dirty == "false") != (args.source_git_status_entry_count == 0)):
        raise ValueError("Git 来源必须包含完整提交哈希及一致的工作区状态")
    topology = allowed_topology()
    if max(p["cores"] + p["system_cores"] for p in plan) > len(topology.physical_cores):
        raise ValueError("物理核不足，拒绝缩减请求预算")
    data, _ = load_inputs(args.inputs)
    identities = model_files(args)
    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    root = writer.layout.root
    inputs = root / "inputs"
    inputs.mkdir()
    shutil.copyfile(args.inputs, inputs / "inputs.json")
    for name in data["files"]:
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.inputs.parent / name, target)
    versions = {}
    for package in ("psutil", "numpy", "tokenizers", "onnxruntime", "faster-whisper", "ctranslate2"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {"platform": platform.platform(), "python": sys.executable,
                "python_version": platform.python_version(), "packages": versions,
                "memory": psutil.virtual_memory()._asdict(),
                "swap": psutil.swap_memory()._asdict(), "cpu_topology": topology.to_dict(),
                "models": identities, "input_manifest_sha256": sha256(args.inputs),
                "source_git": {"commit": args.source_git_commit,
                               "dirty": args.source_git_dirty == "true",
                               "status_entry_count": args.source_git_status_entry_count},
                "runtime_source": source_snapshot(root), "seed": args.seed,
                "real_backends": True, "fallback": False,
                "gpu_telemetry": "not_applicable_cpu_only",
                "evidence_scope": "isolated_auxiliary_only",
                "command": sys.argv, "plan": plan}
    writer.initialize(ExperimentRunConfig("auxiliary_core_budget", args.run_id,
                                         seed=args.seed, parameters=vars(args)), manifest)
    results = []
    models = {"bge": str(args.bge_model.resolve()),
              "bge_cached": str(args.bge_model.resolve()),
              "whisper": str(args.whisper_model.resolve())}
    for index, cell in enumerate(plan):
        name = f"cell_{index:03d}_{cell['task']}_r{cell['system_cores']}_c{cell['cores']}"
        config = {"cell": cell, "inputs": str(inputs / "inputs.json"), "models": models}
        config_path = root / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        output = root / "children" / name
        env = {**os.environ, "OMP_NUM_THREADS": str(cell["cores"]),
               "MKL_NUM_THREADS": str(cell["cores"]), "OPENBLAS_NUM_THREADS": str(cell["cores"]),
               "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
               "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": str(args.seed)}
        command = [sys.executable, "-m", "experiments.elastic_residency.auxiliary_child",
                   "--config", str(config_path), "--output", str(output)]
        print(f"[{index + 1}/{len(plan)}] {name}", flush=True)
        try:
            with (root / f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=args.timeout_s)
            record = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            record["child_exit_code"] = completed.returncode
            record["measurement_valid"] &= completed.returncode == 0
        except Exception as exc:
            record = {**cell, "events": [], "measurement_valid": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        record["artifact"] = str(output.relative_to(root))
        results.append(record)
        writer.append_iteration(record)
        if not record["measurement_valid"]:
            writer.append_failure(FailureRecord("child", "InvalidMeasurement",
                                                 record.get("error") or str(record.get("checks")),
                                                 time.monotonic_ns(), attributes=cell))
        summary = summarize(plan, results, len(data["queries"]))
        writer.write_summary(summary)
        if not record["measurement_valid"] and not args.keep_going:
            break
    return summary


if __name__ == "__main__":
    summary = run(parse_args())
    print(json.dumps(summary["minimum_core_budgets"], ensure_ascii=False), flush=True)
    raise SystemExit(0 if summary["measurement_valid"] else 1)
