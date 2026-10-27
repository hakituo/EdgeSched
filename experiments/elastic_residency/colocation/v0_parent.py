"""V0 共跑减速诊断编排：逐单元起独立子进程，汇总为 manifest / cells.jsonl / summary.json。

用法（先 ``source scripts/experiments/edgesched_env.sh``）::

    $EDGESCHED_PYTHON -m experiments.elastic_residency.colocation.v0_parent --rounds 2

核划分沿用 S0：LLM 14 个物理核（含 SMT 兄弟，共 28 个逻辑 CPU），剩下的 2 个
物理核留给 ``tts_proc_disjoint`` 的 TTS 进程。多轮时各轮单元顺序交替，减少漂移。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.colocation.v0_cells import (  # noqa: E402
    select_cells,
    split_cores,
    summarize,
)
from experiments.elastic_residency.cpu_topology import CPUTopology  # noqa: E402
from experiments.tts.profiling_utils import collect_git_state, sha256_file  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models/llm/Qwen3.5-9B-Q4_K_M.gguf"
EXPECTED_SHA = "1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab"
CELL_TIMEOUT_S = 1200


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("llama_cpp", "torch", "faster_qwen3_tts", "numpy"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "?"))
        except Exception as exc:  # pragma: no cover
            versions[name] = f"unavailable: {type(exc).__name__}"
    return versions


def _driver() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except OSError:  # pragma: no cover
        return ""


def run_one(cell_name: str, args: argparse.Namespace, out: Path, llm: list[int], tts: list[int]) -> dict[str, Any]:
    cmd = [
        sys.executable, "-m", "experiments.elastic_residency.colocation.v0_child",
        "--cell", cell_name, "--out", str(out),
        "--llm-affinity", ",".join(map(str, llm)),
        "--tts-affinity", ",".join(map(str, tts)),
        "--model-sha256", args.model_sha256,
        "--llm-model", str(args.llm_model),
        "--n-gpu-layers", str(args.n_gpu_layers),
        "--reps", str(args.reps),
    ]
    started = time.monotonic()
    log = out.with_suffix(".log")
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                              timeout=CELL_TIMEOUT_S, check=False)
    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
    else:
        result = {"cell": {"name": cell_name}, "passed": False,
                  "error": f"子进程无输出，返回码 {proc.returncode}，见 {log.name}"}
    result["child_returncode"] = proc.returncode
    result["wall_s"] = round(time.monotonic() - started, 1)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cells", nargs="*")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-gpu-layers", type=int, default=24)
    parser.add_argument("--llm-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-sha256", default="", help="留空则现场计算")
    parser.add_argument("--llm-physical", type=int, default=14)
    parser.add_argument("--tts-physical", type=int, default=2)
    parser.add_argument("--run-id", default=f"colocation_v0_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "experiment_results/elastic_residency/colocation_v0")
    args = parser.parse_args(argv)
    if not args.model_sha256:
        args.model_sha256 = sha256_file(args.llm_model.resolve())
    if args.model_sha256 != EXPECTED_SHA:
        print(f"模型哈希不符：{args.model_sha256}", file=sys.stderr)
        return 2
    cells = select_cells(args.cells)
    topology = CPUTopology.discover()
    groups = [list(core.logical_cpus) for core in topology.physical_cores]
    llm, tts = split_cores(groups, args.llm_physical, args.tts_physical)
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "edgesched.colocation_v0.v1",
        "run_id": args.run_id,
        "started_at": dt.datetime.now().astimezone().isoformat(),
        "git": collect_git_state(PROJECT_ROOT),
        "model_path": str(args.llm_model.resolve()),
        "model_sha256": args.model_sha256,
        "n_gpu_layers": args.n_gpu_layers,
        "reps_per_cell": args.reps,
        "rounds": args.rounds,
        "cells": [cell.to_dict() for cell in cells],
        "llm_affinity": llm,
        "tts_disjoint_affinity": tts,
        "topology_physical_cores": len(groups),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "driver": _driver(),
        "packages": _package_versions(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results: list[dict[str, Any]] = []
    with (run_dir / "cells.jsonl").open("w", encoding="utf-8") as sink:
        for round_index in range(args.rounds):
            order = cells if round_index % 2 == 0 else list(reversed(cells))
            for cell in order:
                out = run_dir / f"r{round_index + 1}_{cell.name}.json"
                result = run_one(cell.name, args, out, llm, tts)
                result["round"] = round_index + 1
                results.append(result)
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                row = summarize([result])[cell.name]
                print(f"[r{round_index + 1}] {cell.name:20s} passed={row['passed']} "
                      f"tok/s={row['decode_tok_s']} tts_rtf={row['tts_rtf']} err={row['error']}", flush=True)
    summary = {"by_cell": summarize(results), "failures": [r["cell"]["name"] for r in results if not r.get("passed")]}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["by_cell"], ensure_ascii=False, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
