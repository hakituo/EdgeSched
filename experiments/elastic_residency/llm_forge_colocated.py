#!/usr/bin/env python3
"""真实 LLM+Forge 共置 smoke/正式矩阵入口。"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.llm_forge_child import run_child  # noqa: E402
from experiments.elastic_residency.llm_forge_parent import run_parent  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-model",
        default="/home/leslie/mvp_core/models/llm/Qwen3.5-9B-Q4_K_M.gguf",
    )
    parser.add_argument("--forge-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--screen-analysis",
        default=(
            "experiment_results/elastic_residency/c3_analysis/"
            "c3_h_direct_nvml_20260831_v1.json"
        ),
    )
    parser.add_argument(
        "--forge-summary",
        default=(
            "experiment_results/elastic_residency/forge/"
            "forge_b1_reprofile_20260831_direct_nvml_v1/summary.json"
        ),
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        help="显式候选 CONTEXT:LLM；缺省时使用 screen 中全部 H_D 边界",
    )
    parser.add_argument(
        "--allow-exploratory-fallback",
        action="store_true",
        help="允许测试 screen 已画像且低于 H_D 边界的驻留档；结果只算探索证据",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--forge-sampler", default="Euler")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--child-timeout", type=float, default=2400.0)
    parser.add_argument("--inter-run-cooldown", type=float, default=2.0)
    parser.add_argument("--max-forge-resident-drift-mb", type=float, default=256.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--gpu-warmup-timeout", type=float, default=5.0)
    parser.add_argument("--max-gpu-sample-age-ms", type=float, default=750.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=4.0)
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT
            / "experiment_results"
            / "elastic_residency"
            / "llm_forge_colocated"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"llm_forge_colocated_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")

    # 仅由父进程构造的 child 参数。
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--llm-config", help=argparse.SUPPRESS)
    parser.add_argument("--context-tokens", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--model-layers", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--child-output-json", help=argparse.SUPPRESS)
    parser.add_argument("--child-output-image", help=argparse.SUPPRESS)
    parser.add_argument("--forge-seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--global-idle-reference-mb", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--safety-margin-mb", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--llm-isolated-increment-mb", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--forge-isolated-increment-mb", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--expected-gpu-total-mb", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-forge-resident-baseline-mb", type=float, help=argparse.SUPPRESS
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.child:
        if not args.llm_config or not args.child_output_json or not args.child_output_image:
            raise SystemExit("child 模式缺少 llm-config/output 参数")
        try:
            result = run_child(args)
        except Exception as exc:
            Path(args.child_output_json).write_text(
                json.dumps(
                    {
                        "config": f"{args.llm_config}@C{args.context_tokens}+Forge",
                        "run_index": args.run_index,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
        Path(args.child_output_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs 必须为正且 --warmup 不能为负")
    if args.max_gpu_sample_age_ms <= 0 or args.max_swap_growth_mb < 0:
        raise SystemExit("GPU 样本年龄必须为正且 swap 增长容忍不能为负")
    if args.max_forge_resident_drift_mb < 0:
        raise SystemExit("Forge 常驻基线漂移容忍不能为负")
    root = run_parent(args)
    print(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
