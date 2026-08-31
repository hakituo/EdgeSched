"""端点 smoke 的命令行契约与子进程命令构造。

父进程与子进程共用同一个 parser：父进程负责 artifact 门禁与编排，
带 `--child-config` 时进入单轮测量子进程，避免两处重复声明参数。
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from experiments.elastic_residency.vision.vision_evidence import ALL_STATES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "/home/leslie/mvp_core/models/llm/Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_MMPROJ = "/home/leslie/mvp_core/models/llm/mmproj-BF16.gguf"
DEFAULT_QUESTION = "请用中文描述这张图片里的形状和颜色。"
DEFAULT_FIXED_IMAGE = (
    PROJECT_ROOT
    / "experiment_results"
    / "elastic_residency"
    / "vision_inputs"
    / "edgesched_vision_shapes_448_v1.png"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiment_results" / "elastic_residency" / "vision_endpoint"
)

CHILD_FORWARDED = (
    "model",
    "mmproj",
    "image",
    "question",
    "prompt_mode",
    "vision_eval_mode",
    "context_tokens",
    "generation_tokens",
    "llm_cores",
    "system_cores",
    "sample_interval",
    "gpu_sample_interval",
    "gpu_warmup_timeout",
    "max_gpu_sample_age_ms",
    "max_swap_growth_mb",
    "image_max_tokens",
    "image_min_tokens",
)


def child_command(
    args: argparse.Namespace,
    *,
    entry_script: Path,
    config: str,
    state: str,
    run_index: int,
    model_layers: int,
    expected_visual_tokens: int | None,
    child_json: Path,
) -> list[str]:
    """构造单轮测量子进程命令；参数只从父进程 namespace 转发，不复制默认值。"""

    command = [
        sys.executable,
        str(entry_script),
        "--child-config",
        config,
        "--residency-state",
        state,
        "--run-index",
        str(run_index),
        "--model-layers",
        str(model_layers),
        "--child-output-json",
        str(child_json),
        "--seed",
        str(args.seed + run_index - 1),
    ]
    for name in CHILD_FORWARDED:
        value = getattr(args, name)
        if value is None:
            continue
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    if expected_visual_tokens is not None:
        command.extend(["--expected-visual-tokens", str(expected_visual_tokens)])
    return command


def build_parser() -> argparse.ArgumentParser:
    """端点 smoke 的完整参数契约（父进程 + 子进程）。"""

    parser = argparse.ArgumentParser(
        description="Qwen3.5-4B + mmproj 文本/视觉端点 smoke"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", default=DEFAULT_MMPROJ)
    parser.add_argument(
        "--image", default=None, help="外部固定图片；缺省使用内置确定性图片"
    )
    parser.add_argument("--fixed-image-path", default=str(DEFAULT_FIXED_IMAGE))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--configs", nargs="+", default=["K0", "FULL"])
    parser.add_argument(
        "--residency-states",
        nargs="+",
        default=["TEXT_ONLY", "VISION_WARM_GPU"],
        help=f"可选状态: {', '.join(ALL_STATES)}",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt-mode",
        choices=("model_template", "qwen_im"),
        default="model_template",
    )
    parser.add_argument(
        "--vision-eval-mode",
        choices=("split_encode", "chunk_single"),
        default="split_encode",
    )
    parser.add_argument("--image-max-tokens", type=int, default=0)
    parser.add_argument("--image-min-tokens", type=int, default=0)
    parser.add_argument("--expected-block-count", type=int, default=32)
    parser.add_argument("--expect-model-sha256", default="")
    parser.add_argument("--expect-model-size-bytes", type=int)
    parser.add_argument("--expect-mmproj-sha256", default="")
    parser.add_argument("--expect-mmproj-size-bytes", type=int)
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--gpu-warmup-timeout", type=float, default=5.0)
    parser.add_argument("--max-gpu-sample-age-ms", type=float, default=750.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--child-timeout", type=float, default=1800.0)
    parser.add_argument("--inter-run-cooldown", type=float, default=1.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--run-id",
        default=f"v1_vision_endpoint_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")
    parser.add_argument("--child-config")
    parser.add_argument("--residency-state")
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--expected-visual-tokens", type=int)
    parser.add_argument("--child-output-json")
    return parser
