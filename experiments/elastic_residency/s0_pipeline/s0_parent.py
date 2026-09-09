"""S0 流水线 parent：四配置 pilot 编排、门禁聚合与产物写入。

parent 不加载模型，只负责按冻结顺序调用 child、收集结果并执行门禁；
child 失败默认 fail-fast，pilot 阶段可用 ``--continue-on-failure`` 收全证据。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.elastic_residency.contracts import (
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentRunConfig,
    FailureRecord,
)
from experiments.elastic_residency.s0_pipeline.config import (  # noqa: E402
    FREEZE_STATUS,
    S0_PLACEMENTS,
    default_output_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHILD_MODULE = "experiments.elastic_residency.s0_pipeline.s0_child"


def _run_child(  # noqa: PLR0913
    *, placement: str, args: argparse.Namespace, output_json: Path, run_index: int
) -> tuple[dict[str, Any], str]:
    command = [
        sys.executable,
        "-m",
        CHILD_MODULE,
        "--placement",
        placement,
        "--llm-model",
        args.llm_model,
        "--tts-model",
        args.tts_model,
        "--ref-audio",
        args.ref_audio,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--model-layers",
        str(args.model_layers),
        "--request-id",
        f"{placement.replace(':', '-')}-r{run_index}",
        "--child-output-json",
        str(output_json),
        "--llm-cores",
        str(args.llm_cores),
        "--system-cores",
        str(args.system_cores),
    ]
    if args.enable_playback:
        command.append("--enable-playback")
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    stderr = completed.stderr.strip()
    if output_json.is_file():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    else:
        payload = {
            "passed": False,
            "placement": placement,
            "error": stderr or f"child 退出码 {completed.returncode}",
        }
    payload["child_returncode"] = completed.returncode
    return payload, stderr


def _gate_checks(record: dict[str, Any]) -> dict[str, bool]:
    run = record.get("run", {})
    semantics = run.get("semantics", {})
    swap = record.get("swap_report", {})
    return {
        "child_passed": record.get("passed") is True,
        "has_real_llm_identity": bool(record.get("llm_identity")),
        "has_events": run.get("event_count", 0) > 0,
        "has_execution_intervals": bool(run.get("tts_execution_intervals")),
        "swap_within_tolerance": bool(swap.get("within_tolerance")),
        "no_proxy_only_evidence": (
            semantics.get("execution_event_count", 0) > 0
        ),
    }


def run_parent(args: argparse.Namespace) -> Path:
    for label, path in (
        ("llm_model", args.llm_model),
        ("tts_model", args.tts_model),
        ("ref_audio", args.ref_audio),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} 不存在: {path}")
    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    config = ExperimentRunConfig(
        experiment_name="s0_pipeline",
        run_id=args.run_id,
        backends=(
            BackendEvidence(
                component="llm",
                backend_kind="llama_cpp_python",
                is_real=True,
                model_path=args.llm_model,
            ),
            BackendEvidence(
                component="tts",
                backend_kind="qwen3_tts_local",
                is_real=True,
                model_path=args.tts_model,
            ),
        ),
        parameters={
            "placements": list(args.placements),
            "runs": args.runs,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "enable_playback": args.enable_playback,
        },
    )
    writer.initialize(
        config,
        {
            "freeze_status": FREEZE_STATUS,
            "purpose": "pilot_only_no_performance_conclusion",
            "placements_frozen": list(S0_PLACEMENTS),
        },
    )
    summary: dict[str, Any] = {
        "placements": list(args.placements),
        "runs": args.runs,
        "records": [],
        "all_passed": True,
    }
    for placement in args.placements:
        for run_index in range(1, args.runs + 1):
            started = time.monotonic()
            output_json = (
                writer.layout.root / "children" / f"{placement.replace(':', '-')}_r{run_index}.json"
            )
            output_json.parent.mkdir(parents=True, exist_ok=True)
            record, stderr = _run_child(
                placement=placement,
                args=args,
                output_json=output_json,
                run_index=run_index,
            )
            record["placement"] = placement
            record["run_index"] = run_index
            record["parent_wall_s"] = time.monotonic() - started
            record["acceptance_checks"] = _gate_checks(record)
            record["passed"] = all(record["acceptance_checks"].values())
            events = record.get("run", {}).get("events", [])
            for event in events:
                writer.append_event(event)
            writer.append_iteration(record)
            summary["records"].append(
                {
                    "placement": placement,
                    "run_index": run_index,
                    "passed": record["passed"],
                    "checks": record["acceptance_checks"],
                    "error": record.get("error", ""),
                    "sentence_count": record.get("run", {}).get("sentence_count", 0),
                }
            )
            if not record["passed"]:
                writer.append_failure(
                    FailureRecord(
                        stage="s0_pipeline_child",
                        error_type="GateFailure",
                        message=record.get("error") or stderr or "门禁未通过",
                        monotonic_ns=time.monotonic_ns(),
                        fatal=not args.continue_on_failure,
                        attributes={"placement": placement},
                    )
                )
                summary["all_passed"] = False
                if not args.continue_on_failure:
                    writer.write_summary(summary)
                    return writer.layout.root
    writer.write_summary(summary)
    return writer.layout.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placements", nargs="+", default=list(S0_PLACEMENTS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--llm-model",
        default="/home/leslie/mvp_core/models/llm/Qwen3.5-9B-q4_k_m.gguf",
    )
    parser.add_argument(
        "--tts-model",
        default="/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base",
    )
    parser.add_argument(
        "--ref-audio", default="/home/leslie/mvp_core/models/tts_refs/ref_calm.wav"
    )
    parser.add_argument("--prompt", default="请用三到五句话介绍边缘计算的优势。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--enable-playback", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument(
        "--output-root", default=str(default_output_root(PROJECT_ROOT))
    )
    parser.add_argument(
        "--run-id",
        default=f"s0_pipeline_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = run_parent(args)
    print(json.dumps({"artifacts": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
