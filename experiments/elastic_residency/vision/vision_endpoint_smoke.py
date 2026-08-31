#!/usr/bin/env python3
"""V1 多模态端点 smoke：Qwen3.5-4B × (K0/FULL) × (TEXT_ONLY/VISION_WARM)。

先过 artifact 与 mmproj 能力门禁，再逐单元在独立子进程中真实执行：纯文本轮次只
加载语言模型，视觉轮次用 libmtmd 对固定图片实际执行 projector，并把视觉 token
送入同一个语言模型。产物写入 `experiment_results`，不自动晋升论文正文数据。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.contracts import (  # noqa: E402
    ExperimentArtifactWriter,
    ExperimentEvent,
    FailureRecord,
    TimelineSample,
)
from experiments.elastic_residency.tts_component_offload import _git_source  # noqa: E402
from experiments.elastic_residency.vision.vision_endpoint_child import (  # noqa: E402
    run_child,
)
from experiments.elastic_residency.vision.vision_endpoint_cli import (  # noqa: E402
    build_parser,
    child_command,
)
from experiments.elastic_residency.vision.vision_endpoint_contracts import (  # noqa: E402
    build_run_config,
)
from experiments.elastic_residency.vision.vision_endpoint_gate import (  # noqa: E402
    build_endpoint_gate,
)
from experiments.elastic_residency.vision.vision_endpoint_report import (  # noqa: E402
    summarize_by_cell,
    write_summary_csv,
)
from experiments.elastic_residency.vision.vision_evidence import (  # noqa: E402
    ALL_STATES,
)


def build_plan(
    configs: tuple[str, ...],
    states: tuple[str, ...],
    runs: int,
) -> list[tuple[str, str, int, int]]:
    """按 (配置 → 状态 → 轮次) 生成确定性执行顺序。"""

    if not configs or len(set(configs)) != len(configs):
        raise ValueError("--configs 必须非空且不能重复")
    if not states or len(set(states)) != len(states):
        raise ValueError("--residency-states 必须非空且不能重复")
    for state in states:
        if state not in ALL_STATES:
            raise ValueError(f"未知驻留状态: {state}")
    if runs < 1:
        raise ValueError("--runs 必须大于 0")
    plan: list[tuple[str, str, int, int]] = []
    ordinal = 0
    for config in configs:
        for state in states:
            for run_index in range(1, runs + 1):
                ordinal += 1
                plan.append((config, state, run_index, ordinal))
    return plan


def _count_failures(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _run_units(
    args: argparse.Namespace,
    *,
    writer: ExperimentArtifactWriter,
    plan: list[tuple[str, str, int, int]],
    gate: dict[str, Any],
    model_layers: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for config_name, state, run_index, ordinal in plan:
        label = f"{config_name}_{state}_r{run_index}"
        print(f"[{ordinal}/{len(plan)}] run {label}", flush=True)
        child_json = writer.layout.root / f"{label}_child.json"
        child_log = writer.layout.root / f"{label}.log"
        completed = subprocess_run(
            child_command(
                args,
                entry_script=Path(__file__).resolve(),
                config=config_name,
                state=state,
                run_index=run_index,
                model_layers=model_layers,
                expected_visual_tokens=gate["expected_visual_tokens"],
                child_json=child_json,
            ),
            timeout=args.child_timeout,
        )
        child_log.write_text(
            completed["stdout"] + "\n[stderr]\n" + completed["stderr"],
            encoding="utf-8",
        )
        if completed["returncode"] != 0 or not child_json.is_file():
            writer.append_failure(
                FailureRecord(
                    stage=label,
                    error_type="ChildProcessFailure",
                    message=f"returncode={completed['returncode']}",
                    monotonic_ns=time.monotonic_ns(),
                    attributes={"log": child_log.name},
                )
            )
            if not args.continue_on_failure:
                break
            continue
        record = json.loads(child_json.read_text(encoding="utf-8"))
        record["matrix_execution_ordinal"] = ordinal
        for sample in record.pop("timeline", []):
            writer.append_timeline(TimelineSample(**sample))
        writer.append_event(
            ExperimentEvent(
                name="vision_endpoint_child",
                monotonic_ns=record["child_started_monotonic_ns"],
                wall_time_iso=record["child_started_wall_iso"],
                phase=label,
                attributes={
                    "residency_state": state,
                    "config": config_name,
                    "run_index": run_index,
                    "matrix_execution_ordinal": ordinal,
                },
            )
        )
        writer.append_iteration(record)
        records.append(record)
        failed = [name for name, ok in record["acceptance_checks"].items() if not ok]
        suffix = f" failed_checks={','.join(failed)}" if failed else ""
        print(
            f"[{ordinal}/{len(plan)}] complete passed={record['passed']}{suffix}",
            flush=True,
        )
        if not record["passed"] and not args.continue_on_failure:
            break
        if args.inter_run_cooldown > 0:
            time.sleep(args.inter_run_cooldown)
    return records


def subprocess_run(command: list[str], *, timeout: float) -> dict[str, Any]:
    """执行子进程并返回结构化结果（便于静态测试替换）。"""

    import subprocess  # noqa: PLC0415

    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_parent(args: argparse.Namespace) -> Path:
    gate = build_endpoint_gate(
        model_path=args.model,
        mmproj_path=args.mmproj,
        fixed_image_path=args.fixed_image_path,
        external_image_path=args.image,
        expected_block_count=args.expected_block_count,
        required_context_tokens=args.context_tokens + args.generation_tokens,
        declared={
            "main_model_sha256": args.expect_model_sha256,
            "main_model_size_bytes": args.expect_model_size_bytes,
            "projector_sha256": args.expect_mmproj_sha256,
            "projector_size_bytes": args.expect_mmproj_size_bytes,
        },
    )
    args.image = gate["artifacts"]["image"]["path"]
    model_layers = int(gate["main_model_profile"]["block_count"])
    plan = build_plan(
        tuple(name.upper() for name in args.configs),
        tuple(args.residency_states),
        args.runs,
    )
    writer = ExperimentArtifactWriter(
        Path(args.output_root).expanduser().resolve(), args.run_id
    )
    source_git, git_complete = _git_source(args, writer.layout.root)
    writer.initialize(
        build_run_config(
            args, gate=gate, model_layers=model_layers, expected_records=len(plan)
        ),
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": {"source_copy": source_git},
            "git_provenance_complete": git_complete,
            "endpoint_gate": gate,
        },
    )
    records: list[dict[str, Any]] = []
    if gate["passed"]:
        records = _run_units(
            args, writer=writer, plan=plan, gate=gate, model_layers=model_layers
        )
    else:
        failed = [name for name, ok in gate["checks"].items() if not ok]
        writer.append_failure(
            FailureRecord(
                stage="endpoint_gate",
                error_type="ArtifactGateFailure",
                message=f"failed_checks={','.join(failed)}",
                monotonic_ns=time.monotonic_ns(),
                fatal=True,
            )
        )
    failure_count = _count_failures(writer.layout.failures)
    writer.write_summary(
        {
            "passed": (
                gate["passed"]
                and git_complete
                and len(records) == len(plan)
                and all(record["passed"] for record in records)
                and failure_count == 0
            ),
            "endpoint_gate": gate,
            "expected_records": len(plan),
            "actual_records": len(records),
            "failure_count": failure_count,
            "git_provenance_complete": git_complete,
            "by_cell": summarize_by_cell(records),
            "records": records,
        }
    )
    write_summary_csv(writer.layout.summary_csv, records)
    return writer.layout.root


def main() -> None:
    args = build_parser().parse_args()
    if args.child_config:
        try:
            result = run_child(args)
        except Exception as exc:
            if args.child_output_json:
                Path(args.child_output_json).write_text(
                    json.dumps(
                        {
                            "config": args.child_config,
                            "residency_state": args.residency_state,
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
    result_root = run_parent(args)
    print(result_root)
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
