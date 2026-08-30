#!/usr/bin/env python3
"""SM-A：GPU 计算份额能力探针（P1-C 第一阶段，只做能力 preflight）。

不实现在线联合规划器，也不修改任何生产调度路径。探针顺序：

1. 普通 CUDA 不受控共享基线；
2. MPS daemon 可用性检查；
3. 每个 percentage 使用独立客户端进程（每个静态配置一个全新 daemon）；
4. 核对请求比例、客户端实际 SM 数、context 创建顺序与短 workload；
5. 清理 daemon 与私有目录；
6. 输出 manifest、逐配置 JSONL、summary、failures 和事件记录。

证据口径见 ``sm_probe_records`` 与 ``sm_probe_verdict``：``effective_sm_count``
只接受客户端 CUDA context 实测；``MPS_CAP`` 不是独占 SM 分区；后端不可用
或份额无法验证时输出 ``UNSUPPORTED``，绝不静默退回普通 CUDA。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
)
from experiments.elastic_residency.sm.green_context_probe import (  # noqa: E402
    probe_green_context,
)
from experiments.elastic_residency.sm.mps_runtime import MpsRuntime  # noqa: E402
from experiments.elastic_residency.sm.sm_client_runner import (  # noqa: E402
    run_baseline,
    run_mps_configurations,
)
from experiments.elastic_residency.sm.sm_partition_rules import (  # noqa: E402
    GpuPartitionProbeResult,
)
from experiments.elastic_residency.sm.sm_probe_env import (  # noqa: E402
    build_manifest,
    build_run_config,
    gpu_environment,
    parse_percentages,
    source_git_evidence,
    utc_now_iso,
)
from experiments.elastic_residency.sm.sm_probe_records import (  # noqa: E402
    BASELINE_PHASE_ID,
)
from experiments.elastic_residency.sm.sm_probe_verdict import (  # noqa: E402
    build_summary,
)

EXPERIMENT_NAME = "elastic_residency_sm_capability"
DEFAULT_PERCENTAGES = (100, 75, 50, 25)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    percentages = parse_percentages(args.percentages)
    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    source_git, git_provenance_complete = source_git_evidence(
        args, writer.layout.root
    )
    runtime = MpsRuntime(
        binary=args.mps_binary,
        binary_path=args.mps_binary_path or None,
        temp_root=args.temp_root or (writer.layout.root / "mps_runtime"),
        start_timeout_s=args.mps_start_timeout,
        command_timeout_s=args.mps_command_timeout,
    )
    binary_info = runtime.detect()
    writer.append_event(
        ExperimentEvent(
            name="sm_a_start",
            monotonic_ns=time.monotonic_ns(),
            wall_time_iso=utc_now_iso(),
            phase="sm_a",
            attributes={
                "percentages": list(percentages),
                "mps_binary": binary_info.to_dict(),
            },
        )
    )

    baseline_result, baseline_client = run_baseline(args, writer)
    baseline_payload = baseline_client.get("payload") or {}
    gpu_env = gpu_environment()
    green_context = probe_green_context(
        attempt_device_resource=args.green_context_device_resource
    )
    writer.initialize(
        build_run_config(
            run_id=args.run_id,
            args=args,
            percentages=percentages,
            binary_available=binary_info.available,
        ),
        build_manifest(
            args=args,
            gpu_environment=gpu_env,
            baseline_gpu=baseline_payload.get("gpu") or {},
            binary_info=binary_info.to_dict(),
            green_context=green_context.to_dict(),
            source_git=source_git,
        ),
    )
    writer.append_iteration(baseline_result.to_dict())
    baseline_failed = baseline_result.failure is not None
    if baseline_failed:
        writer.append_failure(
            FailureRecord(
                stage=BASELINE_PHASE_ID,
                error_type="BaselineFailed",
                message=str(baseline_result.failure),
                monotonic_ns=time.monotonic_ns(),
                attributes={"returncode": baseline_client.get("returncode")},
            )
        )
        mps_results: list[GpuPartitionProbeResult] = []
        fallback_count = 0
        daemon_start_failures: list[str] = []
        cleanup_failures: list[str] = []
    else:
        (
            mps_results,
            fallback_count,
            daemon_start_failures,
            cleanup_failures,
        ) = run_mps_configurations(
            args,
            writer,
            runtime,
            percentages,
            # 物理 SM 总数只来自不受控 CUDA 基线。
            baseline_result.profile.total_physical_sms,
        )

    results = [baseline_result, *mps_results]
    summary = build_summary(
        run_id=args.run_id,
        experiment=EXPERIMENT_NAME,
        percentages=percentages,
        results=results,
        baseline_payload=baseline_payload,
        baseline_failed=baseline_failed,
        baseline_failure=baseline_result.failure,
        gpu_environment=gpu_env,
        binary=binary_info.to_dict(),
        binary_available=binary_info.available,
        binary_reason=binary_info.reason,
        daemon_start_failures=daemon_start_failures,
        cleanup_failures=cleanup_failures,
        fallback_count=fallback_count,
        green_context=green_context.to_dict(),
        git_provenance_complete=git_provenance_complete,
    )
    writer.write_summary(summary)
    writer.write_summary_csv(
        {
            "run_id": args.run_id,
            "verdict": summary["verdict"],
            "passed": summary["passed"],
            "fallback_count": fallback_count,
            "total_physical_sms": summary["total_physical_sms"],
            "mps_binary_available": binary_info.available,
            "cleanup_failures": cleanup_failures,
            "source_git_commit": args.source_git_commit,
            "source_git_dirty": source_git["dirty"],
        }
    )
    writer.append_event(
        ExperimentEvent(
            name="sm_a_finish",
            monotonic_ns=time.monotonic_ns(),
            wall_time_iso=utc_now_iso(),
            phase="sm_a",
            attributes={
                "verdict": summary["verdict"],
                "go_conditions": summary["go_conditions"],
            },
        )
    )
    return {"summary": summary, "artifact_root": str(writer.layout.root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT / "experiment_results" / "elastic_residency" / "sm_capability"
        ),
    )
    parser.add_argument(
        "--run-id", default=f"sm_a_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    parser.add_argument(
        "--percentages", type=int, nargs="+", default=list(DEFAULT_PERCENTAGES)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--client-timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--mps-binary", default="nvidia-cuda-mps-control")
    parser.add_argument("--mps-binary-path", default="")
    parser.add_argument("--mps-start-timeout", type=float, default=15.0)
    parser.add_argument("--mps-command-timeout", type=float, default=10.0)
    parser.add_argument("--temp-root", default="")
    parser.add_argument(
        "--mps-start-server",
        action="store_true",
        help=(
            "显式经 stdin 发送 start_server -uid <当前 uid>。默认让首个 CUDA "
            "client 按需拉起 MPS server；只有需要把 server 创建与 client 连接"
            "解耦排查时才开启（仍保持同一用户语义）"
        ),
    )
    parser.add_argument(
        "--sm-estimate-rounding",
        choices=("ceil", "floor", "round"),
        default="ceil",
        help="只用于生成请求/预期 SM 数，不作为实测结果",
    )
    parser.add_argument(
        "--green-context-device-resource",
        action="store_true",
        help="额外执行一次设备 SM 资源只读查询（仍不创建 Green Context）",
    )
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")
    parser.add_argument(
        "--fail-on-no-go",
        action="store_true",
        help="No-Go 时也返回非零退出码（默认 No-Go 不算代码失败）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outcome = run_probe(args)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))
    summary = outcome["summary"]
    if not summary.get("report_complete"):
        return 1
    if args.fail_on_no_go and summary.get("verdict") != "GO":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
