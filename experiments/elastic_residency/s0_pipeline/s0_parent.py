"""S0 流水线 parent：配置成对编排、门禁聚合与产物写入。

parent 不加载模型，只负责按冻结协议调用 child、收集结果并执行门禁；
child 失败默认 fail-fast，收全证据时可用 ``--continue-on-failure``。

成对口径：每个 ``run_index`` 内对所有 placement × 启动策略做**轮转排序**，
使每个配置在各重复轮次里都出现在不同的执行位次，降低位置/热状态偏差。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.elastic_residency.contracts import (
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentRunConfig,
    FailureRecord,
)
from experiments.elastic_residency.s0_pipeline.config import (  # noqa: E402
    DEFAULT_STARTUP_POLICY,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    FREEZE_STATUS,
    FROZEN_PROTOCOL,
    PROMPT_CONTRACT_VERSION,
    S0_PLACEMENTS,
    S0_PROTOCOL_VERSION,
    S0_STARTUP_POLICIES,
    TTS_PLACEMENT_T0,
    default_output_root,
    text_fingerprint,
)
from experiments.elastic_residency.s0_pipeline.measurement_scope import (  # noqa: E402
    SCOPE_VERSION,
)
from experiments.elastic_residency.s0_pipeline.s0_evidence import (  # noqa: E402
    attach_evidence,
    summarize_record,
)
from experiments.tts.qwen3_tts_backends import (  # noqa: E402
    DEFAULT_FASTER_PATH,
    PATH_CUDA_GRAPH,
    PATH_UPSTREAM,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHILD_MODULE = "experiments.elastic_residency.s0_pipeline.s0_child"


def validate_placements(placements: Sequence[str], tts_backend: str) -> None:
    """放置维度必须与后端能力一致。

    加速后端只暴露整模型 ``device``，没有分部件放置接口，因此 T3/T4 必须在
    这里被挡住，而不是跑出一份看着像旧后端的数据。
    """

    if tts_backend != PATH_CUDA_GRAPH:
        return
    unsupported = [
        item for item in placements if not item.endswith(f":{TTS_PLACEMENT_T0}")
    ]
    if unsupported:
        raise ValueError(
            f"{PATH_CUDA_GRAPH} 只支持 {TTS_PLACEMENT_T0} 放置，不支持: {unsupported}"
        )


def _rotated(items: Sequence[str], offset: int) -> list[str]:
    """按 offset 轮转，保证同一 run_index 内各配置执行位次不同。"""

    if not items:
        return []
    offset %= len(items)
    return list(items[offset:]) + list(items[:offset])


def _run_child(  # noqa: PLR0913
    *,
    placement: str,
    startup_policy: str,
    args: argparse.Namespace,
    output_json: Path,
    run_index: int,
) -> tuple[dict[str, Any], str]:
    command = [
        sys.executable,
        "-m",
        CHILD_MODULE,
        "--placement",
        placement,
        "--startup-policy",
        startup_policy,
        "--llm-model",
        args.llm_model,
        "--tts-model",
        args.tts_model,
        "--ref-audio",
        args.ref_audio,
        "--prompt",
        args.prompt,
        "--system-prompt",
        args.system_prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--model-layers",
        str(args.model_layers),
        "--request-id",
        f"{placement.replace(':', '-')}-{startup_policy}-r{run_index}",
        "--child-output-json",
        str(output_json),
        "--llm-cores",
        str(args.llm_cores),
        "--system-cores",
        str(args.system_cores),
        "--tts-backend",
        args.tts_backend,
        "--faster-qwen3-tts-path",
        args.faster_qwen3_tts_path,
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
                backend_kind=f"qwen3_tts_{args.tts_backend}",
                is_real=True,
                model_path=args.tts_model,
            ),
        ),
        parameters={
            "protocol_version": S0_PROTOCOL_VERSION,
            "measurement_scope_version": SCOPE_VERSION,
            "tts_backend": args.tts_backend,
            "placements": list(args.placements),
            "startup_policies": list(args.startup_policies),
            "runs": args.runs,
            "prompt": args.prompt,
            "prompt_sha256": text_fingerprint(args.prompt),
            "system_prompt": args.system_prompt,
            "system_prompt_sha256": text_fingerprint(args.system_prompt),
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "max_new_tokens": args.max_new_tokens,
            "enable_playback": args.enable_playback,
        },
    )
    writer.initialize(
        config,
        {
            "freeze_status": FREEZE_STATUS,
            "protocol": FROZEN_PROTOCOL,
            "protocol_version": S0_PROTOCOL_VERSION,
            "measurement_scope_version": SCOPE_VERSION,
            "purpose": "formal_s0_paired_comparison",
            "placements_frozen": list(S0_PLACEMENTS),
        },
    )
    summary: dict[str, Any] = {
        "protocol_version": S0_PROTOCOL_VERSION,
        "placements": list(args.placements),
        "startup_policies": list(args.startup_policies),
        "runs": args.runs,
        "records": [],
        "all_passed": True,
    }
    for run_index in range(1, args.runs + 1):
        order_index = 0
        for placement in _rotated(list(args.placements), run_index - 1):
            for policy in _rotated(list(args.startup_policies), run_index - 1):
                started = time.monotonic()
                tag = placement.replace(":", "-")
                output_json = (
                    writer.layout.root
                    / "children"
                    / f"{tag}_{policy}_r{run_index}.json"
                )
                output_json.parent.mkdir(parents=True, exist_ok=True)
                record, stderr = _run_child(
                    placement=placement,
                    startup_policy=policy,
                    args=args,
                    output_json=output_json,
                    run_index=run_index,
                )
                record["placement"] = placement
                record["startup_policy"] = policy
                record["run_index"] = run_index
                record["order_index"] = order_index
                order_index += 1
                record["parent_wall_s"] = time.monotonic() - started
                attach_evidence(record)
                for event in record.get("run", {}).get("events", []):
                    writer.append_event(event)
                writer.append_iteration(record)
                summary["records"].append(summarize_record(record))
                if not record["passed"]:
                    writer.append_failure(
                        FailureRecord(
                            stage="s0_pipeline_child",
                            error_type="GateFailure",
                            message=record.get("error") or stderr or "门禁未通过",
                            monotonic_ns=time.monotonic_ns(),
                            fatal=not args.continue_on_failure,
                            attributes={
                                "placement": placement,
                                "startup_policy": policy,
                                "run_index": run_index,
                            },
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
    parser.add_argument(
        "--startup-policies", nargs="+", default=[DEFAULT_STARTUP_POLICY]
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--llm-model",
        default="/home/leslie/mvp_core/models/llm/Qwen3.5-9B-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--tts-model",
        default="/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base",
    )
    parser.add_argument(
        "--ref-audio", default="/home/leslie/mvp_core/models/tts_refs/ref_calm.wav"
    )
    parser.add_argument("--prompt", default="请用三到五句话介绍边缘计算的优势。")
    parser.add_argument("--system-prompt", default=DIRECT_ANSWER_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument(
        "--tts-backend", choices=(PATH_CUDA_GRAPH, PATH_UPSTREAM), default=PATH_CUDA_GRAPH
    )
    parser.add_argument("--faster-qwen3-tts-path", default=DEFAULT_FASTER_PATH)
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
    args.placements = [item.upper() for item in args.placements]
    invalid = [item for item in args.placements if item not in S0_PLACEMENTS]
    if invalid:
        raise SystemExit(f"未冻结的放置配置: {invalid}")
    invalid_startup = [
        item for item in args.startup_policies if item not in S0_STARTUP_POLICIES
    ]
    if invalid_startup:
        raise SystemExit(f"未冻结的启动策略: {invalid_startup}")
    try:
        validate_placements(args.placements, args.tts_backend)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    root = run_parent(args)
    print(json.dumps({"artifacts": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
