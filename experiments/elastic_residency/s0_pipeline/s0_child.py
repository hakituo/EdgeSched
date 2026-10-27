"""S0 流水线 child：真实模型加载、冻结放置、资源门禁与产物写入。

child 只服务一次运行；四配置编排由 ``s0_parent`` 负责。
本文件不含任何 Mock 或 fallback：缺少真实后端时直接失败。
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.cpu_topology import (  # noqa: E402
    CPUTopology,
    PhysicalCoreAllocator,
    process_affinity_scope,
)
from experiments.elastic_residency.gpu_telemetry import DirectNvmlProbe  # noqa: E402
from experiments.elastic_residency.resource_sampler import ResourceSampler  # noqa: E402
from experiments.elastic_residency.s0_pipeline.config import (  # noqa: E402
    DEFAULT_STARTUP_POLICY,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    S0PipelineConfig,
    default_output_root,
    text_fingerprint,
)
from experiments.elastic_residency.s0_pipeline.events import (  # noqa: E402
    EventRecorder,
)
from experiments.elastic_residency.s0_pipeline.playback import (  # noqa: E402
    ContinuousPlayer,
)
from experiments.elastic_residency.s0_pipeline.runner import PipelineRunner  # noqa: E402
from experiments.elastic_residency.s0_pipeline.s0_backends import (  # noqa: E402
    load_llm,
    load_tts,
)
from experiments.elastic_residency.s0_pipeline.token_source import (  # noqa: E402
    LlamaTokenSource,
)
from experiments.elastic_residency.s0_pipeline.tts_executor import (  # noqa: E402
    AcceleratedSentenceSynthesizer,
    SentenceSynthesizer,
)
from experiments.elastic_residency.tts_component_offload import (  # noqa: E402
    _nvidia_memory_snapshot,
)
from experiments.elastic_residency.tts_runtime_evidence import (  # noqa: E402
    SwapUsageSampler,
)
from experiments.tts.profiling_utils import sha256_file  # noqa: E402
from experiments.tts.qwen3_tts_backends import (  # noqa: E402
    DEFAULT_FASTER_PATH,
    PATH_CUDA_GRAPH,
    PATH_UPSTREAM,
)


def run_child(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    import asyncio

    config = S0PipelineConfig(
        placement=args.placement,
        max_output_tokens=args.max_new_tokens,
        startup_policy=args.startup_policy,
        llm_cores=args.llm_cores,
        system_cores=args.system_cores,
        system_prompt=args.system_prompt,
    )
    topology = CPUTopology.discover()
    if len(topology.physical_cores) < config.system_cores + config.llm_cores:
        raise RuntimeError("物理核不足，无法按配置租约执行")
    allocator = PhysicalCoreAllocator(topology)
    system_lease = allocator.reserve_system(config.system_cores)
    llm_lease = allocator.acquire("s0_pipeline", config.llm_cores, use_smt=True)
    process = psutil.Process()
    process.cpu_percent(interval=None)

    recorder = EventRecorder(request_id=args.request_id, config=config.placement)
    sampler: ResourceSampler | None = None
    source: LlamaTokenSource | None = None
    player: ContinuousPlayer | None = None
    result: dict[str, Any] = {}
    swap_sampler = SwapUsageSampler(
        interval_s=config.sample_interval_s,
        tolerance_bytes=int(config.max_swap_growth_mb * 1024 * 1024),
    )
    try:
        with process_affinity_scope(llm_lease.affinity) as affinity:
            swap_sampler.start()
            sampler = ResourceSampler(
                interval_s=config.sample_interval_s,
                gpu_interval_s=config.gpu_sample_interval_s,
                gpu_probe=DirectNvmlProbe(include_process_memory=False),
                thread_affinity=system_lease.affinity,
            )
            sampler.start(wait_for_gpu_timeout_s=config.gpu_warmup_timeout_s)
            baseline = _nvidia_memory_snapshot(process.pid)
            model_sha256 = sha256_file(Path(args.llm_model).expanduser().resolve())
            source, identity, llm_load_s = load_llm(
                args, config, recorder, len(llm_lease.affinity), model_sha256
            )
            wrapper, voice_prompt, inventory, gen_kwargs, tts_load_s = load_tts(
                args, config
            )
            if args.tts_backend == PATH_CUDA_GRAPH:
                synthesizer = AcceleratedSentenceSynthesizer(
                    model=wrapper,
                    voice_prompt=voice_prompt,
                    language=config.language,
                    generation_kwargs=gen_kwargs,
                    recorder=recorder,
                    device="cuda:0",
                    backend_kind=args.tts_backend,
                )
            else:
                synthesizer = SentenceSynthesizer(
                    model=wrapper,
                    voice_prompt=voice_prompt,
                    language=config.language,
                    generation_kwargs=gen_kwargs,
                    recorder=recorder,
                    device="cuda:0",
                    backend_kind=args.tts_backend,
                )
            player = None
            if args.enable_playback:
                player = ContinuousPlayer(
                    sample_rate=args.sample_rate,
                    recorder=recorder,
                    blocksize=config.playback_blocksize,
                    device=config.playback_device,
                    max_pending_sentences=config.audio_queue_maxsize,
                )
                # 打开输出流与校准时钟由 runner 按启动策略决定时机：
                # first_sentence 在消费前，full_prebuffer 在整段合成完成后
            source.start()
            runner = PipelineRunner(
                config=config,
                token_source=source,
                synthesizer=synthesizer,
                recorder=recorder,
                player=player,
                request_id=args.request_id,
            )
            run_result = asyncio.run(runner.run())
            resource_summary = sampler.stop()
            swap_report = swap_sampler.stop()
            result = {
                "passed": run_result.finished and not run_result.error,
                "placement": config.placement,
                "request_id": args.request_id,
                "affinity": list(affinity),
                "llm_identity": identity.to_dict(),
                "llm_load_s": llm_load_s,
                "tts_load_s": tts_load_s,
                "tts_backend": args.tts_backend,
                "component_inventory": inventory,
                "gpu_baseline_mb": baseline,
                "resource_summary": resource_summary,
                "swap_report": swap_report,
                "output_contract": run_result.text_contract,
                "prompt_contract": {
                    "version": config.prompt_contract_version,
                    "prompt": args.prompt,
                    "prompt_sha256": text_fingerprint(args.prompt),
                    "system_prompt": config.system_prompt,
                    "system_prompt_sha256": text_fingerprint(config.system_prompt),
                    **source.prompt_evidence,
                },
                "run": run_result.to_dict(),
            }
    except Exception as exc:  # noqa: BLE001 - child 失败必须落盘
        result = {
            "passed": False,
            "placement": args.placement,
            "request_id": args.request_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if source is not None:
            source.close()
        if player is not None:
            player.close()
        gc.collect()
        llm_lease.release()
        system_lease.release()

    if args.child_output_json:
        output = Path(args.child_output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\
",
            encoding="utf-8",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", required=True)
    parser.add_argument(
        "--llm-model",
        default="/home/leslie/mvp_core/models/llm/Qwen3.5-9B-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--tts-model",
        default="/home/leslie/Autarkia/models/Qwen3-TTS-12Hz-0.6B-Base",
    )
    parser.add_argument(
        "--ref-audio", default="/home/leslie/mvp_core/models/tts_refs/ref_calm.wav"
    )
    parser.add_argument("--prompt", default="请用三到五句话介绍边缘计算的优势。")
    parser.add_argument("--system-prompt", default=DIRECT_ANSWER_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--startup-policy", default=DEFAULT_STARTUP_POLICY)
    parser.add_argument(
        "--tts-backend",
        choices=(PATH_CUDA_GRAPH, PATH_UPSTREAM),
        default=PATH_CUDA_GRAPH,
        help="cuda_graph_static 为当前接入后端（B2 口径，T0）；upstream_dynamic 为旧后端",
    )
    parser.add_argument("--faster-qwen3-tts-path", default=DEFAULT_FASTER_PATH)
    parser.add_argument("--enable-playback", action="store_true")
    parser.add_argument("--request-id", default="s0-pilot")
    parser.add_argument("--child-output-json")
    parser.add_argument(
        "--output-root", default=str(default_output_root(PROJECT_ROOT))
    )
    parser.add_argument(
        "--run-id",
        default=f"s0_pipeline_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_child(args)
    print(json.dumps({"passed": result.get("passed", False)}, ensure_ascii=False))
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
