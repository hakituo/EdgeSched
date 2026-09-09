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
from experiments.elastic_residency.llm_residency_curve import (  # noqa: E402
    resolve_config,
)
from experiments.elastic_residency.resource_sampler import ResourceSampler  # noqa: E402
from experiments.elastic_residency.s0_pipeline.config import (  # noqa: E402
    S0PipelineConfig,
    default_output_root,
)
from experiments.elastic_residency.s0_pipeline.events import (  # noqa: E402
    EventRecorder,
)
from experiments.elastic_residency.s0_pipeline.playback import (  # noqa: E402
    ContinuousPlayer,
)
from experiments.elastic_residency.s0_pipeline.runner import PipelineRunner  # noqa: E402
from experiments.elastic_residency.s0_pipeline.token_source import (  # noqa: E402
    LlamaTokenSource,
)
from experiments.elastic_residency.s0_pipeline.tts_executor import (  # noqa: E402
    SentenceSynthesizer,
)
from experiments.elastic_residency.tts_component_offload import (  # noqa: E402
    _apply_placement,
    _build_xvector_prompt,
    _component_inventory,
    _nvidia_memory_snapshot,
    _synchronize,
    placement_for,
)
from experiments.elastic_residency.tts_runtime_evidence import (  # noqa: E402
    SwapUsageSampler,
)
from experiments.tts.profiling_utils import sha256_file  # noqa: E402


def _load_llm(  # noqa: PLR0913 - child 编排需要完整参数集
    args: argparse.Namespace,
    config: S0PipelineConfig,
    recorder: EventRecorder,
    n_threads: int,
    model_sha256: str,
) -> tuple[LlamaTokenSource, Any, float]:
    started = time.perf_counter()
    source = LlamaTokenSource(
        model_path=args.llm_model,
        n_ctx=config.n_ctx,
        context_tokens=config.context_tokens,
        generation_budget_tokens=config.generation_budget_tokens,
        max_output_tokens=config.max_output_tokens,
        n_gpu_layers=resolve_config(config.llm_config, args.model_layers),
        n_threads=n_threads,
        seed=config.seed,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        temperature=config.temperature,
        model_sha256=model_sha256,
        bridge_maxsize=config.token_queue_maxsize,
        recorder=recorder,
    )
    identity = source.open()
    return source, identity, time.perf_counter() - started


def _load_tts(  # noqa: PLR0913
    args: argparse.Namespace,
    config: S0PipelineConfig,
) -> tuple[Any, Any, Any, dict[str, Any], float]:
    import torch
    from qwen_tts import Qwen3TTSModel

    started = time.perf_counter()
    wrapper = Qwen3TTSModel.from_pretrained(
        str(Path(args.tts_model).expanduser().resolve()),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    spec = placement_for(config.tts_config)
    _apply_placement(wrapper, spec)
    inventory = _component_inventory(wrapper)
    voice_prompt, prompt_build_s = _build_xvector_prompt(
        wrapper, Path(args.ref_audio).expanduser().resolve()
    )
    _synchronize()
    generation_kwargs = wrapper._merge_generate_kwargs(
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    return (
        wrapper,
        voice_prompt,
        inventory,
        generation_kwargs,
        time.perf_counter() - started,
    )


def run_child(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    import asyncio

    config = S0PipelineConfig(
        placement=args.placement,
        max_output_tokens=args.max_new_tokens,
        llm_cores=args.llm_cores,
        system_cores=args.system_cores,
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
            source, identity, llm_load_s = _load_llm(
                args, config, recorder, len(llm_lease.affinity), model_sha256
            )
            wrapper, voice_prompt, inventory, gen_kwargs, tts_load_s = _load_tts(
                args, config
            )
            synthesizer = SentenceSynthesizer(
                model=wrapper,
                voice_prompt=voice_prompt,
                language=config.language,
                generation_kwargs=gen_kwargs,
                recorder=recorder,
                device="cuda:0",
                backend_kind="qwen3_tts_local",
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
                player.open()
                player.calibrate_clock()
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
                "component_inventory": inventory,
                "gpu_baseline_mb": baseline,
                "resource_summary": resource_summary,
                "swap_report": swap_report,
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
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=24000)
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
