"""S0 child 的真实后端加载：LLM 与 TTS。

从 ``s0_child`` 拆出是为了保持单文件规模上限，并让「加载真实后端」
与「编排一次运行」两件事互不干扰。本文件不含任何 Mock 或 fallback，
缺少真实后端时直接抛错。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from experiments.elastic_residency.llm_residency_curve import (  # noqa: E402
    resolve_config,
)
from experiments.elastic_residency.s0_pipeline.config import (  # noqa: E402
    S0PipelineConfig,
    TTS_PLACEMENT_T0,
)
from experiments.elastic_residency.s0_pipeline.events import (  # noqa: E402
    EventRecorder,
)
from experiments.elastic_residency.s0_pipeline.token_source import (  # noqa: E402
    LlamaTokenSource,
)
from experiments.tts.qwen3_tts_backends import PATH_CUDA_GRAPH  # noqa: E402

from experiments.elastic_residency.tts_component_offload import (  # noqa: E402
    _apply_placement,
    _build_xvector_prompt,
    _component_inventory,
    _synchronize,
    placement_for,
)


def load_llm(
    args: argparse.Namespace,
    config: S0PipelineConfig,
    recorder: EventRecorder,
    n_threads: int,
    model_sha256: str,
) -> tuple[LlamaTokenSource, Any, float]:
    """加载真实 llama_cpp 后端并返回 (token 源, 身份, 加载耗时)。"""

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
        system_prompt=config.system_prompt,
        temperature=config.temperature,
        model_sha256=model_sha256,
        bridge_maxsize=config.token_queue_maxsize,
        recorder=recorder,
    )
    identity = source.open()
    return source, identity, time.perf_counter() - started


def load_tts(
    args: argparse.Namespace,
    config: S0PipelineConfig,
) -> tuple[Any, Any, Any, dict[str, Any], float]:
    """加载真实 TTS，返回 (模型, 音色, 清单, 参数, 耗时)。

    推理后端由 ``--tts-backend`` 决定：

    - ``upstream_dynamic``：上游动态路径 + 冻结的分部件放置（旧后端，T0–T4）；
    - ``cuda_graph_static``：B2 画像用的加速 CUDA Graph 路径，整模型驻 GPU，
      放置维度只有 ``T0``，因此 T3/T4 在这里直接失败而不是静默退回。
    """

    if args.tts_backend == PATH_CUDA_GRAPH:
        return _load_tts_cuda_graph(args, config)
    return _load_tts_upstream(args, config)


def _load_tts_cuda_graph(
    args: argparse.Namespace,
    config: S0PipelineConfig,
) -> tuple[Any, Any, Any, dict[str, Any], float]:
    """加速路径：复用 B2 的后端构造，口径与 B2 画像一致。"""

    from types import SimpleNamespace

    from experiments.tts.b2_profile_contract import WARMUP_TEXT
    from experiments.tts.qwen3_tts_backends import cuda_graph_synth

    if config.tts_config != TTS_PLACEMENT_T0:
        raise ValueError(
            f"加速后端只支持整模型 GPU（{TTS_PLACEMENT_T0}），"
            f"收到 {config.tts_config}：当前后端没有分部件放置接口"
        )
    backend_args = SimpleNamespace(
        model=args.tts_model,
        ref_audio=args.ref_audio,
        language=config.language,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        xvec_only=True,
        attn_implementation="eager",
        max_seq_len=2048,
        faster_qwen3_tts_path=args.faster_qwen3_tts_path,
        warmup_text=WARMUP_TEXT,
    )
    started = time.perf_counter()
    synth, meta = cuda_graph_synth(backend_args)
    elapsed = time.perf_counter() - started
    inventory = {
        "backend": PATH_CUDA_GRAPH,
        "placement": TTS_PLACEMENT_T0,
        "load_s": meta.get("load_s"),
        "graph_capture_s": meta.get("graph_capture_s"),
        "component_split": None,
        "note": "加速后端整模型驻 GPU，无 talker/codec/speaker 分部件设备清单",
    }
    return synth, None, inventory, {}, elapsed


def _load_tts_upstream(
    args: argparse.Namespace,
    config: S0PipelineConfig,
) -> tuple[Any, Any, Any, dict[str, Any], float]:
    """上游动态路径 + 冻结放置（旧后端）。"""

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
    prompt_item, _ = _build_xvector_prompt(
        wrapper, Path(args.ref_audio).expanduser().resolve()
    )
    # generate_voice_clone 只接受 dict 或 list[VoiceClonePromptItem]，
    # 传入单个 VoiceClonePromptItem 会在内部下标访问时报错
    voice_prompt = wrapper._prompt_items_to_voice_clone_prompt([prompt_item])
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
