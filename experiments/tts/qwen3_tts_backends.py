"""Qwen3-TTS 的两条推理后端：上游动态路径与 CUDA Graph 静态路径。

两条路径都在函数内部惰性导入重依赖（``torch`` / ``qwen_tts`` /
``faster_qwen3_tts``），因此本模块可以在没有 GPU 或未安装加速包的环境里被
导入和静态验证。

- ``upstream_synth``：与 B2 冻结口径一致的上游调用（x-vector 克隆 +
  非流式完整波形 + ``do_sample=False``）；
- ``cuda_graph_synth``：外部 MIT 项目 ``faster-qwen3-tts`` 的静态 KV cache
  + CUDA Graph 路径，只把它挂到 ``sys.path``，不安装进本仓库环境。
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any, Callable

#: 路径标识；会写入产物，禁止改名而不升协议。
PATH_UPSTREAM = "upstream_dynamic"
PATH_CUDA_GRAPH = "cuda_graph_static"

#: 外部加速项目的默认位置（仓库外，不参与本仓库 git）。
DEFAULT_FASTER_PATH = "/home/leslie/qwen3tts-cugraph-probe/faster-qwen3-tts"

SynthFn = Callable[[str], tuple[Any, int]]


def upstream_synth(args: Any) -> tuple[SynthFn, dict[str, Any]]:
    """上游动态路径：与 B2 完全一致的调用方式（x-vector + 非流式）。"""

    import torch
    from qwen_tts import Qwen3TTSModel

    from experiments.elastic_residency.tts_component_offload import (
        _build_xvector_prompt,
    )

    started = time.perf_counter()
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    prompt_item, prompt_s = _build_xvector_prompt(wrapper, Path(args.ref_audio))
    voice_prompt = wrapper._prompt_items_to_voice_clone_prompt([prompt_item])
    gen_kwargs = wrapper._merge_generate_kwargs(
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    load_s = time.perf_counter() - started

    def synth(text: str) -> tuple[Any, int]:
        return wrapper.generate_voice_clone(
            text=text,
            language=args.language,
            voice_clone_prompt=voice_prompt,
            non_streaming_mode=True,
            **gen_kwargs,
        )

    return synth, {
        "load_s": round(load_s, 3),
        "prompt_build_s": round(prompt_s, 3),
        "graph_capture_s": None,
    }


def cuda_graph_synth(args: Any) -> tuple[SynthFn, dict[str, Any]]:
    """CUDA Graph 静态路径：外部项目 FasterQwen3TTS，仅挂 ``sys.path``。"""

    faster_path = Path(args.faster_qwen3_tts_path).expanduser().resolve()
    if not (faster_path / "faster_qwen3_tts").is_dir():
        raise FileNotFoundError(f"找不到 faster-qwen3-tts: {faster_path}")
    if str(faster_path) not in sys.path:
        sys.path.insert(0, str(faster_path))
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    started = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        max_seq_len=args.max_seq_len,
        local_files_only=True,
    )
    load_s = time.perf_counter() - started
    capture_started = time.perf_counter()
    model.generate_voice_clone(
        text=args.warmup_text,
        language=args.language,
        ref_audio=args.ref_audio,
        xvec_only=args.xvec_only,
        non_streaming_mode=True,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    torch.cuda.synchronize()
    capture_s = time.perf_counter() - capture_started

    def synth(text: str) -> tuple[Any, int]:
        return model.generate_voice_clone(
            text=text,
            language=args.language,
            ref_audio=args.ref_audio,
            xvec_only=args.xvec_only,
            non_streaming_mode=True,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )

    return synth, {
        "load_s": round(load_s, 3),
        "prompt_build_s": None,
        "graph_capture_s": round(capture_s, 3),
        "torch_module_loaded": True,
    }


def release_gpu() -> None:
    """释放上一路径的显存，保证两条路径不同时驻留（8GB 卡）。"""

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def package_version(module_name: str) -> str:
    module = sys.modules.get(module_name)
    return str(getattr(module, "__version__", "unknown"))


def build_synth(args: Any, path: str) -> tuple[SynthFn, dict[str, Any]]:
    builders = {PATH_UPSTREAM: upstream_synth, PATH_CUDA_GRAPH: cuda_graph_synth}
    try:
        builder = builders[path]
    except KeyError as exc:
        raise ValueError(f"未知推理后端路径: {path}") from exc
    return builder(args)
