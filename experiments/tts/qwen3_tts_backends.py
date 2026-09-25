"""Qwen3-TTS 的两条推理后端：上游动态路径与 CUDA Graph 静态路径。

两条路径都在函数内部惰性导入重依赖（``torch`` / ``qwen_tts`` /
``faster_qwen3_tts``），因此本模块可以在没有 GPU 或未安装加速包的环境里被
导入和静态验证。

- ``upstream_synth``：与 B2 冻结口径一致的上游调用（x-vector 克隆 +
  非流式完整波形 + ``do_sample=False``）；
- ``cuda_graph_synth``：外部 MIT 项目 ``faster-qwen3-tts`` 的静态 KV cache
  + CUDA Graph 路径。2026-09-25 起优先使用主环境 pip 安装的 0.2.6
  （与 qwen-tts 0.1.1 / transformers 4.57.3 共存，不需要 transformers 5.x）；
  只有环境里没装时才回退到仓库外本地 checkout（挂 ``sys.path``）。
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Callable

#: 路径标识；会写入产物，禁止改名而不升协议。
PATH_UPSTREAM = "upstream_dynamic"
PATH_CUDA_GRAPH = "cuda_graph_static"

#: 外部加速项目本地 checkout 的回退位置（仓库外，不参与本仓库 git）。
#: 环境已 pip 安装 faster-qwen3-tts 时此路径不参与解析。
DEFAULT_FASTER_PATH = "/home/leslie/qwen3tts-cugraph-probe/faster-qwen3-tts"


def resolve_faster_qwen3_tts(faster_path: str) -> tuple[Any, dict[str, Any]]:
    """解析加速包：优先环境已安装版本，回退到本地 checkout。

    返回 (module, provenance)；两者都没有时抛 FileNotFoundError。
    """

    spec = importlib.util.find_spec("faster_qwen3_tts")
    if spec is not None:
        module = importlib.import_module("faster_qwen3_tts")
        return module, {
            "source": "site-packages",
            "version": getattr(module, "__version__", "unknown"),
            "origin": str(Path(spec.origin or "?").parent),
            "checkout_path": None,
        }
    resolved = Path(faster_path).expanduser().resolve()
    if not (resolved / "faster_qwen3_tts").is_dir():
        raise FileNotFoundError(
            "环境未安装 faster-qwen3-tts，且找不到本地 checkout: "
            f"{resolved}（主环境可 uv pip install faster-qwen3-tts==0.2.6）"
        )
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    module = importlib.import_module("faster_qwen3_tts")
    return module, {
        "source": "local_checkout",
        "version": getattr(module, "__version__", "unknown"),
        "origin": str(resolved / "faster_qwen3_tts"),
        "checkout_path": str(resolved),
    }

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
    """CUDA Graph 静态路径：FasterQwen3TTS（优先环境 pip 包，回退本地 checkout）。"""

    import torch

    faster_module, provenance = resolve_faster_qwen3_tts(args.faster_qwen3_tts_path)
    FasterQwen3TTS = faster_module.FasterQwen3TTS
    # 不同版本 from_pretrained 形参不同（如 0.2.6 无 local_files_only），按签名过滤。
    supported = set(inspect.signature(FasterQwen3TTS.from_pretrained).parameters)
    requested = {
        "device": "cuda",
        "dtype": torch.bfloat16,
        "attn_implementation": args.attn_implementation,
        "max_seq_len": args.max_seq_len,
        "local_files_only": True,
    }
    load_kwargs = {k: v for k, v in requested.items() if k in supported}

    started = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(args.model, **load_kwargs)
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

    meta = {
        "load_s": round(load_s, 3),
        "prompt_build_s": None,
        "graph_capture_s": round(capture_s, 3),
        "torch_module_loaded": True,
    }
    meta.update({f"faster_backend_{k}": v for k, v in provenance.items()})
    return synth, meta


def release_gpu() -> None:
    """释放上一路径的显存，保证两条路径不同时驻留（8GB 卡）。"""

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def accelerated_placement_capability(faster_path: str) -> dict[str, Any]:
    """检查加速后端是否支持组件级设备放置（决定 T3/T4 能否在新后端重测）。

    依据是公开接口而不是猜测：``FasterQwen3TTS.from_pretrained`` 的签名里只有整模型
    ``device``，没有 talker/codec/speaker 的分部件设备参数；``gguf_talker_path`` /
    ``gguf_codec_path`` 只用于 GGML 后端挑权重文件，不是设备放置。
    """

    try:
        faster_module, provenance = resolve_faster_qwen3_tts(faster_path)
        FasterQwen3TTS = faster_module.FasterQwen3TTS
        parameters = sorted(
            inspect.signature(FasterQwen3TTS.from_pretrained).parameters
        )
    except Exception as exc:  # noqa: BLE001 - 探测失败如实记录，不吞
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}
    component_keys = ("talker_device", "codec_device", "speaker_device")
    return {
        "checked": True,
        "engine": PATH_CUDA_GRAPH,
        **{f"faster_backend_{k}": v for k, v in provenance.items()},
        "component_placement_supported": any(
            key in parameters for key in component_keys
        ),
        "whole_model_device_parameter": "device" in parameters,
        "from_pretrained_parameters": parameters,
        "evidence": (
            "from_pretrained 只有整模型 device 参数，没有 talker/codec/speaker "
            "分部件设备参数；gguf_*_path 只用于 GGML 后端的权重文件选择"
        ),
        "cpu_whole_model": "unverified_requires_run",
        "cpu_whole_model_note": (
            "device 参数接受 cpu，但该实现核心是 torch.cuda.CUDAGraph 捕获，"
            "CPU 上图捕获不可用，是否还能运行必须单独实跑确认，"
            "不得默认沿用旧 T4 数值"
        ),
    }


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
