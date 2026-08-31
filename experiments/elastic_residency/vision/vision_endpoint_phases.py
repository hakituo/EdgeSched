"""文本端点与视觉端点的真实执行阶段（不含进程编排与验收装配）。

文本阶段复用 C1/C2 的精确 token prompt 与 prefill/decode 分离口径；视觉阶段用
libmtmd 对固定真实图片执行 projector，再把视觉嵌入送进同一个 Qwen3.5-4B 上下文，
并单独记录 warm/释放的显存与 RSS 差值。任何视觉步骤失败都向上抛错，不做纯文本回退。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.elastic_residency.kv_measurement import capture_llama_kv_state
from experiments.elastic_residency.llm_residency_curve import (
    _build_exact_prompt_tokens,
    _decode_fixed_tokens,
    _token_ids_sha256,
)
from experiments.elastic_residency.vision.mtmd_eval import (
    evaluate_chunks,
    generate_fixed_tokens,
)
from experiments.elastic_residency.vision.mtmd_runtime import (
    MtmdSessionConfig,
    MtmdVisionSession,
)
from experiments.elastic_residency.vision.vision_evidence import (
    RESIDENCY_TEXT_ONLY,
    RESIDENCY_TEXT_ONLY_CHAT,
    build_prompt_evidence,
    memory_delta,
)


SnapshotFn = Callable[[], tuple[dict[str, Any], int]]


@dataclass(frozen=True)
class VisionPhaseConfig:
    """视觉端点一轮执行的冻结参数。"""

    mmproj_path: Path
    image_path: Path
    question: str
    prompt_mode: str
    eval_mode: str
    generation_tokens: int
    use_gpu: bool
    n_threads: int
    image_max_tokens: int = 0
    image_min_tokens: int = 0


def _output_text_evidence(llm: Any, tokens: list[int]) -> dict[str, Any]:
    text = llm.detokenize(tokens).decode("utf-8", errors="replace")
    return {
        "output_text": text,
        "output_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_text_preview": text[:400],
        "generated_tokens_sha256": _token_ids_sha256(tokens),
        "eos_token_count": sum(token == llm.token_eos() for token in tokens),
    }


def _try_kv_state_roundtrip(llm: Any) -> dict[str, Any]:
    """尝试完整状态保存/恢复；失败记录原因，不吞掉证据也不中断整轮。"""

    expected_tokens = int(llm.n_tokens)
    try:
        save_started = time.perf_counter()
        state = llm.save_state()
        save_s = time.perf_counter() - save_started
        state_bytes = int(getattr(state, "llama_state_size", 0))
        llm.reset()
        restore_started = time.perf_counter()
        llm.load_state(state)
        restore_s = time.perf_counter() - restore_started
        return {
            "attempted": True,
            "succeeded": True,
            "error": None,
            "kv_save_s": save_s,
            "kv_restore_s": restore_s,
            "kv_state_bytes": state_bytes,
            "restored_tokens": int(llm.n_tokens),
            "expected_state_tokens": expected_tokens,
        }
    except Exception as exc:  # noqa: BLE001 - 保留原因供人工判定
        return {
            "attempted": True,
            "succeeded": False,
            "error": f"{type(exc).__name__}: {exc}",
            "kv_save_s": None,
            "kv_restore_s": None,
            "kv_state_bytes": None,
            "restored_tokens": None,
            "expected_state_tokens": expected_tokens,
        }


def run_text_phase(
    *,
    llm: Any,
    residency_state: str,
    context_tokens: int,
    generation_tokens: int,
    prompt_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """纯文本端点：固定 token 填充 prompt 或同 prompt 对话对照。"""

    if residency_state == RESIDENCY_TEXT_ONLY:
        prompt_tokens = _build_exact_prompt_tokens(llm, context_tokens)
        prompt_meta: dict[str, Any] = {
            "prompt_construction_mode": "fixed_token_filler",
            "prompt_text": None,
            "prompt_text_sha256": None,
            "prompt_contains_media_marker": False,
            "question": None,
            "media_marker": None,
        }
    elif residency_state == RESIDENCY_TEXT_ONLY_CHAT:
        if not prompt_evidence:
            raise ValueError("TEXT_ONLY_CHAT 需要对话 prompt 证据")
        prompt_meta = dict(prompt_evidence)
        prompt_tokens = list(
            llm.tokenize(
                prompt_meta["prompt_text"].encode("utf-8"),
                add_bos=False,
                special=True,
            )
        )
    else:
        raise ValueError(f"run_text_phase 不接受状态 {residency_state}")
    if len(prompt_tokens) + generation_tokens > int(llm.n_ctx()):
        raise ValueError("文本 prompt 与生成长度超过 n_ctx")
    inference = _decode_fixed_tokens(
        llm=llm,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
    )
    generated = inference.pop("generated_tokens")
    phase_intervals = inference.pop("phase_intervals_ns")
    kv_native_state = capture_llama_kv_state(llm)
    return {
        "prompt": {
            **prompt_meta,
            "prompt_token_count": len(prompt_tokens),
            "prompt_tokens_sha256": _token_ids_sha256(prompt_tokens),
        },
        "inference": inference,
        "output": _output_text_evidence(llm, generated),
        "kv_native_state": kv_native_state,
        "kv_roundtrip": _try_kv_state_roundtrip(llm),
        "phase_intervals_ns": phase_intervals,
        "prompt_position_total": len(prompt_tokens),
        "vision": None,
    }


def run_vision_phase(
    *,
    llm: Any,
    config: VisionPhaseConfig,
    chat_template: str | None,
    snapshot: SnapshotFn,
) -> dict[str, Any]:
    """视觉端点：projector warm → 真实图片编码 → 视觉 token 进入 LLM → 解码。"""

    session = MtmdVisionSession(
        MtmdSessionConfig(
            mmproj_path=Path(config.mmproj_path),
            use_gpu=config.use_gpu,
            n_threads=config.n_threads,
            flash_attn_enabled=_flash_attn_enabled(llm),
            image_max_tokens=config.image_max_tokens,
            image_min_tokens=config.image_min_tokens,
        )
    )
    before_snapshot, before_rss = snapshot()
    session_evidence = session.open(llm)
    after_snapshot, after_rss = snapshot()
    warm_delta = memory_delta(
        before=before_snapshot,
        after=after_snapshot,
        before_rss_bytes=before_rss,
        after_rss_bytes=after_rss,
    )
    bitmaps = []
    chunks = None
    try:
        bitmap = session.bitmap_from_file(config.image_path)
        bitmaps.append(bitmap)
        bitmap_evidence = [item.describe() for item in bitmaps]
        prompt_evidence = build_prompt_evidence(
            mode=config.prompt_mode,
            question=config.question,
            media_marker=session_evidence["media_marker"],
            chat_template=chat_template,
            eos_token=llm.detokenize([llm.token_eos()]).decode(
                "utf-8", errors="replace"
            ),
            bos_token=llm.detokenize([llm.token_bos()]).decode(
                "utf-8", errors="replace"
            ),
        )
        if not prompt_evidence["prompt_contains_media_marker"]:
            raise ValueError("视觉 prompt 中缺少 media marker，图片不会进入模型")
        chunks = session.tokenize(prompt_evidence["prompt_text"], bitmaps)
        chunk_descriptors = chunks.describe()
        text_token_ids = chunks.text_token_ids()
        chunk_evidence = evaluate_chunks(
            llm=llm,
            session=session,
            chunks=chunks,
            mode=config.eval_mode,
        )
        prompt_positions = int(chunk_evidence["n_past_final"])
        if prompt_positions + config.generation_tokens > int(llm.n_ctx()):
            raise ValueError("视觉 prompt 与生成长度超过 n_ctx")
        generation = generate_fixed_tokens(
            llm=llm,
            generation_tokens=config.generation_tokens,
        )
        generated = generation.pop("generated_tokens")
        decode_interval = generation.pop("decode_interval_ns")
        kv_native_state = capture_llama_kv_state(llm)
        output = _output_text_evidence(llm, generated)
        kv_roundtrip = _try_kv_state_roundtrip(llm)
    finally:
        if chunks is not None:
            chunks.free()
        for item in bitmaps:
            item.free()
    release_before_snapshot, release_before_rss = snapshot()
    release_evidence = session.close()
    release_after_snapshot, release_after_rss = snapshot()
    release_delta = memory_delta(
        before=release_before_snapshot,
        after=release_after_snapshot,
        before_rss_bytes=release_before_rss,
        after_rss_bytes=release_after_rss,
    )
    prompt_interval = chunk_evidence["prompt_eval_interval_ns"]
    inference = {
        "prefill_s": chunk_evidence["prompt_eval_s"],
        "prefill_tokens_per_s": (
            prompt_positions / chunk_evidence["prompt_eval_s"]
            if chunk_evidence["prompt_eval_s"] > 0
            else 0.0
        ),
        "first_sample_s": generation["first_sample_s"],
        "ttft_s": chunk_evidence["prompt_eval_s"] + generation["first_sample_s"],
        "decode_visible_s": generation["decode_visible_s"],
        "decode_with_final_eval_s": generation["decode_with_final_eval_s"],
        "decode_tokens_per_s": generation["decode_tokens_per_s"],
        "token_gap_ms": generation["token_gap_ms"],
        "generated_token_count": generation["generated_token_count"],
    }
    return {
        "prompt": {
            **prompt_evidence,
            "prompt_token_count": len(text_token_ids),
            "prompt_tokens_sha256": _token_ids_sha256(text_token_ids),
            "text_token_total_from_chunks": chunk_evidence["text_token_total"],
        },
        "inference": inference,
        "output": output,
        "kv_native_state": kv_native_state,
        "kv_roundtrip": kv_roundtrip,
        "phase_intervals_ns": {
            "prefill": prompt_interval,
            "decode": decode_interval,
            "inference": [prompt_interval[0], decode_interval[1]],
        },
        "prompt_position_total": prompt_positions,
        "vision": {
            "session": session_evidence,
            "release": release_evidence,
            "bitmaps": bitmap_evidence,
            "chunk_descriptors": chunk_descriptors,
            "chunk_execution": chunk_evidence,
            "projector_warm_delta": warm_delta,
            "projector_release_delta": release_delta,
        },
    }


def _flash_attn_enabled(llm: Any) -> bool:
    import llama_cpp  # noqa: PLC0415

    return bool(
        getattr(llm.context_params, "flash_attn_type", None)
        == llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
    )
