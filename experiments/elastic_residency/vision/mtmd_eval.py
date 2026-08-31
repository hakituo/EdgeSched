"""chunk 级真实执行、llama-cpp-python 状态同步与固定长度解码。

两种真实执行模式（都不是 Mock，也不允许静默退回纯文本）：

- ``chunk_single``：图像 chunk 交给 ``mtmd_helper_eval_chunk_single``，与上游
  ``MTMDChatHandler`` 完全一致的调用契约，只给出返回码与位置推进证据；
- ``split_encode``：先 ``mtmd_encode_chunk`` 单独执行 projector，核对
  ``mtmd_get_output_embd`` 返回非空视觉嵌入，再用 ``mtmd_helper_decode_image_chunk``
  把该嵌入送进同一个语言模型上下文，因此能分离 projector 执行时间并直接证明
  视觉嵌入已产生。任何模式下失败都抛错，由上层记录为失败轮次。
"""

from __future__ import annotations

import ctypes
import statistics
import time
from typing import Any

from experiments.elastic_residency.vision.mtmd_runtime import (
    MtmdChunks,
    MtmdUnavailableError,
    MtmdVisionSession,
)
from experiments.tts.profiling_utils import percentile


EVAL_MODES = ("split_encode", "chunk_single")


def _null_post_decode_callback(mtmd: Any) -> Any:
    """构造"不使用"的 post-decode 回调。

    ctypes 不接受 Python `None` 作为 CFUNCTYPE 参数（会报 expected CFunctionType
    instance），必须给该回调类型传一个空函数指针实例。
    """

    return mtmd.mtmd_helper_post_decode_callback()


def _require_capacity(llm: Any, n_past: int, advance: int) -> None:
    limit = int(llm.n_ctx())
    if n_past + advance > limit:
        raise MtmdUnavailableError(
            f"prompt 超出 n_ctx：{n_past} + {advance} > {limit}"
        )


def _eval_text_chunk(
    *,
    llm: Any,
    mtmd: Any,
    chunk: Any,
    n_past: int,
) -> dict[str, Any]:
    count = ctypes.c_size_t()
    pointer = mtmd.mtmd_input_chunk_get_tokens_text(chunk, ctypes.pointer(count))
    if not pointer or not count.value:
        raise MtmdUnavailableError("文本 chunk 没有 token，tokenize 结果无效")
    tokens = [int(pointer[offset]) for offset in range(count.value)]
    _require_capacity(llm, n_past, len(tokens))
    started = time.perf_counter()
    llm.eval(tokens)
    elapsed = time.perf_counter() - started
    return {
        "eval_s": elapsed,
        "token_count": len(tokens),
        "n_past_after": int(llm.n_tokens),
    }


def _eval_image_chunk_single(
    *,
    llm: Any,
    mtmd: Any,
    ctx: Any,
    chunk: Any,
    n_past: int,
    seq_id: int,
    logits_last: bool,
) -> dict[str, Any]:
    import llama_cpp  # noqa: PLC0415

    new_n_past = llama_cpp.llama_pos(0)
    started = time.perf_counter()
    code = int(
        mtmd.mtmd_helper_eval_chunk_single(
            ctx,
            llm._ctx.ctx,
            chunk,
            llama_cpp.llama_pos(n_past),
            llama_cpp.llama_seq_id(seq_id),
            llm.n_batch,
            logits_last,
            ctypes.pointer(new_n_past),
        )
    )
    elapsed = time.perf_counter() - started
    if code != 0:
        raise MtmdUnavailableError(f"mtmd_helper_eval_chunk_single 失败，返回码 {code}")
    return {
        "eval_s": elapsed,
        "projector_encode_s": None,
        "image_decode_s": None,
        "visual_embedding_available": None,
        "return_code": code,
        "n_past_after": int(new_n_past.value),
    }


def _eval_image_chunk_split(
    *,
    llm: Any,
    mtmd: Any,
    ctx: Any,
    chunk: Any,
    n_past: int,
    seq_id: int,
) -> dict[str, Any]:
    import llama_cpp  # noqa: PLC0415

    encode_started = time.perf_counter()
    encode_code = int(mtmd.mtmd_encode_chunk(ctx, chunk))
    encode_s = time.perf_counter() - encode_started
    if encode_code != 0:
        raise MtmdUnavailableError(f"mtmd_encode_chunk 失败，返回码 {encode_code}")
    embeddings = mtmd.mtmd_get_output_embd(ctx)
    if not embeddings:
        raise MtmdUnavailableError("projector 未返回视觉嵌入，视觉输入无效")
    new_n_past = llama_cpp.llama_pos(0)
    decode_started = time.perf_counter()
    decode_code = int(
        mtmd.mtmd_helper_decode_image_chunk(
            ctx,
            llm._ctx.ctx,
            chunk,
            embeddings,
            llama_cpp.llama_pos(n_past),
            llama_cpp.llama_seq_id(seq_id),
            llm.n_batch,
            ctypes.pointer(new_n_past),
            _null_post_decode_callback(mtmd),
            ctypes.c_void_p(0),
        )
    )
    decode_s = time.perf_counter() - decode_started
    if decode_code != 0:
        raise MtmdUnavailableError(
            f"mtmd_helper_decode_image_chunk 失败，返回码 {decode_code}"
        )
    return {
        "eval_s": encode_s + decode_s,
        "projector_encode_s": encode_s,
        "image_decode_s": decode_s,
        "visual_embedding_available": True,
        "return_code": decode_code,
        "n_past_after": int(new_n_past.value),
    }


def evaluate_chunks(
    *,
    llm: Any,
    session: MtmdVisionSession,
    chunks: MtmdChunks,
    mode: str = "split_encode",
    seq_id: int = 0,
) -> dict[str, Any]:
    """真实执行全部 chunk，并把 n_past 同步回 llama-cpp-python。"""

    if mode not in EVAL_MODES:
        raise ValueError(f"未知视觉执行模式: {mode}")
    mtmd = session.mtmd
    ctx = session.require_ctx()
    llm.reset()
    llm._ctx.kv_cache_clear()
    chunk_count = len(chunks)
    if chunk_count == 0:
        raise MtmdUnavailableError("tokenize 没有产生任何 chunk")
    records: list[dict[str, Any]] = []
    total_started_ns = time.monotonic_ns()
    total_started = time.perf_counter()
    for index in range(chunk_count):
        chunk = chunks.chunk_at(index)
        chunk_type = int(mtmd.mtmd_input_chunk_get_type(chunk))
        n_past_before = int(llm.n_tokens)
        entry: dict[str, Any] = {
            "index": index,
            "type": chunk_type,
            "n_pos": int(mtmd.mtmd_input_chunk_get_n_pos(chunk)),
            "n_tokens": int(mtmd.mtmd_input_chunk_get_n_tokens(chunk)),
            "n_past_before": n_past_before,
        }
        if chunk_type == mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT:
            entry["type_name"] = "text"
            entry.update(
                _eval_text_chunk(llm=llm, mtmd=mtmd, chunk=chunk, n_past=n_past_before)
            )
        elif chunk_type == mtmd.MTMD_INPUT_CHUNK_TYPE_IMAGE:
            entry["type_name"] = "image"
            # non-causal 是按 chunk 查询的属性（mtmd_decode_use_non_causal(ctx, chunk)），
            # 不是 context 级能力，因此只能在这里逐 chunk 记录。
            entry["uses_non_causal_encode"] = bool(
                mtmd.mtmd_decode_use_non_causal(ctx, chunk)
            )
            _require_capacity(llm, n_past_before, entry["n_tokens"])
            if mode == "chunk_single":
                entry.update(
                    _eval_image_chunk_single(
                        llm=llm,
                        mtmd=mtmd,
                        ctx=ctx,
                        chunk=chunk,
                        n_past=n_past_before,
                        seq_id=seq_id,
                        logits_last=index == chunk_count - 1,
                    )
                )
            else:
                entry.update(
                    _eval_image_chunk_split(
                        llm=llm,
                        mtmd=mtmd,
                        ctx=ctx,
                        chunk=chunk,
                        n_past=n_past_before,
                        seq_id=seq_id,
                    )
                )
            llm.n_tokens = int(entry["n_past_after"])
        else:
            raise MtmdUnavailableError(
                f"本实验只接受文本与图像 chunk，实际类型 {chunk_type}"
            )
        entry["n_pos_advanced"] = entry["n_past_after"] - n_past_before
        records.append(entry)
    total_s = time.perf_counter() - total_started
    total_finished_ns = time.monotonic_ns()
    image_records = [item for item in records if item["type_name"] == "image"]
    text_records = [item for item in records if item["type_name"] == "text"]

    def _total(field: str) -> float | None:
        values = [item[field] for item in image_records]
        if not values or values[0] is None:
            return None
        return sum(float(value) for value in values)

    embedding_flags = [item["visual_embedding_available"] for item in image_records]
    return {
        "mode": mode,
        "chunk_count": chunk_count,
        "chunks": records,
        "prompt_eval_s": total_s,
        "prompt_eval_interval_ns": [total_started_ns, total_finished_ns],
        "image_chunk_count": len(image_records),
        "text_chunk_count": len(text_records),
        "visual_token_total": sum(int(item["n_tokens"]) for item in image_records),
        "visual_position_total": sum(
            int(item["n_pos_advanced"]) for item in image_records
        ),
        "text_token_total": sum(int(item["token_count"]) for item in text_records),
        "projector_encode_total_s": _total("projector_encode_s"),
        "image_decode_total_s": _total("image_decode_s"),
        "visual_embedding_available": (
            all(embedding_flags) if embedding_flags and embedding_flags[0] is not None
            else None
        ),
        "n_past_final": int(llm.n_tokens),
    }


def generate_fixed_tokens(
    *,
    llm: Any,
    generation_tokens: int,
) -> dict[str, Any]:
    """在 prompt 已评估的上下文上贪心生成固定数量 token（不 reset）。"""

    if generation_tokens < 2:
        raise ValueError("generation_tokens 至少为 2")
    generated: list[int] = []
    token_times: list[float] = []
    started_ns = time.monotonic_ns()
    first_sample_started = time.perf_counter()
    for _ in range(generation_tokens):
        token = int(llm.sample(top_k=1, top_p=1.0, min_p=0.0, temp=0.0))
        token_times.append(time.perf_counter())
        generated.append(token)
        llm.eval([token])
    finished = time.perf_counter()
    finished_ns = time.monotonic_ns()
    gaps_ms = [
        (right - left) * 1000.0 for left, right in zip(token_times, token_times[1:])
    ]
    decode_visible_s = token_times[-1] - token_times[0]
    return {
        "generated_tokens": generated,
        "generated_token_count": len(generated),
        "first_sample_s": token_times[0] - first_sample_started,
        "decode_visible_s": decode_visible_s,
        "decode_with_final_eval_s": finished - token_times[0],
        "decode_tokens_per_s": (
            (len(generated) - 1) / decode_visible_s if decode_visible_s > 0 else 0.0
        ),
        "token_gap_ms": {
            "count": len(gaps_ms),
            "mean": statistics.fmean(gaps_ms),
            "p50": percentile(gaps_ms, 0.50),
            "p95": percentile(gaps_ms, 0.95),
            "max": max(gaps_ms),
            "values": gaps_ms,
        },
        "decode_interval_ns": [started_ns, finished_ns],
    }
