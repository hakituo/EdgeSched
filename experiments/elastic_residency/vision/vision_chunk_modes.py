"""两种真实视觉执行模式的具体调用。

- ``chunk_single``：与 llama-cpp-python 上游 ``MTMDChatHandler`` 完全一致的
  调用契约，只给出返回码与位置推进证据；
- ``split_encode``：先 ``mtmd_encode_chunk`` 单独执行 projector，核对
  ``mtmd_get_output_embd`` 返回非空视觉嵌入，再用
  ``mtmd_helper_decode_image_chunk`` 把该嵌入送进同一个语言模型上下文，因此
  能分离 projector 执行时间并直接证明视觉嵌入已产生。

任何模式下失败都抛错，由上层记录为失败轮次；没有静默退回纯文本的分支。
"""

from __future__ import annotations

import ctypes
import time
from typing import Any

from experiments.elastic_residency.vision.mtmd_runtime import MtmdUnavailableError


def require_capacity(llm: Any, n_past: int, advance: int) -> None:
    limit = int(llm.n_ctx())
    if n_past + advance > limit:
        raise MtmdUnavailableError(
            f"prompt 超出 n_ctx：{n_past} + {advance} > {limit}"
        )


def null_post_decode_callback(mtmd: Any) -> Any:
    """构造"不使用"的 post-decode 回调。

    ctypes 不接受 Python `None` 作为 CFUNCTYPE 参数（会报 expected CFunctionType
    instance），必须给该回调类型传一个空函数指针实例。
    """

    return mtmd.mtmd_helper_post_decode_callback()


def eval_text_chunk(
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
    require_capacity(llm, n_past, len(tokens))
    started = time.perf_counter()
    llm.eval(tokens)
    elapsed = time.perf_counter() - started
    return {
        "eval_s": elapsed,
        "token_count": len(tokens),
        "n_past_after": int(llm.n_tokens),
    }


def eval_image_chunk_single(
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


def eval_image_chunk_split(
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
            null_post_decode_callback(mtmd),
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
