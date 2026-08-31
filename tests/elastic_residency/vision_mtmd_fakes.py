"""假 mtmd 绑定与假 llama 对象，用于离线验证多模态调用契约。

严格边界：这些假件只证明"调用顺序、状态同步与失败路径正确"，
不产生任何 projector 性能、显存或图像理解结论。
"""

from __future__ import annotations

import ctypes
import sys
import types
from typing import Any


CHUNK_TEXT = 0
CHUNK_IMAGE = 1
CHUNK_AUDIO = 2
DEFAULT_MARKER = b"<__media__>"


def install_fake_llama_cpp() -> types.ModuleType:
    """向 sys.modules 注入最小 llama_cpp，仅提供本链路使用的类型与常量。"""

    module = types.ModuleType("llama_cpp")
    module.llama_pos = ctypes.c_int32
    module.llama_seq_id = ctypes.c_int32
    module.LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
    module.LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
    module.llama_get_memory = lambda ctx: 8888
    module.llama_state_get_size = lambda ctx: 70_263_000
    module.llama_state_seq_get_size = lambda ctx, seq_id: 24_000_000
    module.llama_memory_seq_pos_min = lambda memory, seq_id: 0
    module.llama_memory_seq_pos_max = lambda memory, seq_id: 343
    sys.modules["llama_cpp"] = module
    return module


def install_fake_mtmd_module(fake: "FakeMtmd") -> types.ModuleType:
    """把 FakeMtmd 实例暴露为 ``llama_cpp.mtmd_cpp`` 模块，供生产代码 import。"""

    module = types.ModuleType("llama_cpp.mtmd_cpp")
    for name in dir(fake):
        if name.startswith("mtmd_") or name.startswith("MTMD_"):
            setattr(module, name, getattr(fake, name))
    llama_module = sys.modules.get("llama_cpp") or install_fake_llama_cpp()
    llama_module.mtmd_cpp = module
    sys.modules["llama_cpp.mtmd_cpp"] = module
    return module


class FakeContextParams:
    """字段与真实 mtmd_context_params 对齐（本链路实际读写的子集）。"""

    def __init__(self) -> None:
        self.use_gpu = False
        self.print_timings = False
        self.n_threads = 1
        self.flash_attn_type = 0
        self.image_max_tokens = -1
        self.image_min_tokens = -1
        self.media_marker = DEFAULT_MARKER
        self.warmup = True


class FakeInputText(ctypes.Structure):
    """与真实 ``mtmd_input_text`` 同布局，便于 ``ctypes.pointer`` 传递。"""

    _fields_ = [
        ("text", ctypes.c_char_p),
        ("text_len", ctypes.c_size_t),
        ("add_special", ctypes.c_bool),
        ("parse_special", ctypes.c_bool),
    ]


class FakeMtmd:
    """按真实 mtmd_cpp 的函数签名实现的可控假绑定。"""

    MTMD_INPUT_CHUNK_TYPE_TEXT = CHUNK_TEXT
    MTMD_INPUT_CHUNK_TYPE_IMAGE = CHUNK_IMAGE
    MTMD_INPUT_CHUNK_TYPE_AUDIO = CHUNK_AUDIO
    mtmd_bitmap_p_ctypes = ctypes.c_void_p
    mtmd_input_text = FakeInputText

    def __init__(
        self,
        *,
        chunks: list[dict[str, Any]] | None = None,
        init_returns_none: bool = False,
        supports_vision: bool = True,
        tokenize_code: int = 0,
        encode_code: int = 0,
        decode_code: int = 0,
        embeddings_available: bool = True,
        bitmap_available: bool = True,
        declares_vision: bool = True,
    ) -> None:
        self.chunks = chunks or default_chunks()
        self.init_returns_none = init_returns_none
        self.supports_vision = supports_vision
        self.tokenize_code = tokenize_code
        self.encode_code = encode_code
        self.decode_code = decode_code
        self.embeddings_available = embeddings_available
        self.bitmap_available = bitmap_available
        self.declares_vision = declares_vision
        self.calls: list[str] = []
        self.freed_bitmaps: list[int] = []
        self.freed_chunk_handles: list[int] = []
        self.ctx_freed = False

    # ---- 能力与生命周期 ----
    def mtmd_default_marker(self) -> bytes:
        return DEFAULT_MARKER

    def mtmd_get_cap_from_file(self, path: bytes) -> Any:
        self.calls.append("cap_from_file")
        return types.SimpleNamespace(
            inp_vision=self.declares_vision, inp_audio=False
        )

    def mtmd_context_params_default(self) -> FakeContextParams:
        return FakeContextParams()

    def mtmd_init_from_file(self, path: bytes, model: Any, params: Any) -> Any:
        self.calls.append("init")
        self.init_params = params
        return None if self.init_returns_none else 4242

    def mtmd_support_vision(self, ctx: Any) -> bool:
        return self.supports_vision

    def mtmd_support_audio(self, ctx: Any) -> bool:
        return False

    def mtmd_decode_use_mrope(self, ctx: Any) -> bool:
        return True

    def mtmd_decode_use_non_causal(self, ctx: Any, chunk: Any) -> bool:
        # 与真实绑定一致：按 chunk 查询（mtmd_decode_use_non_causal(ctx, chunk)）。
        self.calls.append("use_non_causal")
        return True

    def mtmd_get_marker(self, ctx: Any) -> bytes:
        return DEFAULT_MARKER

    def mtmd_free(self, ctx: Any) -> None:
        self.calls.append("free")
        self.ctx_freed = True

    # ---- bitmap ----
    def mtmd_helper_bitmap_init_from_file(
        self, ctx: Any, path: bytes, placeholder: bool
    ) -> Any:
        self.calls.append("bitmap_init")
        return 1001 if self.bitmap_available else None

    def mtmd_bitmap_get_nx(self, handle: Any) -> int:
        return 448

    def mtmd_bitmap_get_ny(self, handle: Any) -> int:
        return 448

    def mtmd_bitmap_get_n_bytes(self, handle: Any) -> int:
        return 448 * 448 * 3

    def mtmd_bitmap_is_audio(self, handle: Any) -> bool:
        return False

    def mtmd_bitmap_get_id(self, handle: Any) -> bytes:
        return b"fixed-image"

    def mtmd_bitmap_free(self, handle: Any) -> None:
        self.freed_bitmaps.append(int(handle))

    # ---- chunks ----
    def mtmd_input_chunks_init(self) -> Any:
        return 7777

    def mtmd_input_chunks_free(self, handle: Any) -> None:
        self.freed_chunk_handles.append(int(handle))

    def mtmd_input_chunks_size(self, handle: Any) -> int:
        return len(self.chunks)

    def mtmd_input_chunks_get(self, handle: Any, index: int) -> Any:
        return index if index < len(self.chunks) else None

    def mtmd_tokenize(
        self, ctx: Any, chunks: Any, text_ptr: Any, bitmaps: Any, count: int
    ) -> int:
        self.calls.append("tokenize")
        self.tokenized_text = bytes(text_ptr.contents.text)
        self.tokenized_bitmap_count = int(count)
        return self.tokenize_code

    def mtmd_input_chunk_get_type(self, chunk: Any) -> int:
        return int(self.chunks[chunk]["type"])

    def mtmd_input_chunk_get_n_tokens(self, chunk: Any) -> int:
        return int(self.chunks[chunk]["n_tokens"])

    def mtmd_input_chunk_get_n_pos(self, chunk: Any) -> int:
        return int(self.chunks[chunk]["n_pos"])

    def mtmd_input_chunk_get_tokens_text(self, chunk: Any, count_ptr: Any) -> Any:
        tokens = self.chunks[chunk].get("tokens") or []
        count_ptr[0] = len(tokens)
        if not tokens:
            return None
        return (ctypes.c_int32 * len(tokens))(*tokens)

    def mtmd_input_chunk_get_tokens_image(self, chunk: Any) -> Any:
        return chunk if self.chunks[chunk]["type"] == CHUNK_IMAGE else None

    def mtmd_image_tokens_get_n_tokens(self, image_tokens: Any) -> int:
        return int(self.chunks[image_tokens]["n_tokens"])

    def mtmd_image_tokens_get_nx(self, image_tokens: Any) -> int:
        return 14

    def mtmd_image_tokens_get_ny(self, image_tokens: Any) -> int:
        return 14

    def mtmd_image_tokens_get_n_pos(self, image_tokens: Any) -> int:
        return int(self.chunks[image_tokens]["n_pos"])

    # ---- 执行 ----
    def mtmd_encode_chunk(self, ctx: Any, chunk: Any) -> int:
        self.calls.append("encode")
        return self.encode_code

    def mtmd_get_output_embd(self, ctx: Any) -> Any:
        if not self.embeddings_available:
            return None
        return (ctypes.c_float * 4)(0.1, 0.2, 0.3, 0.4)

    def mtmd_helper_decode_image_chunk(
        self,
        ctx: Any,
        lctx: Any,
        chunk: Any,
        embd: Any,
        n_past: Any,
        seq_id: Any,
        n_batch: Any,
        new_n_past_ptr: Any,
        callback: Any,
        user_data: Any,
    ) -> int:
        self.calls.append("decode_image")
        new_n_past_ptr[0] = int(n_past.value) + self.chunks[chunk]["n_pos"]
        return self.decode_code

    def mtmd_helper_eval_chunk_single(
        self,
        ctx: Any,
        lctx: Any,
        chunk: Any,
        n_past: Any,
        seq_id: Any,
        n_batch: Any,
        logits_last: Any,
        new_n_past_ptr: Any,
    ) -> int:
        self.calls.append("eval_chunk_single")
        new_n_past_ptr[0] = int(n_past.value) + self.chunks[chunk]["n_pos"]
        return self.decode_code


def default_chunks() -> list[dict[str, Any]]:
    """默认 chunk 序列：文本 → 图像 → 文本，与真实对话 prompt 结构一致。"""

    return [
        {"type": CHUNK_TEXT, "n_tokens": 12, "n_pos": 12, "tokens": list(range(12))},
        {"type": CHUNK_IMAGE, "n_tokens": 196, "n_pos": 196},
        {"type": CHUNK_TEXT, "n_tokens": 8, "n_pos": 8, "tokens": list(range(100, 108))},
    ]


class FakeLlamaContext:
    def __init__(self) -> None:
        self.ctx = 9001
        self.memory = 8888
        self.cleared = 0

    def kv_cache_clear(self) -> None:
        self.cleared += 1


class FakeLlama:
    """最小 llama-cpp-python 行为：token 计数、eval、sample 与 detokenize。"""

    def __init__(self, *, n_ctx: int = 640, flash_attn_enabled: bool = True) -> None:
        self.model = 5005
        self._ctx = FakeLlamaContext()
        self.n_batch = 512
        self.n_threads = 14
        self.n_tokens = 0
        self._n_ctx = n_ctx
        self.evaluated: list[list[int]] = []
        self.resets = 0
        self.context_params = types.SimpleNamespace(
            flash_attn_type=1 if flash_attn_enabled else 0
        )
        self.metadata: dict[str, str] = {}

    def n_ctx(self) -> int:
        return self._n_ctx

    def reset(self) -> None:
        self.resets += 1
        self.n_tokens = 0

    def eval(self, tokens: list[int]) -> None:
        self.evaluated.append(list(tokens))
        self.n_tokens += len(tokens)

    def sample(self, **_: Any) -> int:
        return 500 + self.n_tokens

    def detokenize(self, tokens: list[int]) -> bytes:
        return ("|".join(str(token) for token in tokens)).encode("utf-8")

    def token_eos(self) -> int:
        return 248046

    def token_bos(self) -> int:
        return 1

    def save_state(self) -> Any:
        self._saved_tokens = self.n_tokens
        return types.SimpleNamespace(llama_state_size=70_263_000)

    def load_state(self, state: Any) -> None:
        self.n_tokens = getattr(self, "_saved_tokens", 0)

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False):
        # 每字节一个 token，只为让精确 token 数构造器快速收敛。
        return list(range(len(text)))
