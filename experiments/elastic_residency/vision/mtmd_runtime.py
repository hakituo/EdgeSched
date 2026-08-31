"""libmtmd 真实视觉运行后端：projector warm、bitmap、tokenize 与释放。

只封装 llama-cpp-python 随包提供的 `llama_cpp.mtmd_cpp` ctypes 绑定，不改变任何
模型初始化参数，也不提供"视觉失败就退回纯文本"的静默路径：projector 不可用时
直接抛 `MtmdUnavailableError`。`use_gpu=True/False` 分别对应 `VISION_WARM_GPU`
与 `VISION_WARM_CPU`，两者都是真实执行，区别只在 projector 所在设备。
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHUNK_TYPE_NAMES = {0: "text", 1: "image", 2: "audio"}


class MtmdUnavailableError(RuntimeError):
    """当前环境或 artifact 无法提供真实多模态执行路径。"""


@dataclass(frozen=True)
class MtmdSessionConfig:
    """projector 会话的冻结配置。"""

    mmproj_path: Path
    use_gpu: bool
    n_threads: int
    flash_attn_enabled: bool
    image_max_tokens: int = 0
    image_min_tokens: int = 0
    print_timings: bool = False

    def residency_state(self) -> str:
        return "VISION_WARM_GPU" if self.use_gpu else "VISION_WARM_CPU"


def import_mtmd() -> Any:
    """导入 mtmd 绑定；缺失时明确失败。"""

    try:
        import llama_cpp.mtmd_cpp as mtmd_cpp  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - 依赖缺失路径
        raise MtmdUnavailableError(f"当前 llama-cpp-python 没有 mtmd 绑定: {exc}") from exc
    return mtmd_cpp


def probe_mmproj_caps(mmproj_path: str | Path, *, mtmd: Any | None = None) -> dict[str, Any]:
    """不加载语言模型，只读取 mmproj 声明的输入能力。"""

    module = mtmd or import_mtmd()
    resolved = Path(mmproj_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到 mmproj: {resolved}")
    caps = module.mtmd_get_cap_from_file(str(resolved).encode("utf-8"))
    return {
        "mmproj_path": str(resolved),
        "declares_vision_input": bool(caps.inp_vision),
        "declares_audio_input": bool(caps.inp_audio),
        "default_media_marker": module.mtmd_default_marker().decode("utf-8"),
    }


class MtmdBitmap:
    """单张真实图片解码后的 bitmap 句柄。"""

    def __init__(self, mtmd: Any, handle: Any, source_path: Path) -> None:
        self._mtmd = mtmd
        self.handle = handle
        self.source_path = source_path

    def describe(self) -> dict[str, Any]:
        identifier = self._mtmd.mtmd_bitmap_get_id(self.handle)
        return {
            "source_path": str(self.source_path),
            "decoded_width_px": int(self._mtmd.mtmd_bitmap_get_nx(self.handle)),
            "decoded_height_px": int(self._mtmd.mtmd_bitmap_get_ny(self.handle)),
            "decoded_bytes": int(self._mtmd.mtmd_bitmap_get_n_bytes(self.handle)),
            "is_audio": bool(self._mtmd.mtmd_bitmap_is_audio(self.handle)),
            "bitmap_id": identifier.decode("utf-8", errors="replace") if identifier else None,
        }

    def free(self) -> None:
        if self.handle is not None:
            self._mtmd.mtmd_bitmap_free(self.handle)
            self.handle = None


class MtmdChunks:
    """`mtmd_tokenize` 输出的 chunk 序列句柄。"""

    def __init__(self, mtmd: Any, handle: Any) -> None:
        self._mtmd = mtmd
        self.handle = handle

    def __len__(self) -> int:
        return int(self._mtmd.mtmd_input_chunks_size(self.handle))

    def chunk_at(self, index: int) -> Any:
        chunk = self._mtmd.mtmd_input_chunks_get(self.handle, index)
        if chunk is None:
            raise MtmdUnavailableError(f"chunk {index} 不可用，tokenize 结果无效")
        return chunk

    def describe(self) -> list[dict[str, Any]]:
        described: list[dict[str, Any]] = []
        for index in range(len(self)):
            chunk = self.chunk_at(index)
            chunk_type = int(self._mtmd.mtmd_input_chunk_get_type(chunk))
            entry: dict[str, Any] = {
                "index": index,
                "type": chunk_type,
                "type_name": CHUNK_TYPE_NAMES.get(chunk_type, f"unknown_{chunk_type}"),
                "n_tokens": int(self._mtmd.mtmd_input_chunk_get_n_tokens(chunk)),
                "n_pos": int(self._mtmd.mtmd_input_chunk_get_n_pos(chunk)),
                "image_tokens": None,
            }
            if chunk_type == self._mtmd.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                image_tokens = self._mtmd.mtmd_input_chunk_get_tokens_image(chunk)
                if image_tokens is None:
                    raise MtmdUnavailableError("图像 chunk 缺少 image tokens，视觉输入无效")
                entry["image_tokens"] = {
                    "n_tokens": int(
                        self._mtmd.mtmd_image_tokens_get_n_tokens(image_tokens)
                    ),
                    "grid_nx": int(self._mtmd.mtmd_image_tokens_get_nx(image_tokens)),
                    "grid_ny": int(self._mtmd.mtmd_image_tokens_get_ny(image_tokens)),
                    "n_pos": int(
                        self._mtmd.mtmd_image_tokens_get_n_pos(image_tokens)
                    ),
                }
            described.append(entry)
        return described

    def text_token_ids(self) -> list[int]:
        tokens: list[int] = []
        for index in range(len(self)):
            chunk = self.chunk_at(index)
            if int(self._mtmd.mtmd_input_chunk_get_type(chunk)) != (
                self._mtmd.MTMD_INPUT_CHUNK_TYPE_TEXT
            ):
                continue
            count = ctypes.c_size_t()
            # 用真实指针而非 byref：两者对 ctypes 函数等价，但指针对象可被
            # 静态测试的假绑定写回，便于离线验证调用契约。
            pointer = self._mtmd.mtmd_input_chunk_get_tokens_text(
                chunk, ctypes.pointer(count)
            )
            if pointer and count.value:
                tokens.extend(int(pointer[offset]) for offset in range(count.value))
        return tokens

    def free(self) -> None:
        if self.handle is not None:
            self._mtmd.mtmd_input_chunks_free(self.handle)
            self.handle = None


class MtmdVisionSession:
    """一次 projector warm 生命周期：init → tokenize/执行 → free。"""

    def __init__(self, config: MtmdSessionConfig, *, mtmd: Any | None = None) -> None:
        self.config = config
        self._mtmd = mtmd or import_mtmd()
        self.ctx: Any = None
        self.open_evidence: dict[str, Any] = {}

    @property
    def mtmd(self) -> Any:
        """暴露底层 mtmd 绑定，供 chunk 执行模块使用。"""

        return self._mtmd

    def open(self, llama: Any) -> dict[str, Any]:
        """在已加载的语言模型上初始化 projector，并记录加载耗时与能力。"""

        if self.ctx is not None:
            raise MtmdUnavailableError("projector 会话已初始化，禁止重复 warm")
        mmproj = Path(self.config.mmproj_path).expanduser().resolve()
        if not mmproj.is_file():
            raise FileNotFoundError(f"找不到 mmproj: {mmproj}")
        params = self._mtmd.mtmd_context_params_default()
        params.use_gpu = bool(self.config.use_gpu)
        params.print_timings = bool(self.config.print_timings)
        params.n_threads = int(self.config.n_threads)
        params.flash_attn_type = self._flash_attn_type()
        if self.config.image_max_tokens > 0:
            params.image_max_tokens = int(self.config.image_max_tokens)
        if self.config.image_min_tokens > 0:
            params.image_min_tokens = int(self.config.image_min_tokens)
        started = time.perf_counter()
        started_ns = time.monotonic_ns()
        ctx = self._mtmd.mtmd_init_from_file(
            str(mmproj).encode("utf-8"), llama.model, params
        )
        load_s = time.perf_counter() - started
        finished_ns = time.monotonic_ns()
        if ctx is None:
            raise MtmdUnavailableError(f"mtmd_init_from_file 失败: {mmproj}")
        self.ctx = ctx
        if not self._mtmd.mtmd_support_vision(ctx):
            self.close()
            raise MtmdUnavailableError("projector 已加载但不支持视觉输入")
        marker = self._mtmd.mtmd_get_marker(ctx)
        self.open_evidence = {
            "residency_state": self.config.residency_state(),
            "projector_use_gpu": bool(self.config.use_gpu),
            "projector_load_s": load_s,
            "projector_load_interval_ns": [started_ns, finished_ns],
            "supports_vision": True,
            "supports_audio": bool(self._mtmd.mtmd_support_audio(ctx)),
            "uses_mrope": bool(self._mtmd.mtmd_decode_use_mrope(ctx)),
            "uses_non_causal_encode": bool(
                self._mtmd.mtmd_decode_use_non_causal(ctx)
            ),
            "media_marker": (
                marker.decode("utf-8") if marker else None
            ),
            "requested_image_max_tokens": int(self.config.image_max_tokens),
            "requested_image_min_tokens": int(self.config.image_min_tokens),
            "n_threads": int(self.config.n_threads),
            "flash_attn_enabled": bool(self.config.flash_attn_enabled),
        }
        return dict(self.open_evidence)

    def _flash_attn_type(self) -> int:
        import llama_cpp  # noqa: PLC0415

        return (
            llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
            if self.config.flash_attn_enabled
            else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
        )

    def require_ctx(self) -> Any:
        if self.ctx is None:
            raise MtmdUnavailableError("projector 会话未初始化")
        return self.ctx

    def bitmap_from_file(self, image_path: str | Path) -> MtmdBitmap:
        """用 libmtmd 自带解码器读取真实图片文件。"""

        resolved = Path(image_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"找不到图片: {resolved}")
        handle = self._mtmd.mtmd_helper_bitmap_init_from_file(
            self.require_ctx(), str(resolved).encode("utf-8"), False
        )
        if handle is None:
            raise MtmdUnavailableError(f"libmtmd 无法解码图片: {resolved}")
        return MtmdBitmap(self._mtmd, handle, resolved)

    def tokenize(self, text: str, bitmaps: list[MtmdBitmap]) -> MtmdChunks:
        """把带 media marker 的 prompt 与真实 bitmap 切成 chunk 序列。"""

        ctx = self.require_ctx()
        chunks_handle = self._mtmd.mtmd_input_chunks_init()
        if chunks_handle is None:
            raise MtmdUnavailableError("mtmd_input_chunks_init 失败")
        chunks = MtmdChunks(self._mtmd, chunks_handle)
        encoded = text.encode("utf-8")
        input_text = self._mtmd.mtmd_input_text()
        input_text.text = encoded
        input_text.text_len = len(encoded)
        input_text.add_special = True
        input_text.parse_special = True
        handles = [bitmap.handle for bitmap in bitmaps]
        bitmap_array = (self._mtmd.mtmd_bitmap_p_ctypes * len(handles))(*handles)
        code = int(
            self._mtmd.mtmd_tokenize(
                ctx,
                chunks_handle,
                ctypes.pointer(input_text),
                bitmap_array,
                len(handles),
            )
        )
        if code != 0:
            chunks.free()
            raise MtmdUnavailableError(f"mtmd_tokenize 失败，返回码 {code}")
        return chunks

    def close(self) -> dict[str, Any]:
        """释放 projector，并记录释放耗时。"""

        if self.ctx is None:
            return {"projector_free_s": None, "projector_freed": False}
        started = time.perf_counter()
        started_ns = time.monotonic_ns()
        self._mtmd.mtmd_free(self.ctx)
        free_s = time.perf_counter() - started
        finished_ns = time.monotonic_ns()
        self.ctx = None
        return {
            "projector_free_s": free_s,
            "projector_freed": True,
            "projector_free_interval_ns": [started_ns, finished_ns],
        }
