"""视觉端点测试用的合成 GGUF 与假 mtmd 绑定构造器。

只用于静态测试链路：合成 GGUF 只写头部 KV，不含张量数据；假 mtmd 绑定只验证
调用顺序与失败路径，不产生任何性能或可行性结论。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.vision import gguf_reader as gr  # noqa: E402


def encode_string(value: str) -> bytes:
    payload = value.encode("utf-8")
    return struct.pack("<Q", len(payload)) + payload


def encode_value(value_type: int, value: Any) -> bytes:
    if value_type == gr.TYPE_STRING:
        return encode_string(value)
    if value_type == gr.TYPE_ARRAY:
        element_type, items = value
        encoded = struct.pack("<I", element_type) + struct.pack("<Q", len(items))
        for item in items:
            encoded += (
                encode_string(item)
                if element_type == gr.TYPE_STRING
                else struct.pack(gr._SCALAR_FORMATS[element_type], item)
            )
        return encoded
    return struct.pack(gr._SCALAR_FORMATS[value_type], value)


def build_gguf_bytes(
    metadata: dict[str, tuple[int, Any]],
    *,
    version: int = 3,
    tensor_count: int = 0,
) -> bytes:
    body = b""
    for key, (value_type, value) in metadata.items():
        body += encode_string(key)
        body += struct.pack("<I", value_type)
        body += encode_value(value_type, value)
    return (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", len(metadata))
        + body
    )


def write_gguf(
    path: Path,
    metadata: dict[str, tuple[int, Any]],
    *,
    version: int = 3,
    tensor_count: int = 0,
) -> Path:
    path.write_bytes(
        build_gguf_bytes(metadata, version=version, tensor_count=tensor_count)
    )
    return path


def main_model_metadata(
    *,
    architecture: str = "qwen35",
    block_count: int = 32,
    embedding_length: int = 2560,
    context_length: int = 262144,
    basename: str = "Qwen3.5-4B",
    with_chat_template: bool = True,
) -> dict[str, tuple[int, Any]]:
    metadata: dict[str, tuple[int, Any]] = {
        "general.architecture": (gr.TYPE_STRING, architecture),
        "general.name": (gr.TYPE_STRING, basename),
        "general.basename": (gr.TYPE_STRING, basename),
        "general.type": (gr.TYPE_STRING, "model"),
        "general.file_type": (gr.TYPE_UINT32, 15),
        "general.base_model.0.name": (gr.TYPE_STRING, "Qwen3.5 4B"),
        "general.base_model.0.repo_url": (
            gr.TYPE_STRING,
            "https://huggingface.co/Qwen/Qwen3.5-4B",
        ),
        "general.quantized_by": (gr.TYPE_STRING, "Unsloth"),
        f"{architecture}.block_count": (gr.TYPE_UINT32, block_count),
        f"{architecture}.context_length": (gr.TYPE_UINT32, context_length),
        f"{architecture}.embedding_length": (gr.TYPE_UINT32, embedding_length),
        f"{architecture}.attention.head_count": (gr.TYPE_UINT32, 16),
        f"{architecture}.attention.head_count_kv": (gr.TYPE_UINT32, 4),
        f"{architecture}.rope.dimension_sections": (
            gr.TYPE_ARRAY,
            (gr.TYPE_INT32, [16, 24, 24, 0]),
        ),
        "tokenizer.ggml.tokens": (
            gr.TYPE_ARRAY,
            (gr.TYPE_STRING, ["<|im_start|>", "<|im_end|>", "hello"]),
        ),
    }
    if with_chat_template:
        metadata["tokenizer.chat_template"] = (
            gr.TYPE_STRING,
            "{%- for message in messages %}{{- '<|im_start|>' + message.role + "
            "'\\n' + message.content + '<|im_end|>\\n' }}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- endif %}",
        )
    return metadata


def mmproj_metadata(
    *,
    basename: str = "Qwen3.5-4B",
    projection_dim: int = 2560,
    has_vision_encoder: bool = True,
    patch_size: int = 16,
    spatial_merge_size: int = 2,
) -> dict[str, tuple[int, Any]]:
    return {
        "general.architecture": (gr.TYPE_STRING, "clip"),
        "general.type": (gr.TYPE_STRING, "mmproj"),
        "general.name": (gr.TYPE_STRING, basename),
        "general.basename": (gr.TYPE_STRING, basename),
        "general.file_type": (gr.TYPE_UINT32, 32),
        "general.base_model.0.name": (gr.TYPE_STRING, "Qwen3.5 4B"),
        "general.base_model.0.repo_url": (
            gr.TYPE_STRING,
            "https://huggingface.co/Qwen/Qwen3.5-4B",
        ),
        "clip.has_vision_encoder": (gr.TYPE_BOOL, has_vision_encoder),
        "clip.projector_type": (gr.TYPE_STRING, "qwen3vl_merger"),
        "clip.vision.block_count": (gr.TYPE_UINT32, 24),
        "clip.vision.embedding_length": (gr.TYPE_UINT32, 1024),
        "clip.vision.image_size": (gr.TYPE_UINT32, 768),
        "clip.vision.patch_size": (gr.TYPE_UINT32, patch_size),
        "clip.vision.spatial_merge_size": (gr.TYPE_UINT32, spatial_merge_size),
        "clip.vision.projection_dim": (gr.TYPE_UINT32, projection_dim),
        "clip.vision.image_mean": (gr.TYPE_ARRAY, (gr.TYPE_FLOAT32, [0.5, 0.5, 0.5])),
    }


def write_artifact_pair(
    directory: Path,
    *,
    main_kwargs: dict[str, Any] | None = None,
    mmproj_kwargs: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """写出一对合成的主模型/mmproj GGUF 文件。"""

    main_path = write_gguf(
        directory / "main.gguf",
        main_model_metadata(**(main_kwargs or {})),
        tensor_count=426,
    )
    mmproj_path = write_gguf(
        directory / "mmproj.gguf",
        mmproj_metadata(**(mmproj_kwargs or {})),
        tensor_count=298,
    )
    return main_path, mmproj_path
