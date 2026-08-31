"""GGUF 头部元数据只读解析（纯标准库，不加载模型、不占显存）。

用途：在任何 GPU 运行之前核验主 GGUF 与 mmproj 的架构、层数、张量数量和
视觉编码器声明，并把结果写进实验 manifest。

边界：
- 只解析 metadata KV 与张量计数，不解析张量数据，不做任何推理；
- 数组字段只保留元素类型、数量与前若干个元素预览，避免把词表整体读进 JSON；
- 版本或字段无法解析时抛错，缺失键返回 ``None``，禁止填默认值冒充。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO


GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS: tuple[int, ...] = (2, 3)
ARRAY_PREVIEW_LIMIT = 8

TYPE_UINT8 = 0
TYPE_INT8 = 1
TYPE_UINT16 = 2
TYPE_INT16 = 3
TYPE_UINT32 = 4
TYPE_INT32 = 5
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10
TYPE_INT64 = 11
TYPE_FLOAT64 = 12

_SCALAR_FORMATS: dict[int, str] = {
    TYPE_UINT8: "<B",
    TYPE_INT8: "<b",
    TYPE_UINT16: "<H",
    TYPE_INT16: "<h",
    TYPE_UINT32: "<I",
    TYPE_INT32: "<i",
    TYPE_FLOAT32: "<f",
    TYPE_BOOL: "<?",
    TYPE_UINT64: "<Q",
    TYPE_INT64: "<q",
    TYPE_FLOAT64: "<d",
}

TYPE_NAMES: dict[int, str] = {
    TYPE_UINT8: "uint8",
    TYPE_INT8: "int8",
    TYPE_UINT16: "uint16",
    TYPE_INT16: "int16",
    TYPE_UINT32: "uint32",
    TYPE_INT32: "int32",
    TYPE_FLOAT32: "float32",
    TYPE_BOOL: "bool",
    TYPE_STRING: "string",
    TYPE_ARRAY: "array",
    TYPE_UINT64: "uint64",
    TYPE_INT64: "int64",
    TYPE_FLOAT64: "float64",
}


class GGUFParseError(ValueError):
    """GGUF 头部不符合规范或使用了未支持的版本。"""


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    if size < 0:
        raise GGUFParseError(f"非法读取长度: {size}")
    payload = stream.read(size)
    if len(payload) != size:
        raise GGUFParseError(f"GGUF 头部提前结束：期望 {size} 字节，实际 {len(payload)}")
    return payload


def _read_scalar(stream: BinaryIO, value_type: int) -> Any:
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GGUFParseError(f"未支持的 GGUF 标量类型: {value_type}")
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(stream, size))[0]


def _read_string(stream: BinaryIO) -> str:
    length = struct.unpack("<Q", _read_exact(stream, 8))[0]
    return _read_exact(stream, length).decode("utf-8", errors="replace")


def _read_array(
    stream: BinaryIO,
    *,
    preview_limit: int,
) -> dict[str, Any]:
    element_type = struct.unpack("<I", _read_exact(stream, 4))[0]
    count = struct.unpack("<Q", _read_exact(stream, 8))[0]
    if element_type == TYPE_ARRAY:
        raise GGUFParseError("GGUF 不支持嵌套数组")
    preview: list[Any] = []
    for index in range(count):
        if element_type == TYPE_STRING:
            value = _read_string(stream)
        else:
            value = _read_scalar(stream, element_type)
        if index < preview_limit:
            preview.append(value)
    return {
        "kind": "array",
        "element_type": element_type,
        "element_type_name": TYPE_NAMES.get(element_type, f"unknown_{element_type}"),
        "count": int(count),
        "preview": preview,
        "preview_truncated": bool(count > preview_limit),
    }


def _read_value(stream: BinaryIO, value_type: int, *, preview_limit: int) -> Any:
    if value_type == TYPE_STRING:
        return _read_string(stream)
    if value_type == TYPE_ARRAY:
        return _read_array(stream, preview_limit=preview_limit)
    return _read_scalar(stream, value_type)


def read_gguf_header(
    path: str | Path,
    *,
    array_preview_limit: int = ARRAY_PREVIEW_LIMIT,
) -> dict[str, Any]:
    """读取单个 GGUF 文件的版本、张量计数与全部 metadata KV。"""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到 GGUF 文件: {resolved}")
    with resolved.open("rb") as stream:
        magic = _read_exact(stream, 4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError(f"不是 GGUF 文件（magic={magic!r}）: {resolved}")
        version = struct.unpack("<I", _read_exact(stream, 4))[0]
        if version not in SUPPORTED_VERSIONS:
            raise GGUFParseError(
                f"未支持的 GGUF 版本 {version}，当前只解析 {SUPPORTED_VERSIONS}"
            )
        tensor_count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        kv_count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        metadata: dict[str, Any] = {}
        value_types: dict[str, str] = {}
        for _ in range(kv_count):
            key = _read_string(stream)
            value_type = struct.unpack("<I", _read_exact(stream, 4))[0]
            metadata[key] = _read_value(
                stream,
                value_type,
                preview_limit=array_preview_limit,
            )
            value_types[key] = TYPE_NAMES.get(value_type, f"unknown_{value_type}")
        header_bytes = stream.tell()
    return {
        "path": str(resolved),
        "gguf_version": int(version),
        "tensor_count": int(tensor_count),
        "metadata_kv_count": int(kv_count),
        "metadata": metadata,
        "metadata_value_types": value_types,
        "header_bytes": int(header_bytes),
        "file_size_bytes": resolved.stat().st_size,
    }


def scalar_metadata(header: dict[str, Any]) -> dict[str, Any]:
    """只保留标量与字符串 KV；数组字段折叠为类型与数量摘要。"""

    result: dict[str, Any] = {}
    for key, value in header["metadata"].items():
        if isinstance(value, dict) and value.get("kind") == "array":
            result[key] = {
                "array_element_type": value["element_type_name"],
                "array_count": value["count"],
            }
        else:
            result[key] = value
    return result


def metadata_int(header: dict[str, Any], key: str) -> int | None:
    """读取整数型 KV；缺失或类型不符时返回 ``None``，不填 0 冒充。"""

    value = header["metadata"].get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not float(value).is_integer():
        return None
    return int(value)


def metadata_str(header: dict[str, Any], key: str) -> str | None:
    """读取字符串型 KV；缺失或类型不符时返回 ``None``。"""

    value = header["metadata"].get(key)
    return value if isinstance(value, str) else None


def metadata_bool(header: dict[str, Any], key: str) -> bool | None:
    """读取布尔型 KV；缺失或类型不符时返回 ``None``。"""

    value = header["metadata"].get(key)
    return bool(value) if isinstance(value, bool) else None
