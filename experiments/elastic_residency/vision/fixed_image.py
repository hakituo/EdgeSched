"""确定性固定测试图片：纯标准库生成，像素内容哈希与编码器无关。

为什么不直接提交一张二进制图片：实验需要能在任何机器上重建同一张图，
并且必须能证明"这张图没有被悄悄换掉"。因此这里固定一份声明式 spec：

- ``spec_sha256``：spec 的规范化 JSON 哈希，描述"要画什么"；
- ``pixel_payload_sha256``：渲染出的原始 RGB 字节哈希，与 PNG 编码器/zlib 版本无关，
  是图片内容的真实身份；
- ``file_sha256``：落盘 PNG 的文件哈希，随 zlib 实现可能变化，只作记录，不作门禁。

图片内容是可被视觉模型描述的简单几何：白底、黑色边框、居中红色圆、
左上蓝色方块、右下绿色方块。它只用于证明 projector 真实执行并产生视觉 token，
不用于任何图像理解质量结论。
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any


FIXED_IMAGE_SPEC: dict[str, Any] = {
    "spec_id": "edgesched_vision_shapes_448_v1",
    "width": 448,
    "height": 448,
    "background_rgb": [255, 255, 255],
    "border": {"thickness": 8, "rgb": [0, 0, 0]},
    "circle": {"center_x": 224, "center_y": 224, "radius": 140, "rgb": [220, 30, 30]},
    "squares": [
        {"x0": 24, "y0": 24, "x1": 120, "y1": 120, "rgb": [30, 60, 220]},
        {"x0": 328, "y0": 328, "x1": 424, "y1": 424, "rgb": [30, 160, 60]},
    ],
    "description_zh": "白色背景、黑色边框、居中红色圆形、左上蓝色方块、右下绿色方块",
}

# 由 spec 渲染出的原始 RGB 字节哈希；spec 或渲染逻辑变化会立刻触发失败。
EXPECTED_PIXEL_PAYLOAD_SHA256 = (
    "a1514034f07cd8362119d5fd124948d1b0970515d83fdba535c5b3a7b243cb00"
)
# spec 规范化 JSON 哈希：d4c6fc1c7f5cb4420a14470cd775fc9f90b6fe2c0644ab62f2913b0d49a0efb8


def spec_sha256(spec: dict[str, Any] | None = None) -> str:
    """对 spec 的规范化 JSON 求哈希，作为"要画什么"的稳定标识。"""

    payload = json.dumps(
        spec if spec is not None else FIXED_IMAGE_SPEC,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_rgb_pixels(spec: dict[str, Any] | None = None) -> bytes:
    """按 spec 渲染 RGB 像素（不含 PNG 行过滤字节）。"""

    spec = spec if spec is not None else FIXED_IMAGE_SPEC
    width = int(spec["width"])
    height = int(spec["height"])
    if width <= 0 or height <= 0:
        raise ValueError("固定图片尺寸必须为正")
    background = bytes(int(value) for value in spec["background_rgb"])
    border = spec["border"]
    border_thickness = int(border["thickness"])
    border_rgb = bytes(int(value) for value in border["rgb"])
    circle = spec["circle"]
    center_x = int(circle["center_x"])
    center_y = int(circle["center_y"])
    radius_squared = int(circle["radius"]) ** 2
    circle_rgb = bytes(int(value) for value in circle["rgb"])
    squares = [
        (
            int(item["x0"]),
            int(item["y0"]),
            int(item["x1"]),
            int(item["y1"]),
            bytes(int(value) for value in item["rgb"]),
        )
        for item in spec["squares"]
    ]

    pixels = bytearray(background * (width * height))
    for y in range(height):
        row_base = y * width * 3
        in_border_row = y < border_thickness or y >= height - border_thickness
        delta_y_squared = (y - center_y) ** 2
        for x in range(width):
            offset = row_base + x * 3
            if in_border_row or x < border_thickness or x >= width - border_thickness:
                pixels[offset : offset + 3] = border_rgb
                continue
            if (x - center_x) ** 2 + delta_y_squared <= radius_squared:
                pixels[offset : offset + 3] = circle_rgb
                continue
            for x0, y0, x1, y1, rgb in squares:
                if x0 <= x < x1 and y0 <= y < y1:
                    pixels[offset : offset + 3] = rgb
                    break
    return bytes(pixels)


def encode_png(width: int, height: int, rgb_pixels: bytes) -> bytes:
    """把 RGB 像素编码为 8-bit truecolor PNG。"""

    if len(rgb_pixels) != width * height * 3:
        raise ValueError("像素长度与声明尺寸不一致")
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)  # PNG filter type 0：无过滤，保证可重建
        raw.extend(rgb_pixels[y * stride : (y + 1) * stride])

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", ihdr),
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            chunk(b"IEND", b""),
        )
    )


def ensure_fixed_image(
    path: str | Path,
    *,
    spec: dict[str, Any] | None = None,
    verify_expected_payload: bool = True,
) -> dict[str, Any]:
    """确保固定图片存在且内容与 spec 一致，返回可写入 manifest 的证据。"""

    spec = spec if spec is not None else FIXED_IMAGE_SPEC
    pixels = render_rgb_pixels(spec)
    payload_sha256 = hashlib.sha256(pixels).hexdigest()
    if verify_expected_payload and payload_sha256 != EXPECTED_PIXEL_PAYLOAD_SHA256:
        raise ValueError(
            "固定图片像素内容哈希与冻结值不一致："
            f"{payload_sha256} != {EXPECTED_PIXEL_PAYLOAD_SHA256}"
        )
    encoded = encode_png(int(spec["width"]), int(spec["height"]), pixels)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_bytes() if target.is_file() else None
    regenerated = existing != encoded
    if regenerated:
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(target)
    return {
        "source": "generated_from_fixed_spec",
        "spec_id": str(spec["spec_id"]),
        "spec_sha256": spec_sha256(spec),
        "width": int(spec["width"]),
        "height": int(spec["height"]),
        "path": str(target),
        "file_size_bytes": len(encoded),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "pixel_payload_sha256": payload_sha256,
        "pixel_payload_bytes": len(pixels),
        "regenerated": regenerated,
        "description_zh": str(spec["description_zh"]),
    }
