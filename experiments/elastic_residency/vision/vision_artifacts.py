"""Qwen3.5 主 GGUF 与 mmproj 的 artifact 画像、哈希与对应关系核验。

本模块只做不加载模型的静态核验，用于在任何 GPU 运行之前回答：
两个文件是否互相匹配、层数与架构是否与论文口径一致、声明的大小/哈希是否成立。

对应关系判据（均来自两个 GGUF 的真实 metadata，不是文件名推断）：

1. mmproj 的 ``general.architecture`` 为 ``clip`` 且 ``general.type`` 为 ``mmproj``；
2. mmproj 声明 ``clip.has_vision_encoder``；
3. mmproj 的 ``clip.vision.projection_dim`` 等于主模型 ``<arch>.embedding_length``；
4. 两者的 ``general.basename`` 与 ``general.base_model.0.*`` 一致；缺少 basename
   时要求非空名称、基础模型名称和来源 URL 匹配。单侧缺失允许名称的空格/连字符差异，
   双侧缺失保持精确名称匹配；两侧 basename 均存在但矛盾时仍拒绝。

只有第 3 条是结构性硬约束（投影输出维度必须等于语言模型隐藏维度），
其余为来源一致性证据。任何一条不成立都不得继续做真实视觉测量。
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from experiments.elastic_residency.vision.gguf_reader import (
    metadata_bool,
    metadata_int,
    metadata_str,
    read_gguf_header,
    scalar_metadata,
)
from experiments.tts.profiling_utils import sha256_file


MAIN_MODEL_EXPECTED_ARCHITECTURE = "qwen35"
PROJECTOR_EXPECTED_ARCHITECTURE = "clip"
PROJECTOR_EXPECTED_GENERAL_TYPE = "mmproj"


def describe_gguf_artifact(
    path: str | Path,
    *,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """读取单个 GGUF 的头部画像；``compute_sha256`` 为 False 时只做静态测试。"""

    header = read_gguf_header(path)
    resolved = Path(header["path"])
    return {
        "path": str(resolved),
        "file_name": resolved.name,
        "file_size_bytes": header["file_size_bytes"],
        "sha256": sha256_file(resolved) if compute_sha256 else None,
        "sha256_computed": bool(compute_sha256),
        "gguf_version": header["gguf_version"],
        "tensor_count": header["tensor_count"],
        "metadata_kv_count": header["metadata_kv_count"],
        "header_bytes": header["header_bytes"],
        "metadata": scalar_metadata(header),
        "_header": header,
    }


def main_model_profile(artifact: dict[str, Any]) -> dict[str, Any]:
    """从主 GGUF 头部提取架构、层数、上下文长度与隐藏维度。"""

    header = artifact["_header"]
    architecture = metadata_str(header, "general.architecture")
    if not architecture:
        raise ValueError("主 GGUF 缺少 general.architecture，无法核验架构")
    return {
        "architecture": architecture,
        "block_count": metadata_int(header, f"{architecture}.block_count"),
        "block_count_key": f"{architecture}.block_count",
        "context_length": metadata_int(header, f"{architecture}.context_length"),
        "embedding_length": metadata_int(header, f"{architecture}.embedding_length"),
        "attention_head_count": metadata_int(
            header, f"{architecture}.attention.head_count"
        ),
        "attention_head_count_kv": metadata_int(
            header, f"{architecture}.attention.head_count_kv"
        ),
        "general_name": metadata_str(header, "general.name"),
        "general_basename": metadata_str(header, "general.basename"),
        "general_type": metadata_str(header, "general.type"),
        "general_file_type": metadata_int(header, "general.file_type"),
        "base_model_name": metadata_str(header, "general.base_model.0.name"),
        "base_model_repo_url": metadata_str(header, "general.base_model.0.repo_url"),
        "quantized_by": metadata_str(header, "general.quantized_by"),
        "has_chat_template": bool(metadata_str(header, "tokenizer.chat_template")),
        # M-RoPE 分段存在说明该语言模型带多模态位置编码，是视觉链路的辅助证据。
        "rope_dimension_sections_present": (
            f"{architecture}.rope.dimension_sections" in header["metadata"]
        ),
    }


def projector_profile(artifact: dict[str, Any]) -> dict[str, Any]:
    """从 mmproj 头部提取视觉编码器结构与投影维度。"""

    header = artifact["_header"]
    return {
        "architecture": metadata_str(header, "general.architecture"),
        "general_type": metadata_str(header, "general.type"),
        "general_name": metadata_str(header, "general.name"),
        "general_basename": metadata_str(header, "general.basename"),
        "general_file_type": metadata_int(header, "general.file_type"),
        "base_model_name": metadata_str(header, "general.base_model.0.name"),
        "base_model_repo_url": metadata_str(header, "general.base_model.0.repo_url"),
        "projector_type": metadata_str(header, "clip.projector_type"),
        "has_vision_encoder": metadata_bool(header, "clip.has_vision_encoder"),
        "has_audio_encoder": metadata_bool(header, "clip.has_audio_encoder"),
        "vision_block_count": metadata_int(header, "clip.vision.block_count"),
        "vision_embedding_length": metadata_int(
            header, "clip.vision.embedding_length"
        ),
        "vision_image_size": metadata_int(header, "clip.vision.image_size"),
        "vision_patch_size": metadata_int(header, "clip.vision.patch_size"),
        "vision_spatial_merge_size": metadata_int(
            header, "clip.vision.spatial_merge_size"
        ),
        "vision_projection_dim": metadata_int(header, "clip.vision.projection_dim"),
    }


def expected_visual_tokens(
    projector: dict[str, Any],
    *,
    width: int,
    height: int,
) -> int | None:
    """按 patch 与 spatial merge 推算视觉 token 数的期望值。

    该值只是设计期望，用于与 libmtmd 的真实统计交叉核对；
    真实 token 数一律以 ``mtmd_image_tokens_get_n_tokens`` 为准。
    """

    patch = projector.get("vision_patch_size")
    merge = projector.get("vision_spatial_merge_size")
    if not patch or not merge or width <= 0 or height <= 0:
        return None
    stride = patch * merge
    if width % stride or height % stride:
        return None
    return (width // stride) * (height // stride)


def verify_artifact_pair(
    *,
    main_artifact: dict[str, Any],
    projector_artifact: dict[str, Any],
    expected_block_count: int | None = None,
    required_context_tokens: int | None = None,
    declared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """核验主模型与 projector 的对应关系及声明的大小/哈希。"""

    main = main_model_profile(main_artifact)
    projector = projector_profile(projector_artifact)
    declared = dict(declared or {})
    checks: dict[str, bool] = {
        "main_model_architecture_matches": (
            main["architecture"] == MAIN_MODEL_EXPECTED_ARCHITECTURE
        ),
        "main_model_block_count_available": main["block_count"] is not None,
        "main_model_embedding_length_available": (
            main["embedding_length"] is not None
        ),
        "main_model_has_chat_template": main["has_chat_template"],
        "projector_architecture_is_clip": (
            projector["architecture"] == PROJECTOR_EXPECTED_ARCHITECTURE
        ),
        "projector_general_type_is_mmproj": (
            projector["general_type"] == PROJECTOR_EXPECTED_GENERAL_TYPE
        ),
        "projector_declares_vision_encoder": (
            projector["has_vision_encoder"] is True
        ),
        "projector_projection_dim_matches_llm_embedding": (
            projector["vision_projection_dim"] is not None
            and projector["vision_projection_dim"] == main["embedding_length"]
        ),
        "projector_basename_matches_main_model": (
            bool(projector["general_basename"])
            and projector["general_basename"] == main["general_basename"]
        ),
        "projector_base_model_matches_main_model": (
            bool(projector["base_model_name"])
            and projector["base_model_name"] == main["base_model_name"]
            and projector["base_model_repo_url"] == main["base_model_repo_url"]
        ),
    }
    if main["general_basename"] is None and projector["general_basename"] is None:
        # 某些量化发布物省略 basename；使用独立的具名门禁记录来源核验分支。
        del checks["projector_basename_matches_main_model"]
        checks["projector_name_and_source_match_without_basename"] = bool(
            main["general_name"] and main["general_name"] == projector["general_name"]
            and main["base_model_name"]
            and main["base_model_name"] == projector["base_model_name"]
            and main["base_model_repo_url"]
            and main["base_model_repo_url"] == projector["base_model_repo_url"]
        )
    elif main["general_basename"] is None or projector["general_basename"] is None:
        # 不同量化发布物可能只有一侧提供 basename；只归一化名称分隔符，不模糊匹配型号。
        del checks["projector_basename_matches_main_model"]
        main_name = re.sub(r"[\s-]+", "", main["general_name"] or "")
        projector_name = re.sub(r"[\s-]+", "", projector["general_name"] or "")
        checks["projector_normalized_name_and_source_match_with_partial_basename"] = bool(
            main_name and main_name == projector_name
            and main["base_model_name"]
            and main["base_model_name"] == projector["base_model_name"]
            and main["base_model_repo_url"]
            and main["base_model_repo_url"] == projector["base_model_repo_url"]
        )
    if expected_block_count is not None:
        checks["main_model_block_count_matches_expected"] = (
            main["block_count"] == expected_block_count
        )
    if required_context_tokens is not None:
        checks["main_model_context_length_covers_run"] = (
            main["context_length"] is not None
            and main["context_length"] >= required_context_tokens
        )
    for role, artifact in (
        ("main_model", main_artifact),
        ("projector", projector_artifact),
    ):
        declared_sha = declared.get(f"{role}_sha256")
        if declared_sha:
            checks[f"{role}_sha256_matches_declared"] = (
                artifact["sha256"] is not None
                and artifact["sha256"] == str(declared_sha).lower()
            )
        declared_size = declared.get(f"{role}_size_bytes")
        if declared_size is not None:
            checks[f"{role}_size_matches_declared"] = (
                artifact["file_size_bytes"] == int(declared_size)
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "main_model": main,
        "projector": projector,
        "declared": declared,
    }


def artifact_manifest_entry(artifact: dict[str, Any]) -> dict[str, Any]:
    """生成可写入 manifest 的 artifact 记录（剔除内部头部对象）。"""

    return {
        key: value for key, value in artifact.items() if not key.startswith("_")
    }
