"""多模态端点门禁：任何真实测量之前必须通过的 artifact 与能力核验。

门禁只做静态解析和 mmproj 能力位读取，不加载语言模型、不占用显存。
不通过时禁止进入 K0/FULL 端点 smoke，也不得把结果写成视觉证据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.elastic_residency.vision.fixed_image import ensure_fixed_image
from experiments.elastic_residency.vision.vision_artifacts import (
    artifact_manifest_entry,
    describe_gguf_artifact,
    expected_visual_tokens,
    projector_profile,
    verify_artifact_pair,
)
from experiments.elastic_residency.vision.vision_evidence import image_file_evidence


def build_endpoint_gate(
    *,
    model_path: str | Path,
    mmproj_path: str | Path,
    fixed_image_path: str | Path,
    external_image_path: str | Path | None = None,
    expected_block_count: int | None = None,
    required_context_tokens: int | None = None,
    declared: dict[str, Any] | None = None,
    include_mtmd_caps: bool = True,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """核验主 GGUF、mmproj 与固定图片，并给出视觉 token 期望值。"""

    main_artifact = describe_gguf_artifact(model_path, compute_sha256=compute_sha256)
    projector_artifact = describe_gguf_artifact(
        mmproj_path, compute_sha256=compute_sha256
    )
    verification = verify_artifact_pair(
        main_artifact=main_artifact,
        projector_artifact=projector_artifact,
        expected_block_count=expected_block_count,
        required_context_tokens=required_context_tokens,
        declared=declared,
    )
    checks = dict(verification["checks"])
    if external_image_path:
        image = image_file_evidence(external_image_path)
    else:
        image = ensure_fixed_image(fixed_image_path)
    checks["image_available"] = bool(Path(image["path"]).is_file())
    checks["image_hash_recorded"] = bool(image["file_sha256"])
    mmproj_caps: dict[str, Any] | None = None
    if include_mtmd_caps:
        from experiments.elastic_residency.vision.mtmd_runtime import (  # noqa: PLC0415
            probe_mmproj_caps,
        )

        mmproj_caps = probe_mmproj_caps(mmproj_path)
        checks["mmproj_declares_vision_input"] = bool(
            mmproj_caps["declares_vision_input"]
        )
    projector = projector_profile(projector_artifact)
    tokens_expectation = (
        expected_visual_tokens(
            projector,
            width=int(image["width"]),
            height=int(image["height"]),
        )
        if image.get("width") and image.get("height")
        else None
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "artifacts": {
            "main_model": artifact_manifest_entry(main_artifact),
            "projector": artifact_manifest_entry(projector_artifact),
            "image": image,
        },
        "main_model_profile": verification["main_model"],
        "projector_profile": projector,
        "mmproj_caps": mmproj_caps,
        "expected_visual_tokens": tokens_expectation,
        "declared": verification["declared"],
    }
