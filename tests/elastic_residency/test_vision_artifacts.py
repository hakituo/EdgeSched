"""GGUF 头部解析、artifact 对应关系与固定图片的静态测试。

只验证链路与判据，不产生任何真实模型或性能结论。
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.vision.fixed_image import (  # noqa: E402
    EXPECTED_PIXEL_PAYLOAD_SHA256,
    FIXED_IMAGE_SPEC,
    encode_png,
    ensure_fixed_image,
    render_rgb_pixels,
)
from experiments.elastic_residency.vision.gguf_reader import (  # noqa: E402
    GGUFParseError,
    metadata_bool,
    metadata_int,
    metadata_str,
    read_gguf_header,
    scalar_metadata,
)
from experiments.elastic_residency.vision.vision_artifacts import (  # noqa: E402
    describe_gguf_artifact,
    expected_visual_tokens,
    main_model_profile,
    projector_profile,
    verify_artifact_pair,
)
from experiments.elastic_residency.vision.vision_endpoint_gate import (  # noqa: E402
    build_endpoint_gate,
)
from tests.elastic_residency.vision_payload_builders import (  # noqa: E402
    build_gguf_bytes,
    main_model_metadata,
    write_artifact_pair,
)


class TestGGUFReader:
    def test_parses_scalars_strings_and_arrays(self, tmp_path: Path) -> None:
        path = tmp_path / "main.gguf"
        path.write_bytes(build_gguf_bytes(main_model_metadata(), tensor_count=426))
        header = read_gguf_header(path)
        assert header["gguf_version"] == 3
        assert header["tensor_count"] == 426
        assert metadata_str(header, "general.architecture") == "qwen35"
        assert metadata_int(header, "qwen35.block_count") == 32
        assert header["metadata"]["tokenizer.ggml.tokens"]["count"] == 3
        assert header["header_bytes"] == path.stat().st_size

    def test_scalar_metadata_folds_arrays(self, tmp_path: Path) -> None:
        path = tmp_path / "main.gguf"
        path.write_bytes(build_gguf_bytes(main_model_metadata()))
        folded = scalar_metadata(read_gguf_header(path))
        assert folded["tokenizer.ggml.tokens"] == {
            "array_element_type": "string",
            "array_count": 3,
        }

    def test_rejects_non_gguf_and_unsupported_version(self, tmp_path: Path) -> None:
        bad_magic = tmp_path / "bad.gguf"
        bad_magic.write_bytes(b"NOPE" + b"\x00" * 32)
        with pytest.raises(GGUFParseError):
            read_gguf_header(bad_magic)
        bad_version = tmp_path / "v1.gguf"
        bad_version.write_bytes(
            b"GGUF" + struct.pack("<I", 1) + struct.pack("<Q", 0) * 2
        )
        with pytest.raises(GGUFParseError):
            read_gguf_header(bad_version)

    def test_truncated_header_fails(self, tmp_path: Path) -> None:
        payload = build_gguf_bytes(main_model_metadata())
        truncated = tmp_path / "cut.gguf"
        truncated.write_bytes(payload[: len(payload) // 2])
        with pytest.raises(GGUFParseError):
            read_gguf_header(truncated)

    def test_typed_getters_return_none_on_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "main.gguf"
        path.write_bytes(build_gguf_bytes(main_model_metadata()))
        header = read_gguf_header(path)
        assert metadata_int(header, "general.architecture") is None
        assert metadata_str(header, "qwen35.block_count") is None
        assert metadata_bool(header, "qwen35.block_count") is None
        assert metadata_int(header, "missing.key") is None


class TestArtifactPair:
    def test_matching_pair_passes_all_checks(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(tmp_path)
        main = describe_gguf_artifact(main_path)
        projector = describe_gguf_artifact(mmproj_path)
        result = verify_artifact_pair(
            main_artifact=main,
            projector_artifact=projector,
            expected_block_count=32,
            required_context_tokens=640,
            declared={
                "main_model_sha256": main["sha256"],
                "main_model_size_bytes": main["file_size_bytes"],
                "projector_sha256": projector["sha256"],
                "projector_size_bytes": projector["file_size_bytes"],
            },
        )
        assert result["passed"] is True
        assert result["checks"]["projector_projection_dim_matches_llm_embedding"]
        assert result["checks"]["main_model_sha256_matches_declared"]
        assert main_model_profile(main)["rope_dimension_sections_present"] is True

    def test_projection_dim_mismatch_fails(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(
            tmp_path, mmproj_kwargs={"projection_dim": 4096}
        )
        result = verify_artifact_pair(
            main_artifact=describe_gguf_artifact(main_path),
            projector_artifact=describe_gguf_artifact(mmproj_path),
        )
        assert result["passed"] is False
        assert not result["checks"]["projector_projection_dim_matches_llm_embedding"]

    def test_missing_vision_encoder_and_wrong_family_fail(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(
            tmp_path,
            mmproj_kwargs={"has_vision_encoder": False, "basename": "Qwen3.5-9B"},
        )
        checks = verify_artifact_pair(
            main_artifact=describe_gguf_artifact(main_path),
            projector_artifact=describe_gguf_artifact(mmproj_path),
        )["checks"]
        assert not checks["projector_declares_vision_encoder"]
        assert not checks["projector_basename_matches_main_model"]

    def test_declared_hash_mismatch_fails(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(tmp_path)
        checks = verify_artifact_pair(
            main_artifact=describe_gguf_artifact(main_path),
            projector_artifact=describe_gguf_artifact(mmproj_path),
            declared={"main_model_sha256": "0" * 64, "projector_size_bytes": 1},
        )["checks"]
        assert not checks["main_model_sha256_matches_declared"]
        assert not checks["projector_size_matches_declared"]

    def test_context_length_gate_and_block_count(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(
            tmp_path, main_kwargs={"context_length": 256, "block_count": 28}
        )
        checks = verify_artifact_pair(
            main_artifact=describe_gguf_artifact(main_path),
            projector_artifact=describe_gguf_artifact(mmproj_path),
            expected_block_count=32,
            required_context_tokens=640,
        )["checks"]
        assert not checks["main_model_context_length_covers_run"]
        assert not checks["main_model_block_count_matches_expected"]

    def test_expected_visual_tokens_uses_patch_and_merge(self, tmp_path: Path) -> None:
        _, mmproj_path = write_artifact_pair(tmp_path)
        projector = projector_profile(describe_gguf_artifact(mmproj_path))
        assert expected_visual_tokens(projector, width=448, height=448) == 196
        assert expected_visual_tokens(projector, width=450, height=448) is None

    def test_missing_chat_template_fails(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(
            tmp_path, main_kwargs={"with_chat_template": False}
        )
        checks = verify_artifact_pair(
            main_artifact=describe_gguf_artifact(main_path),
            projector_artifact=describe_gguf_artifact(mmproj_path),
        )["checks"]
        assert not checks["main_model_has_chat_template"]


class TestFixedImage:
    def test_pixel_payload_hash_is_frozen(self) -> None:
        pixels = render_rgb_pixels()
        assert len(pixels) == FIXED_IMAGE_SPEC["width"] * FIXED_IMAGE_SPEC["height"] * 3
        assert hashlib.sha256(pixels).hexdigest() == EXPECTED_PIXEL_PAYLOAD_SHA256

    def test_png_structure_and_idempotent_write(self, tmp_path: Path) -> None:
        target = tmp_path / "fixed.png"
        first = ensure_fixed_image(target)
        assert first["regenerated"] is True
        assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        second = ensure_fixed_image(target)
        assert second["regenerated"] is False
        assert second["file_sha256"] == first["file_sha256"]
        assert second["pixel_payload_sha256"] == EXPECTED_PIXEL_PAYLOAD_SHA256

    def test_spec_drift_is_rejected(self, tmp_path: Path) -> None:
        drifted = dict(FIXED_IMAGE_SPEC)
        drifted["background_rgb"] = [0, 0, 0]
        with pytest.raises(ValueError):
            ensure_fixed_image(tmp_path / "drift.png", spec=drifted)

    def test_encode_png_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            encode_png(2, 2, b"\x00" * 3)


class TestEndpointGate:
    def test_gate_passes_without_mtmd_and_reports_expectation(
        self, tmp_path: Path
    ) -> None:
        main_path, mmproj_path = write_artifact_pair(tmp_path)
        gate = build_endpoint_gate(
            model_path=main_path,
            mmproj_path=mmproj_path,
            fixed_image_path=tmp_path / "fixed.png",
            expected_block_count=32,
            required_context_tokens=640,
            include_mtmd_caps=False,
        )
        assert gate["passed"] is True
        assert gate["expected_visual_tokens"] == 196
        assert gate["mmproj_caps"] is None
        assert gate["artifacts"]["image"]["source"] == "generated_from_fixed_spec"
        assert gate["checks"]["image_available"] is True

    def test_gate_fails_and_lists_failed_checks(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(
            tmp_path, mmproj_kwargs={"has_vision_encoder": False}
        )
        gate = build_endpoint_gate(
            model_path=main_path,
            mmproj_path=mmproj_path,
            fixed_image_path=tmp_path / "fixed.png",
            include_mtmd_caps=False,
        )
        assert gate["passed"] is False
        assert not gate["checks"]["projector_declares_vision_encoder"]

    def test_gate_accepts_external_image(self, tmp_path: Path) -> None:
        main_path, mmproj_path = write_artifact_pair(tmp_path)
        external = tmp_path / "external.png"
        external.write_bytes(
            encode_png(32, 32, bytes(bytearray([7]) * (32 * 32 * 3)))
        )
        gate = build_endpoint_gate(
            model_path=main_path,
            mmproj_path=mmproj_path,
            fixed_image_path=tmp_path / "unused.png",
            external_image_path=external,
            include_mtmd_caps=False,
        )
        assert gate["artifacts"]["image"]["source"] == "external_file"
        assert gate["artifacts"]["image"]["pixel_payload_sha256"] is None
        assert gate["expected_visual_tokens"] is None
