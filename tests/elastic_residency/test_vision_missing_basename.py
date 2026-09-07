"""缺少 basename 的量化文件仍须通过结构与明确来源核验。"""

import pytest

from experiments.elastic_residency.vision.vision_artifacts import (
    describe_gguf_artifact, verify_artifact_pair,
)
from tests.elastic_residency.vision_payload_builders import (
    main_model_metadata, mmproj_metadata, write_gguf,
)


@pytest.mark.parametrize("fault", [None, "name", "source", "missing_source", "dimension",
                                  "one_basename_only"])
def test_missing_basename_requires_explicit_consistent_identity(tmp_path, fault):
    main = main_model_metadata(basename="Qwen3.5-9B", embedding_length=4096)
    projector = mmproj_metadata(basename="Qwen3.5-9B", projection_dim=4096)
    for data in (main, projector):
        data.pop("general.basename")
        kind, _ = data["general.base_model.0.name"]
        data["general.base_model.0.name"] = (kind, "Qwen3.5 9B Base")
        data["general.base_model.0.repo_url"] = (kind, "https://huggingface.co/Qwen/Qwen3.5-9B-Base")
    if fault == "name":
        projector["general.name"] = (kind, "Different Model")
    elif fault == "source":
        projector["general.base_model.0.repo_url"] = (kind, "https://example.invalid/other")
    elif fault == "missing_source":
        main.pop("general.base_model.0.repo_url")
        projector.pop("general.base_model.0.repo_url")
    elif fault == "dimension":
        number_kind, _ = projector["clip.vision.projection_dim"]
        projector["clip.vision.projection_dim"] = (number_kind, 2560)
    elif fault == "one_basename_only":
        main["general.basename"] = (kind, "Qwen3.5-9B")
    result = verify_artifact_pair(
        main_artifact=describe_gguf_artifact(write_gguf(tmp_path / "main.gguf", main)),
        projector_artifact=describe_gguf_artifact(write_gguf(tmp_path / "vision.gguf", projector)),
        expected_block_count=32, required_context_tokens=640)
    assert result["passed"] is (fault in (None, "one_basename_only"))
    if fault is None:
        assert result["checks"]["projector_name_and_source_match_without_basename"]


@pytest.mark.parametrize("fault", [None, "name", "source", "missing_source", "dimension",
                                  "hash", "conflicting_basenames", "base_name"])
def test_partial_basename_accepts_only_matching_name_source_and_structure(tmp_path, fault):
    main = main_model_metadata(basename="Qwen3.5", embedding_length=4096)
    projector = mmproj_metadata(basename="Qwen3.5-9B", projection_dim=4096)
    projector.pop("general.basename")
    kind, _ = main["general.name"]
    main["general.name"] = (kind, "Qwen3.5 9B")
    projector["general.name"] = (kind, "Qwen3.5-9B")
    for data in (main, projector):
        data["general.base_model.0.name"] = (kind, "Qwen3.5 9B Base")
        data["general.base_model.0.repo_url"] = (kind, "https://huggingface.co/Qwen/Qwen3.5-9B-Base")
    if fault == "name":
        projector["general.name"] = (kind, "Qwen3.5-4B")
    elif fault == "source":
        projector["general.base_model.0.repo_url"] = (kind, "https://example.invalid/other")
    elif fault == "missing_source":
        for data in (main, projector):
            data.pop("general.base_model.0.repo_url")
    elif fault == "base_name":
        projector["general.base_model.0.name"] = (kind, "Different Base")
    elif fault == "dimension":
        number_kind, _ = projector["clip.vision.projection_dim"]
        projector["clip.vision.projection_dim"] = (number_kind, 2560)
    elif fault == "conflicting_basenames":
        projector["general.basename"] = (kind, "Qwen3.5-4B")
    result = verify_artifact_pair(
        main_artifact=describe_gguf_artifact(write_gguf(tmp_path / "main.gguf", main)),
        projector_artifact=describe_gguf_artifact(write_gguf(tmp_path / "vision.gguf", projector)),
        declared={"main_model_sha256": "0" * 64} if fault == "hash" else None,
        expected_block_count=32, required_context_tokens=1536)
    assert result["passed"] is (fault is None)
    if fault is None:
        assert result["checks"]["projector_normalized_name_and_source_match_with_partial_basename"]
