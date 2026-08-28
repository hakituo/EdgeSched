import json
import time

import pytest

from experiments.elastic_residency.contracts import (
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentContractError,
    ExperimentEvent,
    ExperimentRunConfig,
    FailureRecord,
)


def test_artifact_writer_uses_non_overwriting_standard_layout(tmp_path) -> None:
    config = ExperimentRunConfig(
        experiment_name="contract",
        run_id="contract_run",
        backends=(BackendEvidence("llm", "test_fake", False),),
    )
    writer = ExperimentArtifactWriter(tmp_path, config.run_id)
    writer.initialize(config, {"environment": "test"})
    writer.append_event(
        ExperimentEvent("start", time.monotonic_ns(), "2026-08-28T00:00:00Z")
    )
    writer.append_failure(
        FailureRecord("smoke", "Expected", "test", time.monotonic_ns())
    )
    writer.write_summary({"passed": True})
    writer.write_summary_csv(
        {"run_id": config.run_id, "passed": True, "affinity": [0, 2]}
    )

    assert json.loads(writer.layout.run_config.read_text())["run_id"] == "contract_run"
    assert json.loads(writer.layout.manifest.read_text())["run_id"] == "contract_run"
    assert json.loads(writer.layout.events.read_text())["name"] == "start"
    assert json.loads(writer.layout.failures.read_text())["error_type"] == "Expected"
    assert json.loads(writer.layout.summary.read_text())["passed"] is True
    assert writer.layout.iterations.is_file()
    assert writer.layout.timeline.is_file()
    assert writer.layout.summary_csv.is_file()
    assert writer.layout.summary_csv.read_text().splitlines() == [
        "run_id,passed,affinity",
        'contract_run,True,"[0,2]"',
    ]
    with pytest.raises(FileExistsError):
        ExperimentArtifactWriter(tmp_path, config.run_id)


def test_contract_rejects_unsafe_run_ids_and_binary_json(tmp_path) -> None:
    with pytest.raises(ExperimentContractError):
        ExperimentRunConfig("bad", "../escape")
    config = ExperimentRunConfig("binary", "binary_run")
    writer = ExperimentArtifactWriter(tmp_path, config.run_id)
    writer.initialize(config, {})
    with pytest.raises(ExperimentContractError):
        writer.append_iteration({"audio": b"not allowed"})
