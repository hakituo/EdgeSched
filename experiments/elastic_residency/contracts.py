"""弹性驻留实验统一配置、事件与产物契约。"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "edgesched.elastic_residency.v1"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ExperimentContractError(ValueError):
    """实验配置或结果不满足冻结契约。"""


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentContractError("实验 JSON 不允许 NaN 或 Infinity")
    if isinstance(value, bytes):
        raise ExperimentContractError("实验 JSON 不允许嵌入二进制内容")
    return value


@dataclass(frozen=True)
class BackendEvidence:
    """单个真实或 Mock 后端的证据。"""

    component: str
    backend_kind: str
    is_real: bool
    model_path: str = ""
    model_sha256: str = ""
    fallback_occurred: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component.strip() or not self.backend_kind.strip():
            raise ExperimentContractError("backend component/kind 不能为空")
        if self.model_sha256 and len(self.model_sha256) != 64:
            raise ExperimentContractError("model_sha256 必须为 64 位 SHA-256")
        if self.is_real and self.fallback_occurred:
            raise ExperimentContractError("真实后端证据不能同时标记 fallback")


@dataclass(frozen=True)
class ExperimentRunConfig:
    """所有弹性驻留实验共用的运行配置头。"""

    experiment_name: str
    run_id: str
    seed: int = 42
    sample_interval_s: float = 0.05
    gpu_sample_interval_s: float = 0.25
    backends: tuple[BackendEvidence, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ExperimentContractError("experiment_name 不能为空")
        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ExperimentContractError(f"非法 run_id: {self.run_id!r}")
        if self.sample_interval_s <= 0:
            raise ExperimentContractError("sample_interval_s 必须大于 0")
        if self.gpu_sample_interval_s < self.sample_interval_s:
            raise ExperimentContractError(
                "gpu_sample_interval_s 不能小于 sample_interval_s"
            )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class ExperimentEvent:
    name: str
    monotonic_ns: int
    wall_time_iso: str
    phase: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class FailureRecord:
    stage: str
    error_type: str
    message: str
    monotonic_ns: int
    fatal: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class TimelineSample:
    """50 ms 资源/调度时间线的单个采样点。"""

    monotonic_ns: int
    wall_time_iso: str
    scheduler_jitter_ms: float
    process_pid: int
    process_cpu_percent: float
    cpu_sample_valid: bool
    process_rss_bytes: int
    system_available_bytes: int
    system_swap_used_bytes: int
    per_cpu_percent: tuple[float, ...]
    cpu_affinity: tuple[int, ...]
    gpu_available: bool = False
    gpu_sample_age_ms: float = 0.0
    gpu_card_used_mb: float = 0.0
    gpu_card_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_power_w: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_process_used_mb: float = 0.0
    framework_gpu: Mapping[str, float] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path
    run_config: Path
    manifest: Path
    iterations: Path
    events: Path
    timeline: Path
    summary: Path
    summary_csv: Path
    failures: Path


class ExperimentArtifactWriter:
    """以不可覆盖的 run id 写入统一实验产物。"""

    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        *,
        allow_existing: bool = False,
    ) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ExperimentContractError(f"非法 run_id: {run_id!r}")
        run_root = Path(output_root).expanduser().resolve() / run_id
        if run_root.exists() and not allow_existing:
            raise FileExistsError(f"run id 已存在，拒绝覆盖: {run_root}")
        run_root.mkdir(parents=True, exist_ok=allow_existing)
        self.layout = ArtifactLayout(
            root=run_root,
            run_config=run_root / "run_config.json",
            manifest=run_root / "manifest.json",
            iterations=run_root / "iterations.jsonl",
            events=run_root / "events.jsonl",
            timeline=run_root / "timeline.jsonl",
            summary=run_root / "summary.json",
            summary_csv=run_root / "summary.csv",
            failures=run_root / "failures.jsonl",
        )
        self._lock = threading.Lock()

    def initialize(
        self,
        config: ExperimentRunConfig,
        manifest: Mapping[str, Any],
    ) -> None:
        if config.run_id != self.layout.root.name:
            raise ExperimentContractError("config.run_id 与产物目录不一致")
        self._write_json(self.layout.run_config, config.to_dict())
        manifest_data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": config.run_id,
            **_json_value(manifest),
        }
        self._write_json(self.layout.manifest, manifest_data)
        for path in (
            self.layout.iterations,
            self.layout.events,
            self.layout.timeline,
            self.layout.summary_csv,
            self.layout.failures,
        ):
            path.touch(exist_ok=True)

    def append_iteration(self, record: Any) -> None:
        self._append_jsonl(self.layout.iterations, record)

    def append_event(self, record: ExperimentEvent) -> None:
        self._append_jsonl(self.layout.events, record)

    def append_timeline(self, record: TimelineSample) -> None:
        self._append_jsonl(self.layout.timeline, record)

    def append_failure(self, record: FailureRecord) -> None:
        self._append_jsonl(self.layout.failures, record)

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self._write_json(
            self.layout.summary,
            {"schema_version": SCHEMA_VERSION, **_json_value(summary)},
        )

    def write_summary_csv(self, row: Mapping[str, Any]) -> None:
        """写单行机器可读汇总；嵌套值编码为紧凑 JSON。"""

        normalized = {
            str(key): self._csv_value(value) for key, value in row.items()
        }
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(normalized))
        writer.writeheader()
        writer.writerow(normalized)
        self._write_text(self.layout.summary_csv, buffer.getvalue())

    def _write_json(self, path: Path, payload: Any) -> None:
        encoded = json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        self._write_text(path, encoded + "\n")

    def _write_text(self, path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with self._lock:
            temporary.write_text(content, encoding="utf-8", newline="")
            temporary.replace(path)

    @staticmethod
    def _csv_value(value: Any) -> Any:
        normalized = _json_value(value)
        if isinstance(normalized, (dict, list)):
            return json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        return normalized

    def _append_jsonl(self, path: Path, payload: Any) -> None:
        encoded = json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with self._lock, path.open("a", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(encoded + "\n")
