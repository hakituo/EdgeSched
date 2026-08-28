"""
实验配置模块

定义实验配置和结果的数据结构
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json
import time


@dataclass
class ExperimentConfig:
    """实验配置基类"""
    
    name: str
    seed: int = 42
    timeout: float = 300.0
    warmup_runs: int = 3
    repeat_count: int = 5
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(**data)


@dataclass
class ExperimentResult:
    """实验结果基类"""
    
    config: ExperimentConfig
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_data: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    environment: Dict[str, str] = field(default_factory=dict)
    status: str = "completed"
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "raw_data": self.raw_data,
            "environment": self.environment,
            "status": self.status,
            "error_message": self.error_message,
        }
    
    def save(self, filepath: str) -> None:
        """保存结果到 JSON 文件"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> "ExperimentResult":
        """从 JSON 文件加载结果"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = ExperimentConfig.from_dict(data["config"])
        return cls(
            config=config,
            metrics=data.get("metrics", {}),
            raw_data=data.get("raw_data", []),
            timestamp=data.get("timestamp", ""),
            environment=data.get("environment", {}),
            status=data.get("status", "unknown"),
            error_message=data.get("error_message"),
        )


@dataclass
class IsolationMetrics:
    """隔离效果指标"""
    
    ttft_p99_degradation_percent: float = 0.0
    total_latency_p99_degradation_percent: float = 0.0
    tail_latency_ratio: float = 1.0
    oom_errors: int = 0
    rejection_rate: float = 0.0
    
    TTFT_DEGRADATION_THRESHOLD: float = field(default=20.0, repr=False)
    TAIL_LATENCY_RATIO_THRESHOLD: float = field(default=3.0, repr=False)
    
    @property
    def is_good(self) -> bool:
        """判断隔离效果是否良好"""
        return (
            self.ttft_p99_degradation_percent < self.TTFT_DEGRADATION_THRESHOLD
            and self.tail_latency_ratio < self.TAIL_LATENCY_RATIO_THRESHOLD
            and self.oom_errors == 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_p99_degradation_percent": self.ttft_p99_degradation_percent,
            "total_latency_p99_degradation_percent": self.total_latency_p99_degradation_percent,
            "tail_latency_ratio": self.tail_latency_ratio,
            "oom_errors": self.oom_errors,
            "rejection_rate": self.rejection_rate,
            "is_good": self.is_good,
        }


@dataclass
class RecoveryMetrics:
    """崩溃恢复指标"""
    
    process_start_to_healthy_ms: float = 0.0
    healthy_to_first_request_ms: float = 0.0
    total_recovery_time_ms: float = 0.0
    stuck_running_found_count: int = 0
    stuck_running_closed_count: int = 0
    state_consistency_valid: bool = True
    
    RECOVERY_TIME_THRESHOLD_MS: float = field(default=2000.0, repr=False)
    
    @property
    def is_good(self) -> bool:
        """判断恢复效果是否良好"""
        return (
            self.total_recovery_time_ms < self.RECOVERY_TIME_THRESHOLD_MS
            and self.state_consistency_valid
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_start_to_healthy_ms": self.process_start_to_healthy_ms,
            "healthy_to_first_request_ms": self.healthy_to_first_request_ms,
            "total_recovery_time_ms": self.total_recovery_time_ms,
            "stuck_running_found_count": self.stuck_running_found_count,
            "stuck_running_closed_count": self.stuck_running_closed_count,
            "state_consistency_valid": self.state_consistency_valid,
            "is_good": self.is_good,
        }


@dataclass
class CancellationMetrics:
    """取消语义指标"""
    
    cancellation_latency_avg_ms: float = 0.0
    cancellation_latency_p95_ms: float = 0.0
    cancellation_latency_p99_ms: float = 0.0
    no_output_after_cancel_rate: float = 0.0
    late_result_discarded: bool = True
    
    CANCELLATION_LATENCY_THRESHOLD_MS: float = field(default=100.0, repr=False)
    NO_OUTPUT_THRESHOLD: float = field(default=0.95, repr=False)
    
    @property
    def is_good(self) -> bool:
        """判断取消语义是否良好"""
        return (
            self.cancellation_latency_p99_ms < self.CANCELLATION_LATENCY_THRESHOLD_MS
            and self.no_output_after_cancel_rate >= self.NO_OUTPUT_THRESHOLD
            and self.late_result_discarded
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cancellation_latency_avg_ms": self.cancellation_latency_avg_ms,
            "cancellation_latency_p95_ms": self.cancellation_latency_p95_ms,
            "cancellation_latency_p99_ms": self.cancellation_latency_p99_ms,
            "no_output_after_cancel_rate": self.no_output_after_cancel_rate,
            "late_result_discarded": self.late_result_discarded,
            "is_good": self.is_good,
        }
