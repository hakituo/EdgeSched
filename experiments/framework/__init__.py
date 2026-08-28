"""
EdgeSched 实验框架核心模块

提供统一的实验基础设施，包括：
- 路径管理
- 配置管理
- 统计分析
- 可重复性保障
- 结果存储
- 导入辅助
"""

from .paths import PROJECT_ROOT, MVP_CORE_DIR, RESULTS_DIR, setup_paths, get_output_path
from .config import ExperimentConfig, ExperimentResult
from .statistics import StatisticsAnalyzer, report_with_confidence
from .reproducibility import ReproducibilityManager
from .mock_tasks import MockGPUTask, MockCPUTask, MockIOTask, MockLLMAdapter, MockImageAdapter, MockTTSAdapter
try:
    from .real_adapters import Qwen3TTSAdapter
except Exception:
    Qwen3TTSAdapter = None
from .base import BaseExperiment
from .import_helper import ensure_importable, import_service, import_config, import_from_service

__all__ = [
    "PROJECT_ROOT",
    "MVP_CORE_DIR", 
    "RESULTS_DIR",
    "setup_paths",
    "get_output_path",
    "ExperimentConfig",
    "ExperimentResult",
    "StatisticsAnalyzer",
    "report_with_confidence",
    "ReproducibilityManager",
    "MockGPUTask",
    "MockCPUTask",
    "MockIOTask",
    "MockLLMAdapter",
    "MockImageAdapter",
    "MockTTSAdapter",
    "Qwen3TTSAdapter",
    "BaseExperiment",
    "ensure_importable",
    "import_service",
    "import_config",
    "import_from_service",
]
