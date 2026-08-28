"""
实验基类模块

提供统一的实验执行框架
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .config import ExperimentConfig, ExperimentResult
from .paths import setup_paths, get_output_path
from .statistics import StatisticsAnalyzer
from .reproducibility import ReproducibilityManager


class BaseExperiment(ABC):
    """实验基类"""
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化实验
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.reproducibility = ReproducibilityManager(seed=config.seed)
        self.analyzer = StatisticsAnalyzer()
        self.results: List[Dict[str, Any]] = []
    
    def setup_paths(self) -> None:
        """设置 Python 路径"""
        setup_paths()
    
    @abstractmethod
    async def setup(self) -> None:
        """实验初始化（子类实现）"""
        pass
    
    @abstractmethod
    async def run_single(self, **kwargs) -> Dict[str, Any]:
        """单次实验运行（子类实现）"""
        pass
    
    @abstractmethod
    async def teardown(self) -> None:
        """实验清理（子类实现）"""
        pass
    
    async def warmup(self, **kwargs) -> None:
        """预热"""
        for _ in range(self.config.warmup_runs):
            try:
                await self.run_single(**kwargs)
            except Exception:
                pass
    
    async def run(self, **kwargs) -> ExperimentResult:
        """
        运行完整实验
        
        Args:
            **kwargs: 传递给 run_single 的参数
            
        Returns:
            ExperimentResult: 实验结果
        """
        self.reproducibility.setup()
        self.setup_paths()
        
        status = "completed"
        error_message = None
        raw_data = []
        
        try:
            await self.setup()
            await self.warmup(**kwargs)
            
            for i in range(self.config.repeat_count):
                try:
                    result = await asyncio.wait_for(
                        self.run_single(**kwargs),
                        timeout=self.config.timeout
                    )
                    raw_data.append(result)
                except asyncio.TimeoutError:
                    raw_data.append({"error": "timeout", "run": i})
                except Exception as e:
                    raw_data.append({"error": str(e), "run": i})
            
            await self.teardown()
            
        except Exception as e:
            status = "failed"
            error_message = str(e)
            try:
                await self.teardown()
            except Exception:
                pass
        
        metrics = self._aggregate_metrics(raw_data)
        
        return ExperimentResult(
            config=self.config,
            metrics=metrics,
            raw_data=raw_data,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            environment=self.reproducibility.environment.to_dict() 
                if self.reproducibility.environment else {},
            status=status,
            error_message=error_message,
        )
    
    @abstractmethod
    def _aggregate_metrics(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多次实验结果（子类实现）"""
        pass
    
    def save_results(self, filename: str, subdir: Optional[str] = None) -> str:
        """
        保存实验结果
        
        Args:
            filename: 文件名
            subdir: 子目录
            
        Returns:
            str: 保存路径
        """
        output_path = get_output_path(filename, subdir)
        
        result = ExperimentResult(
            config=self.config,
            metrics=self._aggregate_metrics(self.results),
            raw_data=self.results,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            environment=self.reproducibility.environment.to_dict()
                if self.reproducibility.environment else {},
        )
        
        result.save(str(output_path))
        return str(output_path)


class IsolationExperiment(BaseExperiment):
    """隔离效果实验基类"""
    
    def __init__(
        self, 
        name: str = "isolation_experiment",
        seed: int = 42,
        warmup_runs: int = 3,
        repeat_count: int = 5
    ):
        config = ExperimentConfig(
            name=name,
            seed=seed,
            warmup_runs=warmup_runs,
            repeat_count=repeat_count,
        )
        super().__init__(config)
        self.baseline_latencies: List[float] = []
        self.stress_latencies: List[float] = []


class RecoveryExperiment(BaseExperiment):
    """崩溃恢复实验基类"""
    
    def __init__(
        self,
        name: str = "recovery_experiment",
        seed: int = 42,
        repeat_count: int = 5
    ):
        config = ExperimentConfig(
            name=name,
            seed=seed,
            warmup_runs=0,
            repeat_count=repeat_count,
        )
        super().__init__(config)
        self.recovery_times: List[float] = []


class CancellationExperiment(BaseExperiment):
    """取消语义实验基类"""
    
    def __init__(
        self,
        name: str = "cancellation_experiment",
        seed: int = 42,
        repeat_count: int = 10
    ):
        config = ExperimentConfig(
            name=name,
            seed=seed,
            warmup_runs=2,
            repeat_count=repeat_count,
        )
        super().__init__(config)
        self.cancellation_latencies: List[float] = []
