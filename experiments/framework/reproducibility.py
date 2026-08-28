"""
实验可重复性保障模块

提供：
- 随机种子管理
- 环境信息记录
- 配置哈希
"""

import hashlib
import json
import os
import platform
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EnvironmentInfo:
    """环境信息"""
    
    python_version: str = ""
    platform_info: str = ""
    os_name: str = ""
    cpu_count: int = 0
    packages: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def collect(cls) -> "EnvironmentInfo":
        """收集当前环境信息"""
        info = cls(
            python_version=platform.python_version(),
            platform_info=platform.platform(),
            os_name=os.name,
            cpu_count=os.cpu_count() or 0,
        )
        
        try:
            import pkg_resources
            info.packages = {
                pkg.key: pkg.version
                for pkg in pkg_resources.working_set
            }
        except Exception:
            pass
        
        relevant_vars = [
            "XIAOYOU_BENCH_LLM_PATH",
            "XIAOYOU_BENCH_TTS_PATH",
            "XIAOYOU_TTS_REF_AUDIO_PATH",
            "XIAOYOU_BENCH_SD_PATH",
            "XIAOYOU_BENCH_N_CTX",
            "XIAOYOU_BENCH_N_GPU_LAYERS",
            "XIAOYOU_USE_CPP_SCHEDULER",
            "CUDA_VISIBLE_DEVICES",
        ]
        for var in relevant_vars:
            value = os.environ.get(var)
            if value is not None:
                info.env_vars[var] = value
        
        return info
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "os_name": self.os_name,
            "cpu_count": self.cpu_count,
            "packages": self.packages,
            "env_vars": self.env_vars,
        }


class ReproducibilityManager:
    """可重复性管理器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化可重复性管理器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        self.environment: Optional[EnvironmentInfo] = None
        self._initialized = False
    
    def setup(self) -> None:
        """设置随机种子并收集环境信息"""
        if self._initialized:
            return
        
        import random
        random.seed(self.seed)
        
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass
        
        self.environment = EnvironmentInfo.collect()
        self._initialized = True
    
    def calc_config_hash(self, config: Dict[str, Any]) -> str:
        """
        计算配置哈希
        
        Args:
            config: 配置字典
            
        Returns:
            str: 配置哈希值（前16位）
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def create_manifest(
        self, 
        config: Dict[str, Any],
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        创建实验清单
        
        Args:
            config: 实验配置
            experiment_name: 实验名称
            
        Returns:
            Dict: 实验清单
        """
        if not self._initialized:
            self.setup()
        
        return {
            "experiment_name": experiment_name,
            "seed": self.seed,
            "config": config,
            "config_hash": self.calc_config_hash(config),
            "environment": self.environment.to_dict() if self.environment else {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    
    def save_manifest(
        self, 
        filepath: str, 
        config: Dict[str, Any],
        experiment_name: str
    ) -> None:
        """
        保存实验清单到文件
        
        Args:
            filepath: 保存路径
            config: 实验配置
            experiment_name: 实验名称
        """
        manifest = self.create_manifest(config, experiment_name)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
