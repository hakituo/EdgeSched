"""
路径管理模块

提供统一的项目路径查找和管理功能
"""

import os
import sys
from pathlib import Path
from typing import Optional


def find_project_root() -> Path:
    """
    查找项目根目录
    
    策略：
    1. 查找 pyproject.toml
    2. 查找 .git 目录
    3. 查找特定标记文件
    4. 查找 mvp_core 目录（独立部署场景）
    
    Returns:
        Path: 项目根目录
        
    Raises:
        RuntimeError: 无法找到项目根目录
    """
    curr = Path(__file__).resolve()
    
    for parent in curr.parents:
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "legacy").is_dir() and (parent / "core").is_dir():
            return parent
        if (parent / "services").is_dir() and (parent / "experiments").is_dir():
            return parent
    
    raise RuntimeError(
        "Cannot find project root. "
        "Please ensure the experiment is run from within the project directory."
    )


PROJECT_ROOT: Path = find_project_root()

# 判断是否为独立部署（mvp_core 目录本身就是根目录）
if (PROJECT_ROOT / "services").is_dir() and (PROJECT_ROOT / "experiments").is_dir():
    MVP_CORE_DIR: Path = PROJECT_ROOT
else:
    MVP_CORE_DIR: Path = PROJECT_ROOT / "legacy" / "mvp_core"

RESULTS_DIR: Path = MVP_CORE_DIR / "experiment_results"

_path_initialized: bool = False


def setup_paths() -> None:
    """
    设置 Python 模块搜索路径
    
    将项目根目录、legacy 目录和 mvp_core 目录添加到 sys.path
    """
    global _path_initialized
    
    if _path_initialized:
        return
    
    paths_to_add = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "legacy"),
        str(MVP_CORE_DIR),
    ]
    
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
    
    _path_initialized = True


def ensure_results_dir() -> Path:
    """
    确保实验结果目录存在
    
    Returns:
        Path: 实验结果目录路径
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_output_path(filename: str, subdir: Optional[str] = None) -> Path:
    """
    获取输出文件路径
    
    Args:
        filename: 文件名
        subdir: 可选的子目录名
        
    Returns:
        Path: 完整的输出路径
    """
    base = ensure_results_dir()
    if subdir:
        base = base / subdir
        base.mkdir(parents=True, exist_ok=True)
    return base / filename
