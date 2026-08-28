"""
导入辅助模块

处理独立部署和完整项目两种场景的导入问题
"""

from .paths import MVP_CORE_DIR

import sys
import os


def ensure_importable():
    """
    确保 mvp_core 模块可导入
    
    在独立部署场景下，将 MVP_CORE_DIR 添加到 sys.path，
    并创建 mvp_core 模块别名
    """
    path_str = str(MVP_CORE_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    
    # 在独立部署场景下，创建 mvp_core 别名
    # 这样 `from mvp_core.services.xxx` 就能工作
    if 'mvp_core' not in sys.modules:
        import types
        import importlib.util
        
        # 创建 mvp_core 包
        mvp_core = types.ModuleType('mvp_core')
        mvp_core.__path__ = [path_str]
        mvp_core.__file__ = os.path.join(path_str, '__init__.py')
        sys.modules['mvp_core'] = mvp_core
        
        # 添加子模块
        for subdir in ['services', 'data', 'domain', 'utils', 'presentation']:
            subpath = os.path.join(path_str, subdir)
            if os.path.isdir(subpath):
                submod = types.ModuleType(f'mvp_core.{subdir}')
                submod.__path__ = [subpath]
                submod.__file__ = os.path.join(subpath, '__init__.py')
                sys.modules[f'mvp_core.{subdir}'] = submod


def import_service(module_name: str):
    """
    导入服务模块
    
    Args:
        module_name: 模块名，如 'task_scheduler', 'state_backplane'
        
    Returns:
        模块对象
    """
    ensure_importable()
    
    # 尝试两种导入方式
    try:
        import importlib
        return importlib.import_module(f"mvp_core.services.{module_name}")
    except ImportError:
        import importlib
        return importlib.import_module(f"services.{module_name}")


def import_config():
    """导入配置模块"""
    ensure_importable()
    
    try:
        from mvp_core.config import get_settings
        return get_settings
    except ImportError:
        from config import get_settings
        return get_settings


def import_from_service(module_name: str, *names):
    """
    从服务模块导入指定名称
    
    Args:
        module_name: 模块名
        *names: 要导入的名称
        
    Returns:
        导入的对象元组
    """
    ensure_importable()
    
    try:
        module = __import__(
            f"mvp_core.services.{module_name}",
            fromlist=names
        )
    except ImportError:
        module = __import__(
            f"services.{module_name}",
            fromlist=names
        )
    
    if len(names) == 1:
        return getattr(module, names[0])
    return tuple(getattr(module, name) for name in names)
