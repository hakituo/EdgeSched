import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


def _ensure_mvp_core_on_path() -> None:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    mvp_core_dir = os.path.dirname(tests_dir)
    legacy_dir = os.path.dirname(mvp_core_dir)
    repo_root = os.path.dirname(legacy_dir)

    for p in (repo_root, legacy_dir, mvp_core_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_mvp_core_on_path()


def pytest_configure(config) -> None:
    """注册需要真实硬件/环境的测试标记。"""

    config.addinivalue_line(
        "markers",
        "wsl_gpu: 需要 WSL2 + CUDA + MPS 的集成测试，默认不运行",
    )


@pytest.fixture
def short_tmp() -> Iterator[Path]:
    """短绝对路径临时目录（MPS daemon 启动类测试专用）。

    MPS control 通道是 Unix domain socket，``sun_path`` 上限 108 字节；
    pytest 的 ``tmp_path`` 本身约 60+ 字节，叠加 ``edgesched_mps_*`` 私有
    目录名与最长的 ``control_privileged`` socket 名后会超限，使
    ``MpsRuntime.start`` 的路径守卫误伤正常启动路径。这里在 ``/tmp`` 下
    创建短目录并结束后清理，既走真实 ``mkdtemp`` 路径又不触发守卫。
    """

    base = Path(tempfile.mkdtemp(prefix="mt_", dir="/tmp"))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)
