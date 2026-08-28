"""C++ 调度引擎门面（单例）。

结构说明（2026-08 解耦拆分）：
- 共享导入与无状态工具函数 → services/cpp_engine_utils.py
- LLM GPU↔CPU 动态迁移 → services/cpp_engine_migration.py (MigrationMixin)
- LLM 后端装配与推理提交 → services/cpp_engine_llm.py (LLMBackendMixin)
- 图像后端装配与提交 → services/cpp_engine_image.py (ImageBackendMixin)
- 本文件只保留核心：单例、生命周期、熔断器状态机

兼容性：模块级单例 `cpp_scheduler_engine` 与类 `CPPSchedulerEngine` 的
导入路径不变，外部调用方零修改。
"""
import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from mvp_core.utils.logger import logger

try:
    from mvp_core.services.cpp_engine_utils import (
        ResourceManager,
        ResourceIsolationScheduler,
        is_cpp_scheduler_available,
    )
    from mvp_core.services.cpp_engine_migration import MigrationMixin
    from mvp_core.services.cpp_engine_llm import LLMBackendMixin
    from mvp_core.services.cpp_engine_image import ImageBackendMixin
except ImportError:
    from services.cpp_engine_utils import (  # type: ignore
        ResourceManager,
        ResourceIsolationScheduler,
        is_cpp_scheduler_available,
    )
    from services.cpp_engine_migration import MigrationMixin  # type: ignore
    from services.cpp_engine_llm import LLMBackendMixin  # type: ignore
    from services.cpp_engine_image import ImageBackendMixin  # type: ignore

_module_logger = logging.getLogger(__name__)


class CPPSchedulerEngine(MigrationMixin, LLMBackendMixin, ImageBackendMixin):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CPPSchedulerEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.scheduler = None
        self.enabled = bool(is_cpp_scheduler_available())
        self._initialized = True
        self._started = False

        self.resource_manager = None
        if ResourceManager:
            try:
                self.resource_manager = ResourceManager()
            except Exception:
                pass

        self._gpu_config: Optional[Dict[str, Any]] = None
        self._llm_backend: Optional[str] = None

        self._python_llm = None
        self._python_llm_lock = asyncio.Lock()
        self._last_llm_load_error: Optional[str] = None
        self._last_llm_load_ts: float = 0.0

        # 动态迁移相关状态
        self._python_force_cpu = False
        self._python_llm_device: Optional[str] = None
        self._prev_n_gpu_layers: Optional[int] = None
        self._prev_n_ctx: Optional[int] = None
        self._prev_n_batch: Optional[int] = None
        self._saved_llm_state = None
        self._saved_llm_state_ts: float = 0.0

        # 使用单线程 Executor 保证 LLM 操作（加载/推理）都在同一个线程执行，避免 CUDA 上下文问题
        from concurrent.futures import ThreadPoolExecutor
        self._llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MVP_LLMWorker")

        self._cpp_llm_lock = asyncio.Lock()
        self._cpp_worker_ready = False

        self._cpp_img_lock = asyncio.Lock()
        self._cpp_img_worker_ready = False

        self._breaker_threshold = int(os.getenv("XIAOYOU_CPP_BREAKER_THRESHOLD", "3") or 3)
        self._breaker_min_cooldown_s = float(os.getenv("XIAOYOU_CPP_BREAKER_MIN_COOLDOWN_S", "5") or 5)
        self._breaker_max_cooldown_s = float(os.getenv("XIAOYOU_CPP_BREAKER_MAX_COOLDOWN_S", "60") or 60)
        self._breaker = {
            "llm": {"failures": 0, "open_until": 0.0, "cooldown_s": self._breaker_min_cooldown_s},
            "image": {"failures": 0, "open_until": 0.0, "cooldown_s": self._breaker_min_cooldown_s},
        }

        self._last_llm_stats: Optional[Dict[str, Any]] = None

        if not self.enabled:
            logger.warning("C++ Scheduler is not available. Engine disabled.")

    def _breaker_is_open(self, kind: str) -> bool:
        st = self._breaker.get(kind)
        if not isinstance(st, dict):
            return False
        return time.time() < float(st.get("open_until", 0.0) or 0.0)

    def _breaker_on_success(self, kind: str) -> None:
        st = self._breaker.get(kind)
        if not isinstance(st, dict):
            return
        st["failures"] = 0
        st["open_until"] = 0.0
        st["cooldown_s"] = self._breaker_min_cooldown_s

    def _breaker_on_failure(self, kind: str) -> None:
        st = self._breaker.get(kind)
        if not isinstance(st, dict):
            return

        st["failures"] = int(st.get("failures", 0) or 0) + 1

        if int(st["failures"]) >= int(self._breaker_threshold):
            cooldown_s = float(st.get("cooldown_s", self._breaker_min_cooldown_s) or self._breaker_min_cooldown_s)
            cooldown_s = min(max(self._breaker_min_cooldown_s, cooldown_s), self._breaker_max_cooldown_s)
            st["open_until"] = time.time() + cooldown_s
            st["cooldown_s"] = min(cooldown_s * 2.0, self._breaker_max_cooldown_s)

    def start(
        self,
        worker_count: int = 4,
        gpu_config: Optional[Dict[str, Any]] = None,
        preload_llm: bool = False,
    ):
        if not self.enabled or self._started:
            return

        if ResourceIsolationScheduler is None:
            self.enabled = False
            return

        try:
            logger.info("正在启动 MVP 核心调度引擎 (增强版)...")
            self.scheduler = ResourceIsolationScheduler()
            ok = self.scheduler.initialize(int(worker_count))
            if not ok:
                raise RuntimeError("Failed to initialize C++ scheduler")

            if isinstance(gpu_config, dict):
                self._gpu_config = dict(gpu_config)
                backend = str(self._gpu_config.get("backend") or "python").strip().lower()
                if backend not in ("cpp", "python"):
                    backend = "python"

                allow_cpp_llm_worker = str(os.getenv("XIAOYOU_ALLOW_CPP_LLM_WORKER", "")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if backend == "cpp" and not allow_cpp_llm_worker:
                    backend = "python"
                self._llm_backend = backend

                if preload_llm:
                    if backend == "cpp":
                        self._setup_cpp_llm_worker(self._gpu_config)
                    else:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None

                        if loop is not None and loop.is_running():
                            loop.run_in_executor(self._llm_executor, self._setup_python_llm, self._gpu_config, loop)
                        else:
                            try:
                                self._llm_executor.submit(self._setup_python_llm, self._gpu_config, None).result()
                            except Exception:
                                self._setup_python_llm(self._gpu_config, None)

            self._started = True
            logger.info("MVP 核心调度引擎已启动，后端: %s", self._llm_backend)
        except Exception as e:
            logger.error("Failed to start C++ Scheduler: %s", e, exc_info=True)
            self.scheduler = None
            self.enabled = False

    async def stop(self):
        if not self._started:
            return
        self._started = False

        try:
            if self.scheduler:
                await asyncio.to_thread(self.scheduler.shutdown)
        except Exception:
            pass
        finally:
            self.scheduler = None
            self._cpp_worker_ready = False
            self._cpp_img_worker_ready = False

cpp_scheduler_engine = CPPSchedulerEngine()
