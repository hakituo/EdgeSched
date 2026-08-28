"""C++ 引擎共享导入与无状态工具函数。

从 cpp_scheduler_engine.py 拆出：
- resource_manager / cpp_scheduler_wrapper / llama_cpp / admission metrics 的
  双路径守卫导入（mvp_core.* 与 services.* 两种导入方式均可工作）
- llama.cpp 内部补丁、OOM/CUDA 错误识别、nvidia-smi 显存读取等工具函数
"""
import logging
import shutil
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from mvp_core.services.resource_manager import (
        ResourceManager,
        ResourcePriority,
        ResourceState
    )
except ImportError:
    ResourceManager = None
    ResourcePriority = None
    ResourceState = None

try:
    from mvp_core.services.cpp_scheduler_wrapper import (
        ResourceIsolationScheduler,
        GPULLMWorker,
        GPUImageWorker,
        GPUImgWorker,
        LLMModelConfig,
        ImgEngineType,
        ImgGenerationParams,
        ImgTask,
        is_cpp_scheduler_available,
        scheduler_py,
    )
except Exception:
    from services.cpp_scheduler_wrapper import (  # type: ignore
        ResourceIsolationScheduler,
        GPULLMWorker,
        GPUImageWorker,
        GPUImgWorker,
        LLMModelConfig,
        ImgEngineType,
        ImgGenerationParams,
        ImgTask,
        is_cpp_scheduler_available,
        scheduler_py,
    )

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from mvp_core.services.admission_controller import get_runtime_metrics
except ImportError:
    try:
        from services.admission_controller import get_runtime_metrics  # type: ignore
    except ImportError:
        def get_runtime_metrics():  # type: ignore
            return None

_LLAMA_CPP_INTERNALS_PATCHED = False


def _set_attr(obj: object, names: List[str], value: object) -> bool:
    for name in names:
        try:
            if hasattr(obj, name):
                setattr(obj, name, value)
                return True
        except Exception:
            continue
    return False


def _patch_llama_cpp_internals():
    global _LLAMA_CPP_INTERNALS_PATCHED
    if _LLAMA_CPP_INTERNALS_PATCHED:
        return
    try:
        import llama_cpp._internals as _internals

        llama_model_cls = getattr(_internals, "LlamaModel", None)
        if llama_model_cls is not None:
            if hasattr(llama_model_cls, "close"):
                original_close = llama_model_cls.close

                def _safe_close(self):
                    try:
                        if not hasattr(self, "sampler"):
                            setattr(self, "sampler", None)
                        return original_close(self)
                    except AttributeError:
                        return None

                llama_model_cls.close = _safe_close

            if hasattr(llama_model_cls, "__del__"):
                original_del = llama_model_cls.__del__

                def _safe_del(self):
                    try:
                        if not hasattr(self, "sampler"):
                            setattr(self, "sampler", None)
                        return original_del(self)
                    except Exception:
                        try:
                            return getattr(self, "close")()
                        except Exception:
                            return None

                llama_model_cls.__del__ = _safe_del
    except Exception:
        pass
    _LLAMA_CPP_INTERNALS_PATCHED = True


def _is_oom_error(msg: str) -> bool:
    lowered = (msg or "").lower()
    return any(
        k in lowered
        for k in [
            "out of memory",
            "cuda error",
            "ggml-cuda",
            "cublas",
            "hip error",
            "vram",
            "memory allocation",
            "failed to allocate",
            "not enough memory",
        ]
    )


def _is_cuda_backend_error(msg: str) -> bool:
    lowered = (msg or "").lower()
    return any(
        k in lowered
        for k in [
            "ggml-cuda",
            "cuda error",
            "cublas",
            "hip error",
            "illegal memory access",
            "device-side assert",
            "driver",
        ]
    )


def _nvidia_smi_total_used_mb() -> Optional[int]:
    try:
        if not shutil.which("nvidia-smi"):
            return None
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0:
            return None
        total = 0
        for line in (proc.stdout or "").splitlines():
            line = (line or "").strip()
            if not line:
                continue
            try:
                total += int(float(line))
            except Exception:
                continue
        return total
    except Exception:
        return None
