import os
import sys
import logging

logger = logging.getLogger(__name__)


def _import_scheduler_py():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if os.name == "nt":
        try:
            if os.path.isdir(current_dir):
                os.add_dll_directory(current_dir)
        except Exception:
            pass

        try:
            mvp_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            project_root = os.path.dirname(mvp_root)
            repo_root = os.path.dirname(project_root)

            libuv_candidates = [
                os.path.join(repo_root, "cpp_scheduler", "build", "_deps", "libuv-build", "Release"),
                os.path.join(repo_root, "cpp_scheduler", "build", "_deps", "libuv-build", "Debug"),
                os.path.join(project_root, "cpp_scheduler", "build", "_deps", "libuv-build", "Release"),
                os.path.join(project_root, "cpp_scheduler", "build", "_deps", "libuv-build", "Debug"),
                os.path.join(mvp_root, "cpp_scheduler", "build", "_deps", "libuv-build", "Release"),
                os.path.join(mvp_root, "cpp_scheduler", "build", "_deps", "libuv-build", "Debug"),
            ]
            for libuv_dir in libuv_candidates:
                if os.path.isdir(libuv_dir):
                    os.add_dll_directory(libuv_dir)
        except Exception:
            pass
        try:
            import llama_cpp  # type: ignore

            llama_dir = os.path.dirname(getattr(llama_cpp, "__file__", "") or "")
            llama_lib_dir = os.path.join(llama_dir, "lib")
            if os.path.isdir(llama_lib_dir):
                os.add_dll_directory(llama_lib_dir)
        except Exception:
            pass

    if current_dir not in sys.path:
        sys.path.append(current_dir)

    mvp_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(mvp_root)
    repo_root = os.path.dirname(project_root)

    search_paths = [
        os.path.join(mvp_root, "cpp_scheduler", "build", "Release"),
        os.path.join(mvp_root, "cpp_scheduler", "build", "Debug"),
        os.path.join(mvp_root, "cpp_scheduler", "build"),
        os.path.join(project_root, "cpp_scheduler", "build", "Release"),
        os.path.join(project_root, "cpp_scheduler", "build", "Debug"),
        os.path.join(project_root, "cpp_scheduler", "build"),
        os.path.join(repo_root, "cpp_scheduler", "build", "Release"),
        os.path.join(repo_root, "cpp_scheduler", "build", "Debug"),
        os.path.join(repo_root, "cpp_scheduler", "build"),
        os.path.join(repo_root, "core", "services", "scheduler"),
    ]

    for path in search_paths:
        if not os.path.isdir(path):
            continue

        try:
            has_lib = any(
                f.startswith("scheduler_py") and (f.endswith(".pyd") or f.endswith(".so"))
                for f in os.listdir(path)
            )
        except Exception:
            has_lib = False

        if not has_lib:
            continue

        if path not in sys.path:
            sys.path.append(path)

        if os.name == "nt":
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

        try:
            import scheduler_py  # type: ignore

            logger.info("Successfully imported scheduler_py from %s", path)
            return scheduler_py
        except Exception as e:
            logger.warning("Failed to import scheduler_py from %s: %s", path, e)

    try:
        import scheduler_py  # type: ignore

        logger.info("Successfully imported scheduler_py from environment")
        return scheduler_py
    except Exception:
        pass

    logger.warning("Could not find scheduler_py extension. C++ Scheduler features disabled.")
    return None


scheduler_py = _import_scheduler_py()

if scheduler_py:
    ResourceIsolationScheduler = scheduler_py.ResourceIsolationScheduler
    GPULLMWorker = scheduler_py.GPULLMWorker
    GPUImageWorker = getattr(scheduler_py, "GPUImageWorker", None)
    GPUImgWorker = getattr(scheduler_py, "GPUImgWorker", None) or GPUImageWorker
    LLMModelConfig = scheduler_py.LLMModelConfig
    ImgEngineType = (
        getattr(scheduler_py, "ImgEngineType", None)
        or getattr(scheduler_py, "ImageModel", None)
        or getattr(scheduler_py, "ImageEngineType", None)
    )
    ImgGenerationParams = (
        getattr(scheduler_py, "ImgGenerationParams", None)
        or getattr(scheduler_py, "ImageGenerationParams", None)
    )
    ImgTask = (
        getattr(scheduler_py, "ImgTask", None)
        or getattr(scheduler_py, "ImageTask", None)
    )
else:
    ResourceIsolationScheduler = None
    GPULLMWorker = None
    GPUImageWorker = None
    GPUImgWorker = None
    LLMModelConfig = None
    ImgEngineType = None
    ImgGenerationParams = None
    ImgTask = None


def is_cpp_scheduler_available():
    return scheduler_py is not None
