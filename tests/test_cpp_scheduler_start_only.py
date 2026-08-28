import asyncio
import ctypes
import os
import sys
import time


def _bootstrap_cpp_scheduler() -> None:
    build_dir = os.environ.get("XIAOYOU_CPP_SCHED_BUILD", "/home/leslie/mvp_core/cpp_scheduler/build")
    build_bin_dir = os.path.join(build_dir, "bin")
    libllama_path = os.path.join(build_bin_dir, "libllama.so.0")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    if os.path.exists(libllama_path):
        ctypes.CDLL(libllama_path, mode=ctypes.RTLD_GLOBAL)


_bootstrap_cpp_scheduler()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("XIAOYOU_ALLOW_CPP_LLM_WORKER", "1")

from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine


async def main():
    model_path = os.environ.get(
        "XIAOYOU_LLM_MODEL_PATH",
        "/home/leslie/mvp_core/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf",
    )
    gpu_config = {
        "backend": "cpp",
        "model_path": model_path,
        "model_type": "llama",
        "gpu_device_id": 0,
        "n_gpu_layers": -1,
        "max_context_size": 256,
        "max_batch_size": 16,
    }
    print("=== Start Only Validation ===")
    t0 = time.time()
    cpp_scheduler_engine.start(worker_count=1, gpu_config=gpu_config, preload_llm=True)
    print(f"started in {time.time() - t0:.3f}s enabled={cpp_scheduler_engine.enabled}")
    await asyncio.sleep(2.0)
    await cpp_scheduler_engine.stop()
    print("stopped")


if __name__ == "__main__":
    asyncio.run(main())
