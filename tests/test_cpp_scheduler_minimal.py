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


async def main() -> None:
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
        "image_engine_type": "SD15_TURBO",
        "image_output_dir": "/home/leslie/mvp_core/output/images",
    }

    print("=== Minimal C++ Scheduler Validation ===")
    print(f"model_path={model_path}")
    print(f"gpu_config={gpu_config}")

    start = time.time()
    cpp_scheduler_engine.start(worker_count=1, gpu_config=gpu_config, preload_llm=True)
    print(f"engine_started_in={time.time() - start:.3f}s enabled={cpp_scheduler_engine.enabled}")

    first_token = None
    collected = []
    gen_start = time.time()
    try:
        async for token in cpp_scheduler_engine.submit_llm_task(
            "请只回复两个汉字：你好",
            request_id="cpp-minimal-1",
            max_tokens=8,
            temperature=0.1,
            stream=False,
            first_token_timeout=30.0,
            total_timeout=45.0,
        ):
            if first_token is None:
                first_token = time.time() - gen_start
                print(f"first_token_s={first_token:.3f}")
            collected.append(token)
            if len("".join(collected)) >= 8:
                break
    finally:
        await cpp_scheduler_engine.stop()

    print(f"output={''.join(collected)!r}")


if __name__ == "__main__":
    asyncio.run(main())
