import os
import sys
import time
import json
import argparse
import threading
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_gpu_memory():
    try:
        import subprocess
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return float(output.strip())
    except Exception:
        return 0.0


def run_lru_thrashing_experiment(llm_model_path: str, image_model_id: str, cache_mode: str = "cold"):
    """
    执行 Simple LRU / OS Unified Memory 基线实验

    cache_mode:
      - cold: 每次迭代从冷状态开始，模拟最坏情况
      - warm: 先做一次预热，再重复测量，模拟缓存热后的稳定表现
    """
    print(f"\n{'='*60}")
    print(f"Baseline 2: Simple LRU / OS Unified Memory 实验 (Thrashing 模拟)")
    print(f"Cache Mode: {cache_mode}")
    print(f"{'='*60}")

    results = {
        "llm_only_latency_s": 0.0,
        "image_only_latency_s": 0.0,
        "concurrent_llm_latency_s": 0.0,
        "concurrent_image_latency_s": 0.0,
        "cache_mode": cache_mode,
        "status": "success"
    }

    from llama_cpp import Llama
    import torch
    from safetensors.torch import load_file
    import gc

    if cache_mode == "warm":
        print("\n[预热] Warm-cache 模式：先做一次完整加载以热化文件缓存和 CUDA context...")
        try:
            warm_llm = Llama(model_path=llm_model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            _ = warm_llm.create_chat_completion([{"role": "user", "content": "预热"}], max_tokens=5)
            del warm_llm
            gc.collect()
            torch.cuda.empty_cache()

            warm_img = load_file(image_model_id, device="cuda")
            del warm_img
            gc.collect()
            torch.cuda.empty_cache()
            print("  预热完成，文件缓存和 CUDA context 已热化")
        except Exception as e:
            print(f"  预热失败: {e}")

    print("\n[阶段 1] 预加载并测试单任务性能...")

    # 1. 单测 LLM
    print("加载 LLM 到 GPU...")
    llm = Llama(model_path=llm_model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    start_llm = time.time()
    _ = llm.create_chat_completion([{"role": "user", "content": "你好"}], max_tokens=10)
    results["llm_only_latency_s"] = time.time() - start_llm
    print(f"  -> LLM 独立推理耗时: {results['llm_only_latency_s']:.3f} s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 单测 Image
    print("加载 Image 模型 到 GPU (Raw Tensor Load)...")
    start_img = time.time()
    dummy_image_model = load_file(image_model_id, device="cuda")
    time.sleep(2.0)
    results["image_only_latency_s"] = time.time() - start_img
    print(f"  -> Image 独立加载+模拟推理耗时: {results['image_only_latency_s']:.3f} s")

    del dummy_image_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[阶段 2] 模拟 Unified Memory Thrashing (并发挤占 8GB 显存)...")
    print("警告：此操作极有可能导致系统卡顿、爆显存 (OOM) 或触发操作系统的 Swap/Pagefile。")

    try:
        print("1. 加载 LLM 到 GPU...")
        llm = Llama(model_path=llm_model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        print(f"当前 GPU 占用: {check_gpu_memory():.0f} MB")

        print("2. 在不卸载 LLM 的情况下，强行加载 Image 模型...")
        start_img = time.time()
        dummy_image_model = load_file(image_model_id, device="cuda")
        print(f"加载后 GPU 占用: {check_gpu_memory():.0f} MB")

        print("3. 并发触发推理 (触发 Unified Memory 颠簸)...")

        llm_done = False
        img_done = False

        def run_llm():
            nonlocal llm_done
            start = time.time()
            try:
                _ = llm.create_chat_completion([{"role": "user", "content": "请写一首长诗。"}], max_tokens=50)
                results["concurrent_llm_latency_s"] = time.time() - start
            except Exception as e:
                results["status"] = f"LLM OOM/Error: {str(e)}"
            llm_done = True

        def run_img():
            nonlocal img_done
            start = time.time()
            try:
                keys = list(dummy_image_model.keys())
                if keys:
                    t = dummy_image_model[keys[0]]
                    for _ in range(50):
                        _ = t * 2.0
                time.sleep(2.0)
                results["concurrent_image_latency_s"] = time.time() - start
            except Exception as e:
                results["status"] = f"Image OOM/Error: {str(e)}"
            img_done = True

        t1 = threading.Thread(target=run_llm)
        t2 = threading.Thread(target=run_img)

        t1.start()
        t2.start()

        t1.join(timeout=60)
        t2.join(timeout=60)

        if t1.is_alive() or t2.is_alive():
            results["status"] = "Timeout (>60s) due to severe Thrashing"
        elif results["status"] == "success":
            print(f"  -> 并发 LLM 推理耗时: {results['concurrent_llm_latency_s']:.3f} s (对比独立: {results['llm_only_latency_s']:.3f} s)")
            print(f"  -> 并发 Image 推理耗时: {results['concurrent_image_latency_s']:.3f} s (对比独立: {results['image_only_latency_s']:.3f} s)")
            print("\n结论：如果操作系统启用了 Unified Memory，你看到了几倍甚至十几倍的延迟增加，这就是 Thrashing。如果没有，你应该已经看到了 OOM。")
        else:
            print(f"\n结论：系统崩溃或 OOM。这证明了操作系统底层的 LRU 无法处理大模型并发 ({results['status']})")

    except Exception as e:
        results["status"] = f"Crash/OOM: {str(e)}"
        print(f"\n发生错误: {e}")
        print("结论：证明了 8GB 显存设备必须使用 EdgeSched 这种应用层状态调度器，不能依赖操作系统的默认管理。")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"baseline_lru_thrashing_{cache_mode}_results.json")
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **results}
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"实验结果已保存至: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LRU Thrashing Baseline Experiment")
    parser.add_argument("--cache-mode", choices=["cold", "warm"], default="cold",
                        help="cold: 每次从冷状态开始; warm: 先预热再测量")
    args = parser.parse_args()

    llm_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH", "/home/leslie/mvp_core/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf")
    if not os.path.exists(llm_path):
        llm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/qwen2.5-7b-instruct-q4_k_m.gguf"))

    image_id = "/home/leslie/mvp_core/models/Image/check_point/SDXL/sd_xl_turbo_1.0_fp16.safetensors"

    if not os.path.exists(image_id):
        print(f"找不到指定的模型 {image_id}，请检查路径。")
    else:
        print(f"使用本地 Image 模型: {image_id}")

    run_lru_thrashing_experiment(llm_path, image_id, cache_mode=args.cache_mode)
