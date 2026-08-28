#!/usr/bin/env python3
"""
测试实验脚本的初始化
"""

import os
import sys
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_gpu_memory():
    """检查 GPU 显存使用"""
    try:
        import subprocess
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return float(output.strip())
    except Exception as e:
        print(f"[GPU Check] 检查显存失败: {e}")
        return None


async def main():
    print("=" * 60)
    print("测试实验脚本初始化")
    print("=" * 60)
    
    try:
        from services.cpp_scheduler_engine import cpp_scheduler_engine
    except Exception as e:
        print(f"[Error] 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    model_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH", "")
    if not model_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "models", "qwen2.5-7b-instruct-q4_k_m.gguf")
    
    print(f"模型路径: {model_path}")
    if not os.path.exists(model_path):
        print(f"[Error] 模型文件不存在: {model_path}")
        return 1
    
    # 测试前显存
    gpu_mem_before = check_gpu_memory()
    if gpu_mem_before:
        print(f"测试前 GPU 显存: {gpu_mem_before:.0f} MB")
    
    print("\n正在启动调度器...")
    config = {
        "model_path": model_path,
        "max_context_size": 4096,
        "n_gpu_layers": -1,
        "max_batch_size": 512,
    }
    
    try:
        cpp_scheduler_engine.start(
            worker_count=1,
            gpu_config=config,
            preload_llm=True,
        )
        
        print("等待模型加载完成...")
        await asyncio.sleep(8)
        
        # 加载后显存
        gpu_mem_after = check_gpu_memory()
        if gpu_mem_after:
            print(f"\n加载后 GPU 显存: {gpu_mem_after:.0f} MB")
            if gpu_mem_before:
                diff = gpu_mem_after - gpu_mem_before
                print(f"显存增加: {diff:.0f} MB")
        
        print("\n简单测试推理...")
        ttft_start = time.time()
        first_token = False
        
        try:
            async for token in cpp_scheduler_engine.submit_llm_task("你好", max_tokens=50):
                if not first_token:
                    ttft = time.time() - ttft_start
                    first_token = True
                    print(f"TTFT: {ttft*1000:.1f} ms")
                print(token, end="", flush=True)
        except Exception as e:
            print(f"\n推理失败: {e}")
        
        print("\n")
        
        # 推理后显存
        gpu_mem_infer = check_gpu_memory()
        if gpu_mem_infer:
            print(f"推理后 GPU 显存: {gpu_mem_infer:.0f} MB")
        
        print("\n正在停止...")
        await cpp_scheduler_engine.stop()
        print("完成")
        
        return 0
        
    except Exception as e:
        print(f"[Error] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
