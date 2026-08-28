#!/usr/bin/env python3
"""
超级简化的测试，只看第一次加载模型
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
    print("只测试第一次加载模型")
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
    
    print("\n正在启动调度器 (preload_llm=True)...")
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
        
        print("等待模型加载完成 (10秒)...")
        for i in range(10):
            await asyncio.sleep(1)
            gpu_mem = check_gpu_memory()
            if gpu_mem:
                print(f"  [{i+1}/10] GPU 显存: {gpu_mem:.0f} MB")
        
        # 最终显存
        gpu_mem_final = check_gpu_memory()
        if gpu_mem_final:
            print(f"\n最终 GPU 显存: {gpu_mem_final:.0f} MB")
        
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
