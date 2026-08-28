#!/usr/bin/env python3
"""
独立测试 llama-cpp-python 的 GPU 加载和显存使用
"""

import os
import sys
import time

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


def main():
    print("=" * 60)
    print("llama-cpp-python GPU 独立测试")
    print("=" * 60)
    
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
    
    print("\n正在加载 llama-cpp-python...")
    try:
        from llama_cpp import Llama
        print("llama-cpp-python 导入成功")
    except Exception as e:
        print(f"[Error] 导入失败: {e}")
        return 1
    
    print("\n正在加载模型 (n_gpu_layers=-1, n_ctx=4096)...")
    try:
        start = time.time()
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            n_batch=512,
            n_threads=4,
            verbose=True
        )
        load_time = time.time() - start
        print(f"模型加载成功，耗时: {load_time:.2f}s")
        
        # 检查模型信息
        if hasattr(llm, "model"):
            print(f"\n模型信息:")
            if hasattr(llm.model, "n_layers"):
                print(f"  层数: {llm.model.n_layers}")
            if hasattr(llm.model, "n_ctx"):
                print(f"  上下文: {llm.model.n_ctx}")
        
        # 加载后显存
        time.sleep(1)
        gpu_mem_after = check_gpu_memory()
        if gpu_mem_after:
            print(f"\n加载后 GPU 显存: {gpu_mem_after:.0f} MB")
            if gpu_mem_before:
                diff = gpu_mem_after - gpu_mem_before
                print(f"显存增加: {diff:.0f} MB")
        
        # 简单测试推理
        print("\n正在测试推理...")
        prompt = "你好，请介绍一下你自己。"
        print(f"Prompt: {prompt}")
        
        start = time.time()
        first_token = False
        ttft = 0
        
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            stream=True,
        )
        
        for chunk in output:
            if not first_token:
                ttft = time.time() - start
                first_token = True
                print(f"TTFT: {ttft*1000:.1f} ms")
            
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                print(content, end="", flush=True)
        
        print("\n")
        
        # 推理后显存
        gpu_mem_infer = check_gpu_memory()
        if gpu_mem_infer:
            print(f"推理后 GPU 显存: {gpu_mem_infer:.0f} MB")
        
        print("\n测试完成，清理...")
        if hasattr(llm, "close"):
            llm.close()
        
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"[Error] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
