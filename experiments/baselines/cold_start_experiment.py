import os
import sys
import time
import json
import gc
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def _force_gpu_cleanup():
    """强制释放 GPU 显存"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

def check_gpu_memory():
    """检查 GPU 显存使用"""
    try:
        import subprocess
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True
        )
        parts = output.strip().split(",")
        used_mb = float(parts[0].strip())
        total_mb = float(parts[1].strip())
        return used_mb, total_mb
    except Exception:
        return 0.0, 0.0

def check_system_memory():
    """检查系统内存（使用 available 而非 percent，避免 buffer/cache 误报）"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
            "used_pct": mem.percent,
            "available_pct": round(mem.available / mem.total * 100, 1),
        }
    except ImportError:
        return None

def generate_long_prompt(num_tokens: int = 2000) -> str:
    """生成一段长文本作为对话历史的模拟"""
    sentence = "这是用户与AI之间的一段非常长的历史对话记录。AI正在努力记住所有的上下文。用户提问了很多关于系统架构和资源调度的问题。AI做出了详细的回答。这只是一个用于测试大语言模型KV Cache（键值缓存）恢复时间的模拟文本。"
    repeat_times = max(1, num_tokens // 50)
    history = "\n".join([f"回合 {i}: " + sentence for i in range(repeat_times)])
    return history + "\n用户: 请总结一下我们刚才的对话。"

def run_cold_start_experiment(model_path: str, context_sizes: list):
    """
    执行冷启动（Naive Cold-Start）与状态热切换（Atomic Hot-Swap）的对比实验

    论文 Section 6.3: 对比重启式恢复 vs Atomic Hot-Swap 的恢复延迟
    实验条件: n_gpu_layers=-1 (全GPU)，不可降级，否则数据不可比
    """
    from llama_cpp import Llama

    results = {
        "cold_start": {},
        "hot_swap": {}
    }

    for ctx_idx, ctx_size in enumerate(context_sizes):
        print(f"\n{'='*50}")
        print(f"正在测试上下文长度: {ctx_size} Tokens ({ctx_idx+1}/{len(context_sizes)})")
        print(f"{'='*50}")

        vram_used, vram_total = check_gpu_memory()
        mem_info = check_system_memory()
        if mem_info:
            print(f"  [资源] VRAM: {vram_used:.0f}/{vram_total:.0f} MB | RAM: {mem_info['available_gb']}/{mem_info['total_gb']}GB 可用 ({mem_info['available_pct']}%)")
        else:
            print(f"  [资源] VRAM: {vram_used:.0f}/{vram_total:.0f} MB")

        print("[1/4] 初始化 LLM 引擎 (Cold Start 阶段 1: 加载权重)...")
        start_load = time.time()
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        load_time = time.time() - start_load
        print(f"  -> 模型加载耗时: {load_time:.2f} 秒")

        print(f"[2/4] 模拟冷启动重计算 (Cold Start 阶段 2: Prefill {ctx_size} tokens)...")
        prompt = generate_long_prompt(ctx_size)

        start_prefill = time.time()
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            stream=False
        )
        prefill_time = time.time() - start_prefill
        print(f"  -> 冷启动 Prefill 耗时: {prefill_time:.2f} 秒")

        results["cold_start"][ctx_size] = {
            "load_time_s": load_time,
            "prefill_time_s": prefill_time,
            "total_recovery_time_s": load_time + prefill_time
        }

        print("[3/4] 模拟 Urgent Mode: 保存 KV Cache 到内存...")
        start_save = time.time()
        state = llm.save_state()
        save_time = time.time() - start_save

        try:
            if hasattr(state, 'data'):
                state_size_mb = len(state.data) / (1024 * 1024)
            elif isinstance(state, (bytes, bytearray)):
                state_size_mb = len(state) / (1024 * 1024)
            elif hasattr(state, '__len__'):
                state_size_mb = len(state) / (1024 * 1024)
            else:
                state_size_mb = 0.0
        except Exception:
            state_size_mb = 0.0

        print(f"  -> KV Cache 提取耗时: {save_time:.3f} 秒 (大小: {state_size_mb:.1f} MB)")

        del llm
        _force_gpu_cleanup()

        vram_used, vram_total = check_gpu_memory()
        print(f"  [资源] 释放后 VRAM: {vram_used:.0f}/{vram_total:.0f} MB")

        print("[4/4] 模拟 Atomic Hot-Swap: 注入 KV Cache...")
        start_reload = time.time()
        llm_new = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        reload_time = time.time() - start_reload

        start_restore = time.time()
        llm_new.load_state(state)
        restore_time = time.time() - start_restore

        print(f"  -> 预加载权重耗时 (后台隐式): {reload_time:.2f} 秒")
        print(f"  -> 注入 KV Cache 耗时 (用户显式中断): {restore_time:.3f} 秒")

        results["hot_swap"][ctx_size] = {
            "save_time_s": save_time,
            "restore_time_s": restore_time,
            "state_size_mb": state_size_mb,
            "user_interruption_time_s": restore_time
        }

        del llm_new
        del state
        _force_gpu_cleanup()

        if ctx_idx < len(context_sizes) - 1:
            print("  等待显存释放 (5s)...")
            time.sleep(5)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "baseline_cold_start_results.json")
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **results}
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n实验完成！结果已保存至: {out_file}")

    print("\n--- 实验结论总结 ---")
    print(f"{'Context Size':<15} | {'Naive Cold-Start (s)':<25} | {'EdgeSched Hot-Swap Interruption (s)':<35}")
    print("-" * 80)
    for ctx in context_sizes:
        cold = results["cold_start"][ctx]["total_recovery_time_s"]
        hot = results["hot_swap"][ctx]["user_interruption_time_s"]
        print(f"{ctx:<15} | {cold:<25.3f} | {hot:<35.3f}")

if __name__ == "__main__":
    print("启动 Baseline 1: Naive Cold-Start 实验")

    model_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH")
    if not model_path:
        model_path = "/home/leslie/mvp_core/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf"
        if not os.path.exists(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/qwen2.5-7b-instruct-q4_k_m.gguf"))

    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型文件: {model_path}")
        print("请设置环境变量 XIAOYOU_LLM_MODEL_PATH 或将模型放置于正确的约定路径。")
        sys.exit(1)

    print(f"使用 LLM 模型: {model_path}")

    test_context_sizes = [512, 1024, 2048, 3072]
    run_cold_start_experiment(model_path, test_context_sizes)
