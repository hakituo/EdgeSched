import base64
import gc
import json
import os
import sys
import time
import urllib.request


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _forge_post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _save_first_image_b64(image_b64: str, out_path: str) -> None:
    image_bytes = base64.b64decode(image_b64.split(",", 1)[0])
    with open(out_path, "wb") as f:
        f.write(image_bytes)


def _load_llm(model_path: str):
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )


def _measure_ttft(llm, prompt: str) -> float:
    start = time.time()
    stream = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=24,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {}).get("content", "")
        if delta:
            return time.time() - start
    raise RuntimeError("LLM 未返回任何 token")


def run_manual_switch_experiment(model_path: str, forge_base_url: str):
    results = {
        "llm_model_path": model_path,
        "forge_base_url": forge_base_url,
        "image_width": 1024,
        "image_height": 1024,
        "steps": 8,
        "gpu_phase_ttft_s": [],
        "manual_unload_time_s": None,
        "image_tasks": [],
        "llm_reload_time_s": None,
        "restored_ttft_s": None,
    }

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
        "images",
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Baseline 1: Manual llama.cpp Switch 实验")
    print("模拟: 图像任务到达时，手动卸载 LLM -> 生图 -> 再重新加载 LLM。")
    print("=" * 60)

    print("\n[阶段 1] GPU 上正常聊天...")
    llm_load_start = time.time()
    llm = _load_llm(model_path)
    print(f"初始 LLM 加载耗时: {time.time() - llm_load_start:.3f}s")
    for i in range(3):
        ttft = _measure_ttft(llm, f"这是第 {i+1} 轮聊天")
        results["gpu_phase_ttft_s"].append(ttft)
        print(f"第 {i+1} 轮 GPU TTFT: {ttft:.3f}s")
        time.sleep(1.0)

    print("\n[阶段 2] 手动卸载 LLM，让位给图像任务...")
    unload_start = time.time()
    del llm
    gc.collect()
    results["manual_unload_time_s"] = time.time() - unload_start
    print(f"LLM 手动卸载耗时: {results['manual_unload_time_s']:.3f}s")

    print("\n[阶段 3] 生成两张 1024x1024 图片...")
    txt2img_url = forge_base_url.rstrip("/") + "/sdapi/v1/txt2img"
    for i in range(2):
        img_start = time.time()
        payload = {
            "prompt": f"生成第 {i+1} 张高清图片，赛博朋克风格，精致细节",
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "cfg_scale": 1.0,
            "sampler_name": "Euler",
        }
        resp = _forge_post(txt2img_url, payload)
        latency = time.time() - img_start
        out_path = os.path.join(output_dir, f"manual_switch_{int(time.time() * 1000)}_{i+1}.png")
        if resp.get("images"):
            _save_first_image_b64(resp["images"][0], out_path)
        results["image_tasks"].append({
            "index": i + 1,
            "latency_s": latency,
            "path": out_path,
        })
        print(f"第 {i+1} 张图片耗时: {latency:.3f}s -> {out_path}")

    print("\n[阶段 4] 手动重新加载 LLM 并恢复聊天...")
    reload_start = time.time()
    llm = _load_llm(model_path)
    results["llm_reload_time_s"] = time.time() - reload_start
    results["restored_ttft_s"] = _measure_ttft(llm, "图片生成结束后，继续聊天")
    print(f"LLM 重新加载耗时: {results['llm_reload_time_s']:.3f}s")
    print(f"恢复后 TTFT: {results['restored_ttft_s']:.3f}s")

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiment_results",
    )
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "baseline_manual_llama_switch_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n实验完成！结果已保存至: {out_file}")


if __name__ == "__main__":
    model_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH", "/home/leslie/mvp_core/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf")
    forge_base_url = os.environ.get("XIAOYOU_FORGE_BASE_URL", "http://127.0.0.1:7860")
    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型文件: {model_path}")
        sys.exit(1)
    run_manual_switch_experiment(model_path, forge_base_url)
