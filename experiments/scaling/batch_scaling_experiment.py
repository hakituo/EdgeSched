"""
Forge API batch scaling 特性实验

目的: 在真实 Forge 后端上测出 T(b) 曲线, 判断 batch 聚批是否值得进入调度模型。

固定总共生成 N 张图, 对每个 batch size b:
  - b=1: 执行 N 次 txt2img 调用
  - b=2: 执行 N/2 次调用 (每次 batch_size=2)
  - b=4: 执行 N/4 次调用

计算:
  T(b)          = 单次 batch_size=b 调用的耗时 (跨迭代取均值)
  Throughput(b) = b / T(b)
  Speedup(b)    = Throughput(b) / Throughput(1)

结论分三档:
  - Speedup <= 1.10: batch 几乎无收益, 首请求立即执行, 不考虑等待
  - 1.10 < Speedup <= 1.50: 有一定收益, 队列有积压时顺带批处理, 不主动等待
  - Speedup > 1.50: 收益明显, 才值得研究 micro-batching 等待窗口

用法 (WSL2, Forge 服务需先启动):
  /home/leslie/.venvs/mvp_core-cu128/bin/python \
      experiments/scaling/batch_scaling_experiment.py
  可选参数:
      --total-images 8 --batch-sizes 1,2,4,8 --iterations 3
      --width 1024 --height 1024 --steps 8 --cfg-scale 1.0

产物: experiment_results/batch_scaling/batch_scaling_<时间戳>.json
注意: 本实验只测真实 Forge 特性曲线, 结果不直接写入论文正文数据源。
"""

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MVP_CORE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(MVP_CORE_DIR, "experiment_results", "batch_scaling")


def query_vram_used_mb() -> int | None:
    """查询当前 GPU 显存占用 (MB), 失败返回 None。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class VRAMSampler:
    """在 txt2img 请求期间后台采样显存峰值, 用于区分计算瓶颈与显存换出瓶颈。"""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            v = query_vram_used_mb()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 5)

    def peak_mb(self) -> int | None:
        # 去掉最后 1 个样本: 请求结束后显存可能已释放, 峰值取自请求进行中
        valid = self.samples[:-1] if len(self.samples) > 1 else self.samples
        return max(valid) if valid else None


def forge_get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def forge_post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_forge_available(base_url: str) -> bool:
    try:
        forge_get(base_url.rstrip("/") + "/sdapi/v1/sd-models", timeout=5)
        return True
    except Exception:
        return False


def collect_env_info(base_url: str) -> dict:
    """尽力采集当前 Forge 模型与 GPU 信息, 失败不阻塞实验。"""
    info = {"forge_base_url": base_url}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        info["gpu"] = out.strip()
    except Exception as e:
        info["gpu_error"] = str(e)
    try:
        models = forge_get(base_url.rstrip("/") + "/sdapi/v1/sd-models", timeout=10)
        info["available_models"] = [m.get("title") for m in models if m.get("title")]
    except Exception as e:
        info["available_models_error"] = str(e)
    try:
        options = forge_get(base_url.rstrip("/") + "/sdapi/v1/options", timeout=10)
        info["active_model"] = options.get("sd_model_checkpoint")
    except Exception as e:
        info["active_model_error"] = str(e)
    return info


def generate_batch(
    base_url: str,
    batch_size: int,
    seed: int,
    args,
) -> dict:
    """执行一次 batch_size=b 的 txt2img 调用, 返回延迟与实际图片数。"""
    payload = {
        "prompt": "a serene mountain lake at sunrise, photorealistic, fine details",
        "negative_prompt": "",
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "sampler_name": args.sampler,
        "seed": seed,
        "batch_size": batch_size,
        "n_iter": 1,
    }
    url = base_url.rstrip("/") + "/sdapi/v1/txt2img"
    start = time.perf_counter()
    with VRAMSampler() as sampler:
        resp = forge_post(url, payload, timeout=args.request_timeout)
    latency_s = time.perf_counter() - start
    n_images = len(resp.get("images", []))
    return {
        "latency_s": latency_s,
        "n_images": n_images,
        "vram_peak_mb": sampler.peak_mb(),
    }


def aggregate_results(raw_calls: dict, total_images: int) -> dict:
    """raw_calls: {batch_size: [ {latency_s, n_images, iteration, vram_peak_mb}, ... ]}"""
    aggregated = {}
    throughput_b1 = None
    for b in sorted(raw_calls.keys()):
        latencies = [c["latency_s"] for c in raw_calls[b]]
        if not latencies:
            continue
        vram_peaks = [c["vram_peak_mb"] for c in raw_calls[b] if c.get("vram_peak_mb") is not None]
        mean_t = statistics.mean(latencies)
        entry = {
            "batch_size": b,
            "n_samples": len(latencies),
            "calls_per_iteration": total_images // b,
            "T_mean_s": mean_t,
            "T_stdev_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "T_min_s": min(latencies),
            "T_max_s": max(latencies),
            "per_image_time_s": mean_t / b,
            "throughput_imgs_per_s": b / mean_t,
            "vram_peak_max_mb": max(vram_peaks) if vram_peaks else None,
            "vram_peak_mean_mb": statistics.mean(vram_peaks) if vram_peaks else None,
        }
        if b == 1:
            throughput_b1 = entry["throughput_imgs_per_s"]
        aggregated[b] = entry
    for b, entry in aggregated.items():
        if throughput_b1 and throughput_b1 > 0:
            entry["speedup_vs_b1"] = entry["throughput_imgs_per_s"] / throughput_b1
        else:
            entry["speedup_vs_b1"] = None
    return aggregated


def classify_batch_gain(aggregated: dict) -> str:
    """根据最大 speedup 给出三档结论之一。"""
    b1 = aggregated.get(1)
    if not b1:
        return "缺少 b=1 基线, 无法判断"
    best_b, best_speedup = None, 0.0
    for b, entry in aggregated.items():
        s = entry.get("speedup_vs_b1")
        if s and s > best_speedup:
            best_b, best_speedup = b, s
    if best_speedup is None or best_speedup <= 0:
        return "无法判断 (speedup 计算异常)"
    if best_speedup <= 1.10:
        return (
            f"batch 几乎无收益 (最大 speedup {best_speedup:.2f}x @ b={best_b}): "
            "首请求立即执行, 不考虑提前等待"
        )
    if best_speedup <= 1.50:
        return (
            f"batch 有一定收益但不大 (最大 speedup {best_speedup:.2f}x @ b={best_b}): "
            "队列有多个排队任务时可顺带批处理, 不主动等待"
        )
    return (
        f"batch 收益明显 (最大 speedup {best_speedup:.2f}x @ b={best_b}): "
        "值得进一步研究 micro-batching 等待窗口 (仍需结合请求到达间隔)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Forge API batch scaling 特性实验 (T(b) 曲线)"
    )
    parser.add_argument("--forge-base-url", type=str,
                        default=os.environ.get("XIAOYOU_FORGE_BASE_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--total-images", type=int, default=8,
                        help="每次迭代固定生成的总图片数 (默认 8)")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="逗号分隔的 batch size 列表, 必须能整除 total-images")
    parser.add_argument("--iterations", type=int, default=3,
                        help="重复迭代次数 (默认 3)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="预热调用次数 (b=1, 触发模型加载, 默认 1)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler", type=str, default="Euler")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子基址, 每次调用 seed = base + iter*1000 + b*100 + call")
    parser.add_argument("--cooldown-s", type=float, default=2.0,
                        help="相邻两次生成之间的冷却间隔秒数 (默认 2.0)")
    parser.add_argument("--request-timeout", type=float, default=1800.0,
                        help="单次 txt2img 请求超时秒数 (默认 1800)")
    args = parser.parse_args()

    batch_sizes = sorted({int(x) for x in args.batch_sizes.split(",") if x.strip()})
    for b in batch_sizes:
        if b < 1 or args.total_images % b != 0:
            print(f"错误: batch size {b} 无效 (必须 >=1 且整除 total-images={args.total_images})")
            sys.exit(1)

    print("=" * 60)
    print("Forge batch scaling 特性实验 (T(b) 曲线)")
    print("=" * 60)
    print(f"Forge API:      {args.forge_base_url}")
    print(f"总图片数/迭代:  {args.total_images}")
    print(f"batch sizes:    {batch_sizes}")
    print(f"迭代次数:       {args.iterations}")
    print(f"分辨率/steps:   {args.width}x{args.height} / {args.steps} steps / cfg {args.cfg_scale}")

    print("\n检查 Forge 服务...")
    if not check_forge_available(args.forge_base_url):
        print("  ❌ Forge 服务不可用！请先启动 Forge 服务再运行本脚本。")
        print("  启动命令: wsl -d Ubuntu-24.04-E -e bash -c 'cd /home/leslie/mvp_core/stable-diffusion-webui-forge-main && source /home/leslie/.venvs/forge-cu128/bin/activate && python launch.py --api --listen --port 7860 --skip-version-check --skip-python-version-check'")
        sys.exit(1)
    print("  ✅ Forge 服务可用")

    env_info = collect_env_info(args.forge_base_url)
    print(f"  当前模型: {env_info.get('active_model', '未知')}")

    if args.warmup > 0:
        print(f"\n[预热] 执行 {args.warmup} 次 b=1 生成 (触发模型加载, 不计入统计)...")
        for i in range(args.warmup):
            t0 = time.perf_counter()
            result = generate_batch(args.forge_base_url, 1, args.seed + 900000 + i, args)
            print(f"  预热 {i + 1}/{args.warmup}: {result['latency_s']:.2f}s "
                  f"(含加载, 墙钟 {time.perf_counter() - t0:.2f}s, "
                  f"返回 {result['n_images']} 张)")
            time.sleep(args.cooldown_s)

    # raw_calls[b] = [ {latency_s, n_images, iteration, vram_peak_mb}, ... ]
    raw_calls = {b: [] for b in batch_sizes}
    all_group_totals = {b: [] for b in batch_sizes}
    failed_batches = {}  # b -> 错误描述

    for iteration in range(args.iterations):
        # 交替正/逆序执行, 平衡 GPU 热漂移对某个 batch size 的系统性偏差
        order = batch_sizes if iteration % 2 == 0 else list(reversed(batch_sizes))
        print(f"\n{'=' * 60}")
        print(f"[迭代 {iteration + 1}/{args.iterations}] 执行顺序: {order}")
        print("=" * 60)

        for b in order:
            if b in failed_batches:
                continue
            calls = args.total_images // b
            group_start = time.perf_counter()
            group_failed = False
            for call_idx in range(calls):
                seed = args.seed + iteration * 1000 + b * 100 + call_idx
                try:
                    result = generate_batch(args.forge_base_url, b, seed, args)
                except urllib.error.HTTPError as e:
                    # 典型场景: 显存不足 (OOM), Forge 返回 500
                    body = ""
                    try:
                        body = e.read().decode("utf-8", errors="replace")[:300]
                    except Exception:
                        pass
                    msg = f"HTTP {e.code}: {body}"
                    print(f"  ❌ b={b} 调用 {call_idx + 1}/{calls} 失败: {msg}")
                    failed_batches[b] = msg
                    group_failed = True
                    break
                except Exception as e:
                    print(f"  ❌ b={b} 调用 {call_idx + 1}/{calls} 失败: {e}")
                    failed_batches[b] = str(e)
                    group_failed = True
                    break
                raw_calls[b].append({
                    "latency_s": result["latency_s"],
                    "n_images": result["n_images"],
                    "iteration": iteration + 1,
                    "vram_peak_mb": result.get("vram_peak_mb"),
                })
                if result["n_images"] != b:
                    print(f"  ⚠️ 请求 batch_size={b} 但返回 {result['n_images']} 张, Forge 可能截断了 batch")
                vram_str = ""
                if result.get("vram_peak_mb") is not None:
                    vram_str = f", VRAM峰值 {result['vram_peak_mb']}MB"
                print(f"  b={b}: 调用 {call_idx + 1}/{calls} "
                      f"T={result['latency_s']:.3f}s{vram_str}")
                if not (b == order[-1] and call_idx == calls - 1):
                    time.sleep(args.cooldown_s)
            if group_failed:
                print(f"  b={b} 已标记失败, 后续迭代将跳过该 batch size")
                continue
            group_total = time.perf_counter() - group_start
            all_group_totals[b].append(group_total)
            print(f"  b={b} 组完成: {calls} 次调用, 墙钟 {group_total:.3f}s "
                  f"(纯生成 {sum(c['latency_s'] for c in raw_calls[b][-calls:]):.3f}s)")
            time.sleep(args.cooldown_s)

    aggregated = aggregate_results(raw_calls, args.total_images)
    for b in batch_sizes:
        if b in aggregated and all_group_totals[b]:
            aggregated[b]["group_wall_time_mean_s"] = statistics.mean(all_group_totals[b])

    conclusion = classify_batch_gain(aggregated)
    if failed_batches:
        failed_str = "; ".join(f"b={b} 失败 ({msg.splitlines()[0][:80]})" for b, msg in sorted(failed_batches.items()))
        conclusion += f"。另有 batch size 被跳过: {failed_str} —— 在当前 GPU 显存下这本身就是有效结论 (最大可行 batch 受限)"

    print("\n" + "=" * 60)
    print("汇总 (每迭代固定生成 {} 张)".format(args.total_images))
    print("=" * 60)
    header = (f"{'b':>4} | {'T(b)均值':>10} | {'每图耗时':>10} | {'吞吐(img/s)':>12} | "
              f"{'加速比':>7} | {'VRAM峰值':>9} | {'组墙钟':>8}")
    print(header)
    print("-" * len(header))
    for b in batch_sizes:
        if b not in aggregated:
            print(f"{b:>4} | {'失败: ' + failed_batches[b].splitlines()[0][:70]}")
            continue
        e = aggregated[b]
        speedup = e["speedup_vs_b1"]
        vram = e.get("vram_peak_max_mb")
        vram_str = f"{vram}MB" if vram is not None else "n/a"
        wall = e.get("group_wall_time_mean_s")
        wall_str = f"{wall:7.2f}s" if wall is not None else "n/a"
        print(f"{b:>4} | {e['T_mean_s']:>9.3f}s | {e['per_image_time_s']:>9.3f}s | "
              f"{e['throughput_imgs_per_s']:>12.3f} | {speedup:>6.2f}x | "
              f"{vram_str:>9} | {wall_str:>8}")
    print(f"\n结论: {conclusion}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"batch_scaling_{timestamp}.json")
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "forge_batch_scaling",
        "env": env_info,
        "params": {
            "total_images": args.total_images,
            "batch_sizes": batch_sizes,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "sampler": args.sampler,
            "seed_base": args.seed,
            "cooldown_s": args.cooldown_s,
        },
        "aggregated": {str(b): aggregated[b] for b in batch_sizes if b in aggregated},
        "failed_batches": {str(b): failed_batches[b] for b in batch_sizes if b in failed_batches},
        "raw_calls": {str(b): raw_calls[b] for b in batch_sizes},
        "conclusion": conclusion,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_file}")


if __name__ == "__main__":
    main()
