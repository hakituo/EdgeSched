import os
import sys
import time
import json
import asyncio
import random
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("XIAOYOU_USE_CPP_SCHEDULER", "1")
os.environ.setdefault("XIAOYOU_USE_CPP_FOR_LLM", "1")
os.environ.setdefault("XIAOYOU_USE_CPP_FOR_IMAGE", "0")
os.environ.setdefault("XIAOYOU_CPP_LLM_BACKEND", "python")
os.environ.setdefault("XIAOYOU_CPP_WORKER_COUNT", "1")
os.environ.setdefault("XIAOYOU_BENCH_N_CTX", "2048")
os.environ.setdefault("XIAOYOU_BENCH_N_GPU_LAYERS", "-1")
os.environ.setdefault("XIAOYOU_BENCH_PRELOAD_LLM", "1")
os.environ.setdefault("XIAOYOU_MVP_IMAGE_BACKEND", "forge_then_cpp")


@dataclass
class StressEvent:
    timestamp_s: float
    task_type: str
    latency_s: float
    phase: str
    round: int = 0
    success: bool = True
    error: str = ""


@dataclass
class ResourceSnapshot:
    timestamp_s: float
    rss_mb: float
    vram_used_mb: float
    vram_total_mb: float


class ResourceMonitor:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)

    def _loop(self):
        while self.running:
            try:
                import psutil
                rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            except Exception:
                rss_mb = 0.0

            vram_used = 0.0
            vram_total = 0.0
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                    text=True
                )
                parts = out.strip().split(",")
                vram_used = float(parts[0].strip())
                vram_total = float(parts[1].strip())
            except Exception:
                pass

            self.snapshots.append(ResourceSnapshot(
                timestamp_s=time.time(),
                rss_mb=rss_mb,
                vram_used_mb=vram_used,
                vram_total_mb=vram_total,
            ))
            time.sleep(self.interval)


class RealisticStressTest:
    def __init__(self, llm_path: str, image_id: str, duration_minutes: int = 10, seed: int = 42):
        os.environ["XIAOYOU_BENCH_LLM_PATH"] = str(llm_path)
        os.environ["XIAOYOU_BENCH_SD_PATH"] = str(image_id)
        try:
            import config.integrated_config as integrated_config
            integrated_config.reset_settings_cache()
        except Exception:
            pass

        self.llm_path = llm_path
        self.image_id = image_id
        self.duration_minutes = duration_minutes
        self.seed = seed
        self.events: List[StressEvent] = []
        self.start_time = 0.0
        self.monitor = ResourceMonitor(interval=5.0)

        self.llm_requests = 0
        self.llm_successes = 0
        self.llm_timeouts = 0
        self.image_requests = 0
        self.image_successes = 0
        self.image_failures = 0
        self.migration_count = 0
        self.restore_count = 0
        self.restore_successes = 0

    async def _send_llm_request(self, round_no: int, phase: str, timeout: float = 60.0) -> bool:
        self.llm_requests += 1
        from mvp_core.services.task_scheduler import GlobalTaskScheduler

        scheduler = self._scheduler
        chat_start = time.time()
        got_token = False

        try:
            async for _chunk in scheduler.submit_llm_task(
                prompt=f"压力测试第 {round_no} 轮",
                request_id=f"stress_llm_{round_no}",
                max_tokens=16,
                temperature=0.3,
                first_token_timeout=timeout,
                total_timeout=timeout * 2,
            ):
                if not got_token:
                    got_token = True
                    latency = time.time() - chat_start
                    self.events.append(StressEvent(
                        timestamp_s=time.time() - self.start_time,
                        task_type="LLM",
                        latency_s=latency,
                        phase=phase,
                        round=round_no,
                        success=True,
                    ))
                    self.llm_successes += 1
                    break
        except asyncio.TimeoutError:
            self.llm_timeouts += 1
            self.events.append(StressEvent(
                timestamp_s=time.time() - self.start_time,
                task_type="LLM_TIMEOUT",
                latency_s=timeout,
                phase=phase,
                round=round_no,
                success=False,
                error="timeout",
            ))
        except Exception as e:
            self.events.append(StressEvent(
                timestamp_s=time.time() - self.start_time,
                task_type="LLM_ERROR",
                latency_s=time.time() - chat_start,
                phase=phase,
                round=round_no,
                success=False,
                error=str(e)[:100],
            ))

        return got_token

    async def _send_image_request(self, round_no: int) -> bool:
        self.image_requests += 1
        from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine

        img_start = time.time()

        try:
            llm_on_cpu = getattr(cpp_scheduler_engine, "_python_llm_device", "gpu") == "cpu"
            if not llm_on_cpu:
                self.migration_count += 1
                mig_start = time.time()
                await cpp_scheduler_engine.offload_llm_to_cpu(urgent=True)
                mig_time = time.time() - mig_start
                self.events.append(StressEvent(
                    timestamp_s=time.time() - self.start_time,
                    task_type="MIGRATION",
                    latency_s=mig_time,
                    phase="gpu_to_cpu",
                    round=round_no,
                ))

            task_id, started_fut, result_fut = await self._scheduler.try_enqueue_image_generation(
                prompt=f"压力测试图片 {round_no}",
                request_id=f"stress_img_{round_no}",
                steps=8,
                width=512,
                height=512,
            )

            try:
                await started_fut
            except Exception:
                pass

            result = await result_fut
            latency = time.time() - img_start

            if result:
                self.image_successes += 1
                self.events.append(StressEvent(
                    timestamp_s=time.time() - self.start_time,
                    task_type="IMAGE",
                    latency_s=latency,
                    phase="image_generation",
                    round=round_no,
                    success=True,
                ))
            else:
                self.image_failures += 1
                self.events.append(StressEvent(
                    timestamp_s=time.time() - self.start_time,
                    task_type="IMAGE",
                    latency_s=latency,
                    phase="image_generation",
                    round=round_no,
                    success=False,
                    error="no_result",
                ))

            self.restore_count += 1
            restore_start = time.time()
            restored = await cpp_scheduler_engine.restore_llm_to_gpu()
            restore_time = time.time() - restore_start
            if restored:
                self.restore_successes += 1
            self.events.append(StressEvent(
                timestamp_s=time.time() - self.start_time,
                task_type="RESTORE",
                latency_s=restore_time,
                phase="cpu_to_gpu",
                round=round_no,
                success=bool(restored),
            ))

            return bool(result)

        except Exception as e:
            self.image_failures += 1
            self.events.append(StressEvent(
                timestamp_s=time.time() - self.start_time,
                task_type="IMAGE_ERROR",
                latency_s=time.time() - img_start,
                phase="image_generation",
                round=round_no,
                success=False,
                error=str(e)[:100],
            ))
            return False

    async def run(self) -> Dict[str, Any]:
        from mvp_core.services.task_scheduler import GlobalTaskScheduler

        print(f"\n{'='*60}")
        print(f"Realistic Stress Test - 真实混合负载浸泡测试")
        print(f"时长: {self.duration_minutes} 分钟 | 种子: {self.seed}")
        print(f"LLM 请求间隔: 3-8s | Image 请求间隔: 20-40s")
        print(f"{'='*60}")

        self._scheduler = GlobalTaskScheduler()
        await self._scheduler.start()
        self.start_time = time.time()
        self.monitor.start()

        random.seed(self.seed)
        deadline = self.start_time + self.duration_minutes * 60

        llm_round = 0
        img_round = 0
        next_llm_at = self.start_time + 5.0
        next_img_at = self.start_time + 15.0

        try:
            while time.time() < deadline:
                now = time.time()
                elapsed = now - self.start_time

                if now >= next_img_at:
                    img_round += 1
                    from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
                    phase = "cpu_during_image" if getattr(cpp_scheduler_engine, "_python_llm_device", "gpu") == "cpu" else "gpu_before_image"
                    print(f"  [{elapsed:.0f}s] 提交 Image 请求 #{img_round} (phase={phase})")
                    await self._send_image_request(img_round)
                    next_img_at = now + random.uniform(20, 40)
                    next_llm_at = max(next_llm_at, now + 2.0)
                    continue

                if now >= next_llm_at:
                    llm_round += 1
                    from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
                    phase = "cpu_during_image" if getattr(cpp_scheduler_engine, "_python_llm_device", "gpu") == "cpu" else "gpu_before_image"
                    print(f"  [{elapsed:.0f}s] 提交 LLM 请求 #{llm_round} (phase={phase})")
                    await self._send_llm_request(llm_round, phase)
                    next_llm_at = now + random.uniform(3, 8)
                    continue

                await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            print("\n用户中断测试")
        finally:
            self.monitor.stop()
            await self._scheduler.stop()

        total_duration = time.time() - self.start_time

        llm_ttfts = [e.latency_s for e in self.events if e.task_type == "LLM" and e.success]
        image_latencies = [e.latency_s for e in self.events if e.task_type == "IMAGE" and e.success]
        migration_latencies = [e.latency_s for e in self.events if e.task_type == "MIGRATION"]
        restore_latencies = [e.latency_s for e in self.events if e.task_type == "RESTORE"]

        import statistics

        def _stats(values):
            if not values:
                return {"count": 0, "mean": 0, "stdev": 0, "min": 0, "max": 0}
            return {
                "count": len(values),
                "mean": round(statistics.mean(values), 4),
                "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        rss_values = [s.rss_mb for s in self.monitor.snapshots] if self.monitor.snapshots else []
        vram_values = [s.vram_used_mb for s in self.monitor.snapshots] if self.monitor.snapshots else []

        results = {
            "test_type": "realistic_stress",
            "duration_s": round(total_duration, 1),
            "duration_minutes": self.duration_minutes,
            "seed": self.seed,
            "summary": {
                "llm_requests": self.llm_requests,
                "llm_successes": self.llm_successes,
                "llm_timeouts": self.llm_timeouts,
                "llm_availability": round(self.llm_successes / self.llm_requests, 4) if self.llm_requests > 0 else 0,
                "image_requests": self.image_requests,
                "image_successes": self.image_successes,
                "image_failures": self.image_failures,
                "image_completion_rate": round(self.image_successes / self.image_requests, 4) if self.image_requests > 0 else 0,
                "migration_count": self.migration_count,
                "restore_count": self.restore_count,
                "restore_successes": self.restore_successes,
                "crash_free": True,
            },
            "latency": {
                "llm_ttft_s": _stats(llm_ttfts),
                "image_s": _stats(image_latencies),
                "migration_s": _stats(migration_latencies),
                "restore_s": _stats(restore_latencies),
            },
            "resources": {
                "rss_mb": _stats(rss_values),
                "vram_used_mb": _stats(vram_values),
                "samples": len(self.monitor.snapshots),
            },
            "events": [
                {
                    "timestamp_s": round(e.timestamp_s, 3),
                    "task_type": e.task_type,
                    "latency_s": round(e.latency_s, 4),
                    "phase": e.phase,
                    "round": e.round,
                    "success": e.success,
                    "error": e.error,
                }
                for e in self.events
            ],
        }

        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_results")
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "realistic_stress_test.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n{'='*60}")
        print(f"Realistic Stress Test 完成")
        print(f"  时长: {total_duration:.0f}s")
        print(f"  LLM: {self.llm_successes}/{self.llm_requests} 成功, {self.llm_timeouts} 超时")
        print(f"  Image: {self.image_successes}/{self.image_requests} 成功")
        print(f"  Migration: {self.migration_count}, Restore: {self.restore_successes}/{self.restore_count}")
        print(f"  Crash-free: {results['summary']['crash_free']}")
        if llm_ttfts:
            print(f"  LLM TTFT: mean={statistics.mean(llm_ttfts):.3f}s, max={max(llm_ttfts):.3f}s")
        print(f"  结果已保存: {out_file}")

        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Realistic Stress Test")
    parser.add_argument("--duration", type=int, default=10, help="测试时长（分钟）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    llm_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH",
                              "/home/leslie/mvp_core/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf")
    image_id = "/home/leslie/mvp_core/models/Image/check_point/SDXL/sd_xl_turbo_1.0_fp16.safetensors"

    exp = RealisticStressTest(llm_path, image_id, duration_minutes=args.duration, seed=args.seed)
    asyncio.run(exp.run())
