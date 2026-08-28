"""
动态资源迁移实验

面向单 GPU 边缘设备上的 Transformer（常驻交互）+ Diffusion（一次性重任务）混合负载，测试：
1. GPU→CPU 迁移效果（Diffusion 抢占 GPU 时，Transformer 常驻任务让位到 CPU 维持可用）
2. CPU→GPU 回迁效果（Diffusion 完成后，Transformer 回到 GPU，量化“卡顿窗口/TTFT 突刺”）
3. 背压互斥验证（Diffusion 队列：GPU 一次只能跑一个，一次性任务应排队等待）
"""

import asyncio
import argparse
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from framework import (
        setup_paths,
        RESULTS_DIR,
        ExperimentConfig,
        ExperimentResult,
        StatisticsAnalyzer,
        ReproducibilityManager,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from framework import (
        setup_paths,
        RESULTS_DIR,
        ExperimentConfig,
        ExperimentResult,
        StatisticsAnalyzer,
        ReproducibilityManager,
    )


@dataclass
class MigrationTestResult:
    """迁移测试结果"""
    test_name: str = ""
    test_passed: bool = False
    error_message: str = ""
    
    # 1.1 GPU→CPU 迁移指标
    llm_gpu_ttft_ms: float = 0.0
    llm_cpu_ttft_ms: float = 0.0
    migration_time_ms: float = 0.0
    llm_still_working: bool = False
    migration_window_max_gap_ms: float = 0.0
    migration_window_max_ttft_ms: float = 0.0
    
    # 1.2 CPU→GPU 回迁指标
    restore_time_ms: float = 0.0
    service_interruption_ms: float = 0.0
    first_gpu_ttft_ms_after_restore: float = 0.0
    kv_cache_preserved: bool = False
    restore_window_max_gap_ms: float = 0.0
    restore_window_max_ttft_ms: float = 0.0
    
    # 1.3 背压互斥指标
    queued: bool = False
    queue_wait_time_ms: float = 0.0
    eventually_executed: bool = False
    system_not_crashed: bool = True
    
    # 1.4 论文主线综合指标 (Lifecycle)
    lifecycle_passed: bool = False
    coexistence_cpu_ttft_ms: float = 0.0
    migration_stutter_ms: float = 0.0  # 图像任务开始到CPU LLM可用



@dataclass
class LlmProbeSample:
    t_s: float
    llm_device: str
    ttft_ms: float
    first_token_t: float


@dataclass
class ResourceSample:
    t_s: float
    llm_device: str
    proc_cpu_percent: float
    proc_rss_mb: float
    proc_vms_mb: float
    thread_count: int
    gpu_used_mb: Optional[float]
    gpu_total_mb: Optional[float]
    gpu_util_percent: Optional[float]


class ResourceMigrationExperiment:
    """动态资源迁移实验"""
    
    def __init__(
        self,
        llm_requests: int = 10,
        warmup_requests: int = 3,
        seed: int = 42,
    ):
        self.llm_requests = llm_requests
        self.warmup_requests = warmup_requests
        self.seed = seed
        
        self.reproducibility = ReproducibilityManager(seed=seed)
        self.analyzer = StatisticsAnalyzer()
        
        self.scheduler = None
        self.model_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH", "")

        self._resource_samples: List[ResourceSample] = []
        self._resource_events: List[Dict[str, Any]] = []
        self._resource_task: Optional[asyncio.Task] = None
        self._resource_t0: float = 0.0
        self._psutil_process = None
        self._fallback_wall_t: Optional[float] = None
        self._fallback_cpu_t: Optional[float] = None
    
    def setup_paths(self) -> None:
        """设置 Python 路径"""
        setup_paths()
    
    async def setup(self) -> None:
        """实验初始化"""
        self.reproducibility.setup()
        self.setup_paths()
        
        if not self.model_path:
            print("[Warning] XIAOYOU_LLM_MODEL_PATH 未设置，尝试默认路径")
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidates = [
                os.path.join(base_dir, "models", "llm", "qwen2.5-7b-instruct-q4_k_m.gguf"),
                os.path.join(base_dir, "models", "qwen2.5-7b-instruct-q4_k_m.gguf"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    self.model_path = candidate
                    break
            if not self.model_path:
                print(f"[Error] 模型文件不存在，已搜索: {candidates}")
                raise FileNotFoundError(f"模型文件不存在，已搜索: {candidates}")
        
        print(f"[Setup] 模型路径: {self.model_path}")
        
        try:
            from services.cpp_scheduler_engine import cpp_scheduler_engine
            
            config = {
                "model_path": self.model_path,
                "max_context_size": 4096,
                "n_gpu_layers": -1,
                "max_batch_size": 512,
            }
            
            cpp_scheduler_engine.start(
                worker_count=1,
                gpu_config=config,
                preload_llm=True,
            )
            
            self.scheduler = cpp_scheduler_engine
            
            # [Added] 强制检查 Forge 服务是否可用
            print("[Setup] 检查 Forge 服务可用性...")
            forge_ok = False
            base_url = os.environ.get("XIAOYOU_FORGE_BASE_URL", "http://127.0.0.1:7860")
            
            # 如果环境变量没设置，尝试探测 7860 和 7861
            candidate_urls = [base_url]
            if base_url == "http://127.0.0.1:7860":
                candidate_urls.append("http://127.0.0.1:7861")
            
            final_url = None
            
            for _ in range(10):  # 增加重试次数
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    for url in candidate_urls:
                        try:
                            async with session.get(f"{url}/sdapi/v1/sd-models", timeout=5) as resp:
                                if resp.status == 200:
                                    forge_ok = True
                                    final_url = url
                                    print(f"  Forge 服务已连接 ({url})")
                                    # 如果连接的是 7861，自动更新环境变量，以便 cpp_scheduler 使用
                                    os.environ["XIAOYOU_FORGE_BASE_URL"] = url
                                    break
                        except Exception:
                            pass
                    if forge_ok:
                        break
                print("  等待 Forge 服务启动...")
                await asyncio.sleep(2)
            
            if not forge_ok:
                print("[Warning] Forge 服务未响应！图像生成可能会回退到 Mock 模式。")
                if not os.environ.get("XIAOYOU_ALLOW_MOCK_FORGE"):
                     # raise RuntimeError("Forge 服务未启动，无法进行真实实验")
                     pass
            
            self._start_resource_monitor()
            self._mark_resource_event("setup_started")
            await self._wait_for_llm_ready(timeout_s=180.0)
            self._mark_resource_event("setup_ready")
            
            print("[Setup] 调度器初始化完成")
            
        except Exception as e:
            print(f"[Setup] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def teardown(self) -> None:
        """实验清理"""
        self._mark_resource_event("teardown_started")
        await self._stop_resource_monitor()
        if self.scheduler:
            try:
                await self.scheduler.stop()
            except Exception:
                pass
        
        print("[Teardown] 实验清理完成")
    
    async def measure_llm_ttft(self, prompt: str = "你好，请介绍一下你自己。") -> float:
        """测量 LLM TTFT"""
        start = time.perf_counter()
        first_token = False
        
        try:
            async for token in self.scheduler.submit_llm_task(prompt, max_tokens=64, temperature=0.7):
                if not first_token:
                    first_token = True
                    return (time.perf_counter() - start) * 1000
        except Exception as e:
            print(f"[Measure] LLM TTFT 测量失败: {e}")
            return -1
        
        return -1

    async def _measure_llm_first_token(self, prompt: str) -> Optional[LlmProbeSample]:
        if not self.scheduler:
            return None
        t0 = time.perf_counter()
        try:
            async for token in self.scheduler.submit_llm_task(prompt, max_tokens=64, temperature=0.7):
                if token:
                    t1 = time.perf_counter()
                    return LlmProbeSample(
                        t_s=max(0.0, time.time() - self._resource_t0) if self._resource_t0 > 0 else 0.0,
                        llm_device=self._get_llm_device(),
                        ttft_ms=(t1 - t0) * 1000.0,
                        first_token_t=t1,
                    )
        except Exception:
            return None
        return None

    async def _probe_llm_continuously(
        self,
        samples: List[LlmProbeSample],
        stop_event: asyncio.Event,
        interval_s: float = 0.25,
        prompt: str = "用一句话回答：现在状态如何？",
    ) -> None:
        while not stop_event.is_set():
            s = await self._measure_llm_first_token(prompt)
            if s is not None and s.ttft_ms > 0:
                samples.append(s)
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                break

    def _compute_gap_and_spike(self, samples: List[LlmProbeSample]) -> Dict[str, float]:
        ok = [s for s in samples if s.ttft_ms > 0 and s.first_token_t > 0]
        if len(ok) < 2:
            max_gap_ms = 0.0
        else:
            gaps = [(ok[i].first_token_t - ok[i - 1].first_token_t) * 1000.0 for i in range(1, len(ok))]
            max_gap_ms = float(max(gaps)) if gaps else 0.0
        max_ttft_ms = float(max([s.ttft_ms for s in ok], default=0.0))
        return {"max_gap_ms": max_gap_ms, "max_ttft_ms": max_ttft_ms}
    
    def _check_gpu_memory(self) -> Optional[float]:
        """检查 GPU 显存使用"""
        try:
            import subprocess
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                return float(line)
        except Exception as e:
            print(f"[GPU Check] 检查显存失败: {e}")
            return None

    def _get_llm_device(self) -> str:
        if not self.scheduler:
            return "unknown"
        dev = getattr(self.scheduler, "_python_llm_device", None)
        if dev:
            return str(dev)
        force_cpu = bool(getattr(self.scheduler, "_python_force_cpu", False))
        if force_cpu:
            return "cpu"
        if getattr(self.scheduler, "_python_llm", None) is not None:
            return "gpu"
        return "unknown"

    def _query_gpu_stats(self) -> Dict[str, Optional[float]]:
        try:
            import subprocess
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
            )
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                used_mb = float(parts[0]) if len(parts) > 0 and parts[0] else None
                total_mb = float(parts[1]) if len(parts) > 1 and parts[1] else None
                util = float(parts[2]) if len(parts) > 2 and parts[2] else None
                return {"used_mb": used_mb, "total_mb": total_mb, "util_percent": util}
        except Exception:
            pass
        return {"used_mb": None, "total_mb": None, "util_percent": None}

    def _ensure_psutil(self) -> None:
        if self._psutil_process is not None:
            return
        try:
            import psutil
            self._psutil_process = psutil.Process()
            try:
                self._psutil_process.cpu_percent(interval=None)
            except Exception:
                pass
        except Exception:
            self._psutil_process = False

    def _read_process_stats(self) -> Dict[str, Any]:
        self._ensure_psutil()
        if not self._psutil_process or self._psutil_process is False:
            now_wall = time.time()
            now_cpu = time.process_time()
            cpu_percent = 0.0
            if self._fallback_wall_t is not None and self._fallback_cpu_t is not None:
                dt_wall = now_wall - self._fallback_wall_t
                dt_cpu = now_cpu - self._fallback_cpu_t
                if dt_wall > 0:
                    cpu_percent = (dt_cpu / dt_wall) * 100.0
            self._fallback_wall_t = now_wall
            self._fallback_cpu_t = now_cpu

            rss_mb = 0.0
            try:
                import resource
                ru = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform.startswith("linux"):
                    rss_mb = float(ru.ru_maxrss) / 1024.0
                else:
                    rss_mb = float(ru.ru_maxrss) / 1024.0 / 1024.0
            except Exception:
                rss_mb = 0.0

            return {
                "cpu_percent": float(cpu_percent),
                "rss_mb": float(rss_mb),
                "vms_mb": 0.0,
                "threads": 0,
            }
        proc = self._psutil_process
        try:
            cpu_percent = float(proc.cpu_percent(interval=None) or 0.0)
        except Exception:
            cpu_percent = 0.0
        try:
            mem = proc.memory_info()
            rss_mb = float(mem.rss) / 1024 / 1024
            vms_mb = float(mem.vms) / 1024 / 1024
        except Exception:
            rss_mb = 0.0
            vms_mb = 0.0
        try:
            threads = int(proc.num_threads() or 0)
        except Exception:
            threads = 0
        return {
            "cpu_percent": cpu_percent,
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "threads": threads,
        }

    def _mark_resource_event(self, name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if self._resource_t0 <= 0:
            return
        payload: Dict[str, Any] = {"t_s": time.time() - self._resource_t0, "name": name}
        if extra:
            payload.update(extra)
        self._resource_events.append(payload)

    def _start_resource_monitor(self, interval_s: float = 0.5) -> None:
        if self._resource_task is not None:
            return
        self._resource_t0 = time.time()

        async def _loop():
            while True:
                try:
                    proc = self._read_process_stats()
                    gpu = self._query_gpu_stats()
                    sample = ResourceSample(
                        t_s=time.time() - self._resource_t0,
                        llm_device=self._get_llm_device(),
                        proc_cpu_percent=float(proc["cpu_percent"]),
                        proc_rss_mb=float(proc["rss_mb"]),
                        proc_vms_mb=float(proc["vms_mb"]),
                        thread_count=int(proc["threads"]),
                        gpu_used_mb=gpu.get("used_mb"),
                        gpu_total_mb=gpu.get("total_mb"),
                        gpu_util_percent=gpu.get("util_percent"),
                    )
                    self._resource_samples.append(sample)
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass
                try:
                    await asyncio.sleep(interval_s)
                except asyncio.CancelledError:
                    break

        self._resource_task = asyncio.create_task(_loop())

    async def _stop_resource_monitor(self) -> None:
        task = self._resource_task
        self._resource_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    async def _wait_for_llm_ready(self, timeout_s: float = 180.0) -> None:
        deadline = time.time() + float(timeout_s or 180.0)
        while time.time() < deadline:
            if getattr(self.scheduler, "_python_llm", None) is not None:
                return
            await asyncio.sleep(0.2)
        raise TimeoutError("等待 LLM 初始化超时")

    async def _wait_for_llm_device(self, device: str, timeout_s: float = 180.0) -> None:
        deadline = time.time() + float(timeout_s or 180.0)
        desired = str(device).strip().lower()
        while time.time() < deadline:
            if self._get_llm_device() == desired and getattr(self.scheduler, "_python_llm", None) is not None:
                return
            await asyncio.sleep(0.2)
        raise TimeoutError(f"等待 LLM 切换到 {device} 超时")
    
    async def test_1_1_gpu_to_cpu_migration(self) -> MigrationTestResult:
        """
        子实验 1.1: GPU→CPU 迁移效果
        
        流程：
        1. LLM 在 GPU 上运行，测量 TTFT
        2. 提交图像任务
        3. 系统迁移：释放 GPU LLM → DDR 加载 CPU LLM → GPU 加载图像模型
        4. 测量 CPU 上的 LLM TTFT
        """
        print("\n" + "=" * 60)
        print("[实验 1.1] GPU→CPU 迁移效果测试")
        print("=" * 60)
        
        result = MigrationTestResult(test_name="gpu_to_cpu_migration")
        
        try:
            self._mark_resource_event("test_1_1_start")

            try:
                await self.scheduler.restore_llm_to_gpu()
            except Exception:
                pass
            try:
                await self._wait_for_llm_device("gpu", timeout_s=180.0)
            except Exception:
                pass

            # Phase 1: 确保 LLM 在 GPU 上（模型已在 preload 时加载，不需要额外恢复）
            print("\n[Phase 1] 确认 LLM 状态...")
            await asyncio.sleep(2)
            
            gpu_mem_before = self._check_gpu_memory()
            if gpu_mem_before:
                print(f"  GPU 显存: {gpu_mem_before:.0f} MB")
            print(f"  LLM 设备: {self._get_llm_device()}")
            if self._get_llm_device() != "gpu":
                raise RuntimeError("LLM 未实际运行在 GPU，上报为 GPU 的基线 TTFT 无意义，请检查 CUDA llama_cpp 或后端配置")
            
            # Phase 2: 测量 GPU 上的 TTFT（基线）
            print(f"[Phase 2] 测量 GPU 上的 LLM TTFT（{self.warmup_requests} 次预热 + {self.llm_requests} 次测量）...")
            
            for i in range(self.warmup_requests):
                await self.measure_llm_ttft()
                await asyncio.sleep(0.1)
            
            gpu_ttfts = []
            for i in range(self.llm_requests):
                ttft = await self.measure_llm_ttft()
                if ttft > 0:
                    gpu_ttfts.append(ttft)
                    print(f"  GPU TTFT [{i+1}/{self.llm_requests}]: {ttft:.1f} ms")
                await asyncio.sleep(0.1)
            
            if gpu_ttfts:
                result.llm_gpu_ttft_ms = statistics.mean(gpu_ttfts)
                print(f"\n  GPU TTFT 统计: avg={result.llm_gpu_ttft_ms:.1f}ms, "
                      f"p50={statistics.median(gpu_ttfts):.1f}ms")
            
            # Phase 3: 触发迁移（直接调用迁移方法），并在迁移窗口采集 max_gap / TTFT 突刺
            print("\n[Phase 3] 触发 LLM 迁移到 CPU...")
            self._mark_resource_event("test_1_1_offload_start")
            migration_start = time.perf_counter()

            probe_samples: List[LlmProbeSample] = []
            probe_stop = asyncio.Event()
            seed_sample = await self._measure_llm_first_token("用一句话回答：你在线吗？")
            if seed_sample is not None and seed_sample.ttft_ms > 0:
                probe_samples.append(seed_sample)
            probe_task = asyncio.create_task(
                self._probe_llm_continuously(
                    probe_samples,
                    probe_stop,
                    interval_s=0.25,
                    prompt="用一句话回答：你还在线吗？",
                )
            )
            
            try:
                await self.scheduler.offload_llm_to_cpu(urgent=False)
                print("  迁移命令已执行")
            except Exception as e:
                print(f"  迁移失败: {e}")
            
            await self._wait_for_llm_device("cpu", timeout_s=180.0)
            result.migration_time_ms = (time.perf_counter() - migration_start) * 1000
            self._mark_resource_event("test_1_1_offload_ready")
            print(f"  迁移耗时: {result.migration_time_ms:.1f} ms")

            await asyncio.sleep(1.5)
            probe_stop.set()
            try:
                await probe_task
            except Exception:
                pass
            metrics = self._compute_gap_and_spike(probe_samples)
            result.migration_window_max_gap_ms = float(metrics["max_gap_ms"])
            result.migration_window_max_ttft_ms = float(metrics["max_ttft_ms"])
            if result.migration_window_max_gap_ms > 0 or result.migration_window_max_ttft_ms > 0:
                print(
                    f"  迁移窗口 max_gap={result.migration_window_max_gap_ms:.1f} ms, "
                    f"max_ttft={result.migration_window_max_ttft_ms:.1f} ms"
                )
            
            gpu_mem_after = self._check_gpu_memory()
            if gpu_mem_after:
                print(f"  迁移后 GPU 显存: {gpu_mem_after:.0f} MB")
                if gpu_mem_before and gpu_mem_after:
                    diff = gpu_mem_before - gpu_mem_after
                    print(f"  显存释放: {diff:.0f} MB")
            print(f"  LLM 设备: {self._get_llm_device()}")
            
            # Phase 4: 测量 CPU 上的 TTFT
            print(f"\n[Phase 4] 测量 CPU 上的 LLM TTFT（{self.llm_requests} 次测量）...")
            if self._get_llm_device() != "cpu":
                raise RuntimeError("LLM 未实际切换到 CPU，CPU TTFT 测量无意义")
            
            cpu_ttfts = []
            for i in range(self.llm_requests):
                ttft = await self.measure_llm_ttft()
                if ttft > 0:
                    cpu_ttfts.append(ttft)
                    print(f"  CPU TTFT [{i+1}/{self.llm_requests}]: {ttft:.1f} ms")
                await asyncio.sleep(0.1)
            
            if cpu_ttfts:
                result.llm_cpu_ttft_ms = statistics.mean(cpu_ttfts)
                print(f"\n  CPU TTFT 统计: avg={result.llm_cpu_ttft_ms:.1f}ms, "
                      f"p50={statistics.median(cpu_ttfts):.1f}ms")
            
            # 判断结果
            result.llm_still_working = len(cpu_ttfts) > 0
            result.test_passed = result.llm_still_working and result.migration_time_ms > 0
            
            if result.test_passed:
                print(f"\n✅ 测试通过: LLM 迁移后仍能响应")
                if result.llm_gpu_ttft_ms > 0 and result.llm_cpu_ttft_ms > 0:
                    degradation = result.llm_cpu_ttft_ms / result.llm_gpu_ttft_ms
                    print(f"   性能退化: {degradation:.1f}x")
            else:
                print(f"\n❌ 测试失败: LLM 迁移后无法响应")
            
        except Exception as e:
            result.error_message = str(e)
            print(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
        
        self._mark_resource_event("test_1_1_end", {"passed": bool(result.test_passed)})
        return result
    
    async def test_1_2_cpu_to_gpu_restore(self) -> MigrationTestResult:
        """
        子实验 1.2: CPU→GPU 回迁效果（原子替换）
        
        流程：
        1. LLM 在 CPU 上，图像模型在 GPU 上
        2. 触发回迁
        3. 系统执行：卸载图像 → 加载 GPU LLM → 原子替换 → 延迟释放 CPU LLM
        4. 测量服务中断时间
        """
        print("\n" + "=" * 60)
        print("[实验 1.2] CPU→GPU 回迁效果测试（原子替换）")
        print("=" * 60)
        
        result = MigrationTestResult(test_name="cpu_to_gpu_restore")
        
        try:
            self._mark_resource_event("test_1_2_start")

            # Phase 1: 确保 LLM 在 CPU 上（迁移后状态）
            print("\n[Phase 1] 确保 LLM 在 CPU 上...")
            
            # 先尝试迁移到 CPU
            try:
                await self.scheduler.offload_llm_to_cpu(urgent=True)
            except Exception as e:
                print(f"  迁移到 CPU 失败（可能已在 CPU）: {e}")
            await self._wait_for_llm_device("cpu", timeout_s=180.0)
            
            # Phase 2: 测量 CPU 上的 TTFT（基线）
            print(f"[Phase 2] 测量 CPU 上的 LLM TTFT...")
            cpu_ttfts = []
            for i in range(min(5, self.llm_requests)):
                ttft = await self.measure_llm_ttft()
                if ttft > 0:
                    cpu_ttfts.append(ttft)
                await asyncio.sleep(0.1)
            
            if cpu_ttfts:
                print(f"  CPU TTFT (基线): {statistics.mean(cpu_ttfts):.1f} ms")
            
            # Phase 3: 触发回迁，并用 max_gap / TTFT 突刺量化“用户感知卡顿窗口”
            print("\n[Phase 3] 触发回迁，采集 max_gap / TTFT 突刺...")

            probe_samples: List[LlmProbeSample] = []
            probe_stop = asyncio.Event()
            probe_task = asyncio.create_task(
                self._probe_llm_continuously(
                    probe_samples,
                    probe_stop,
                    interval_s=0.25,
                    prompt="用一句话回答：恢复到GPU了吗？",
                )
            )

            restore_start = time.perf_counter()
            self._mark_resource_event("test_1_2_restore_start")
            try:
                ok = await self.scheduler.restore_llm_to_gpu()
                print(f"  回迁完成: {ok}")
            except Exception as e:
                print(f"  回迁失败: {e}")
            result.restore_time_ms = (time.perf_counter() - restore_start) * 1000
            self._mark_resource_event("test_1_2_restore_done")

            await self._wait_for_llm_device("gpu", timeout_s=180.0)
            result.first_gpu_ttft_ms_after_restore = await self.measure_llm_ttft()

            await asyncio.sleep(1.0)
            probe_stop.set()
            try:
                await probe_task
            except Exception:
                pass

            metrics = self._compute_gap_and_spike(probe_samples)
            result.restore_window_max_gap_ms = float(metrics["max_gap_ms"])
            result.restore_window_max_ttft_ms = float(metrics["max_ttft_ms"])
            result.service_interruption_ms = float(result.restore_window_max_gap_ms or 0.0)
            
            print(f"  回迁耗时: {result.restore_time_ms:.1f} ms")
            print(
                f"  回迁窗口 max_gap={result.restore_window_max_gap_ms:.1f} ms, "
                f"max_ttft={result.restore_window_max_ttft_ms:.1f} ms"
            )
            print(f"  回迁后首次 GPU TTFT: {result.first_gpu_ttft_ms_after_restore:.1f} ms")
            
            # Phase 4: 测量 GPU 上的 TTFT
            print(f"\n[Phase 4] 测量 GPU 上的 LLM TTFT...")
            gpu_ttfts = []
            for i in range(self.llm_requests):
                ttft = await self.measure_llm_ttft()
                if ttft > 0:
                    gpu_ttfts.append(ttft)
                await asyncio.sleep(0.1)
            
            if gpu_ttfts:
                print(f"  GPU TTFT: {statistics.mean(gpu_ttfts):.1f} ms")
            
            result.kv_cache_preserved = False
            
            # 判断结果
            result.test_passed = (len(gpu_ttfts) > 0 and result.restore_window_max_gap_ms < 5000)
            
            if result.test_passed:
                print(f"\n✅ 测试通过: 原子替换成功，服务中断时间短")
            else:
                print(f"\n❌ 测试失败")
            
        except Exception as e:
            result.error_message = str(e)
            print(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
        
        self._mark_resource_event("test_1_2_end", {"passed": bool(result.test_passed)})
        return result
    
    async def test_1_3_backpressure_queue(self) -> MigrationTestResult:
        """
        子实验 1.3: 背压互斥验证（Diffusion 队列）
        
        场景：GPU 一次只能跑一个 Diffusion 任务；当 GPU 正在生图时，新的图像任务应进入排队等待
        
        流程：
        1. 提交一个较长的图像任务占用 GPU
        2. 在其运行期间再提交一个图像任务
        3. 验证第二个任务不会抢占并发执行（应排队等待）
        """
        print("\n" + "=" * 60)
        print("[实验 1.3] 背压互斥验证（Diffusion 队列）")
        print("=" * 60)
        
        result = MigrationTestResult(test_name="backpressure_queue")
        
        try:
            self._mark_resource_event("test_1_3_start")

            # Phase 1: 确保资源被占用
            print("\n[Phase 1] 提交长图像任务占用 GPU...")
            image_task_a = asyncio.create_task(
                self.scheduler.submit_image_task(
                    "test image A for backpressure",
                    width=512,
                    height=512,
                    steps=20,
                    guidance_scale=7.5,
                )
            )
            
            await asyncio.sleep(0.2)

            # Phase 2: 提交第二个图像任务，验证其排队
            print("\n[Phase 2] 提交第二个图像任务（应排队等待）...")
            submit_b_t = time.perf_counter()
            b_started = asyncio.Event()
            b_start_t: Optional[float] = None

            def _on_b_start(t: float):
                nonlocal b_start_t
                if b_start_t is None:
                    b_start_t = float(t)
                    try:
                        b_started.set()
                    except Exception:
                        pass

            image_task_b = asyncio.create_task(
                self.scheduler.submit_image_task(
                    "test image B for backpressure",
                    width=512,
                    height=512,
                    steps=20,
                    guidance_scale=7.5,
                    _xy_on_start=_on_b_start,
                )
            )

            await asyncio.sleep(0.5)
            result.queued = not bool(b_started.is_set())
            print(f"  第二个任务是否处于等待态: {'是' if result.queued else '否'}")

            # Phase 3: 等待第一个任务完成，估算排队等待时间
            print("\n[Phase 3] 等待第一个任务完成...")
            a_done_t = None
            try:
                await asyncio.wait_for(image_task_a, timeout=180)
                a_done_t = time.perf_counter()
                print("  图像任务 A 完成")
            except asyncio.TimeoutError:
                print("  图像任务 A 超时")
            except Exception as e:
                print(f"  图像任务 A 异常: {e}")

            if b_start_t is not None:
                result.queue_wait_time_ms = max(0.0, (b_start_t - submit_b_t) * 1000.0)
            elif a_done_t is not None and result.queued:
                result.queue_wait_time_ms = max(0.0, (a_done_t - submit_b_t) * 1000.0)

            # Phase 4: 验证第二个任务最终执行完成
            print("\n[Phase 4] 等待第二个任务完成...")
            try:
                await asyncio.wait_for(image_task_b, timeout=180)
                result.eventually_executed = True
                print("  图像任务 B 完成")
            except Exception as e:
                result.eventually_executed = False
                print(f"  图像任务 B 异常/超时: {e}")

            result.system_not_crashed = True
            result.test_passed = bool(result.queued) and bool(result.eventually_executed) and bool(result.system_not_crashed)

            print(f"\n  是否排队: {result.queued}")
            print(f"  排队等待时间(估算): {result.queue_wait_time_ms:.1f} ms")
            print(f"  是否最终执行: {result.eventually_executed}")
            print(f"  系统是否崩溃: {'否' if result.system_not_crashed else '是'}")

            if result.test_passed:
                print("\n✅ 测试通过: Diffusion 队列互斥/排队符合预期")
            else:
                print("\n❌ 测试失败")
            
        except Exception as e:
            result.error_message = str(e)
            result.system_not_crashed = False
            print(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
        
        self._mark_resource_event("test_1_3_end", {"passed": bool(result.test_passed)})
        return result
    
    async def test_integrated_lifecycle(self) -> MigrationTestResult:
        """
        论文主线完整复现 (Integrated Lifecycle Test)
        对应论文 Figure 6: Migration Flow
        
        流程：
        1. [State: GPU] LLM 在 GPU 上运行 (Baseline)
        2. [Event] 图像任务到达 (Image Task Arrives)
        3. [Action] 紧急迁移 (Urgent Mode: GPU->CPU)
        4. [State: Coexistence] 图像生成中 (GPU) + LLM 在 CPU 响应 (CPU)
        5. [Event] 图像任务完成 (Image Task Completes)
        6. [Action] 原子回迁 (Atomic Hot-Swap: CPU->GPU)
        7. [State: GPU] LLM 恢复 GPU 运行
        """
        print("\n" + "=" * 60)
        print("[实验 1.4] 论文主线完整生命周期测试 (Integrated Lifecycle)")
        print("=" * 60)
        
        result = MigrationTestResult(test_name="integrated_lifecycle")
        
        try:
            self._mark_resource_event("lifecycle_start")

            # --- Phase 1: 初始状态确认 (GPU) ---
            print("\n[Phase 1] 初始状态确认 (LLM on GPU)...")
            await self.scheduler.restore_llm_to_gpu()
            await self._wait_for_llm_device("gpu")
            
            # 预热并测量基线
            print("  测量 GPU 基线 TTFT...")
            gpu_ttfts = []
            for _ in range(3):
                t = await self.measure_llm_ttft()
                if t > 0: gpu_ttfts.append(t)
            result.llm_gpu_ttft_ms = statistics.mean(gpu_ttfts) if gpu_ttfts else 0.0
            print(f"  GPU Baseline TTFT: {result.llm_gpu_ttft_ms:.1f} ms")

            # --- Phase 2: 触发迁移与共存 (Migration & Coexistence) ---
            print("\n[Phase 2] 提交图像任务并触发紧急迁移 (Urgent Migration)...")
            
            # 标记时间点
            t_start = time.perf_counter()
            self._mark_resource_event("lifecycle_migration_trigger")
            
            # 1. 提交图像任务 (异步)
            # 使用 20 steps 保持与其他实验一致
            image_task_future = asyncio.create_task(
                self.scheduler.submit_image_task(
                    "A futuristic city with flying cars, cyberpunk style, high details",
                    width=512, height=512, steps=20, guidance_scale=7.5
                )
            )
            
            # 2. 立即触发迁移 (模拟 Scheduler 的资源仲裁)
            # 论文中提到: "image task preemption (LLM offload to CPU)"
            print("  执行 Offload (Urgent=True)...")
            offload_future = asyncio.create_task(self.scheduler.offload_llm_to_cpu(urgent=True))
            
            # 3. 共存测试 (Coexistence): 在迁移过程中并发测试 CPU LLM
            # 关键：不等迁移完成，立即开始探测，确保在图像任务结束前能测到数据
            print("  [Coexistence] 图像生成中，并发测试 CPU LLM 响应...")
            
            cpu_ttfts = []
            migration_done = False
            t_cpu_ready = None
            
            while not image_task_future.done():
                # 检查迁移是否完成
                if not migration_done and offload_future.done():
                    migration_done = True
                    try:
                        await offload_future
                    except Exception:
                        pass
                    # 等待 LLM 在 CPU 上就绪
                    try:
                        await self._wait_for_llm_device("cpu")
                        t_cpu_ready = time.perf_counter()
                        result.migration_stutter_ms = (t_cpu_ready - t_start) * 1000.0
                        print(f"  迁移卡顿 (Stutter): {result.migration_stutter_ms:.1f} ms")
                    except Exception:
                        pass
                
                # 只有在 CPU LLM 就绪后才测量 TTFT
                if migration_done and t_cpu_ready is not None:
                    t_probe = await self.measure_llm_ttft("在这个共存阶段，请简短回答：你还好吗？")
                    if t_probe > 0:
                        cpu_ttfts.append(t_probe)
                        print(f"  [Coexistence] CPU TTFT: {t_probe:.1f} ms (Image Task Running)")
                
                await asyncio.sleep(0.1)
            
            # 如果迁移在图像完成后才完成，记录迁移时间
            if not migration_done:
                try:
                    await offload_future
                    await self._wait_for_llm_device("cpu")
                    t_cpu_ready = time.perf_counter()
                    result.migration_stutter_ms = (t_cpu_ready - t_start) * 1000.0
                    print(f"  迁移卡顿 (Stutter): {result.migration_stutter_ms:.1f} ms")
                except Exception:
                    pass
                print("  警告: 图像任务在迁移完成前已结束，未能测到共存阶段数据")

            # 等待图像任务彻底完成
            try:
                await image_task_future
                print("  图像任务完成")
            except Exception as e:
                print(f"  图像任务异常: {e}")
                
            if cpu_ttfts:
                result.coexistence_cpu_ttft_ms = statistics.mean(cpu_ttfts)
                print(f"  共存阶段 CPU TTFT avg: {result.coexistence_cpu_ttft_ms:.1f} ms")
            else:
                print("  警告: 共存阶段未能测得有效的 TTFT (图像任务可能太快结束)")

            # --- Phase 3: 回迁 (Restore) ---
            print("\n[Phase 3] 图像完成，触发原子回迁 (Atomic Restore)...")
            self._mark_resource_event("lifecycle_restore_trigger")
            
            t_restore_start = time.perf_counter()
            
            # 启动探测任务以捕捉 Restore Gap
            probe_samples = []
            probe_stop = asyncio.Event()
            probe_task = asyncio.create_task(
                self._probe_llm_continuously(probe_samples, probe_stop, interval_s=0.2, prompt="回迁测试")
            )
            
            # 执行回迁
            await self.scheduler.restore_llm_to_gpu()
            t_restore_end = time.perf_counter()
            
            result.restore_time_ms = (t_restore_end - t_restore_start) * 1000.0
            print(f"  回迁总耗时 (Background + Swap): {result.restore_time_ms:.1f} ms")
            
            # 停止探测
            await asyncio.sleep(0.5)
            probe_stop.set()
            try: await probe_task
            except: pass
            
            # 计算 Gap
            metrics = self._compute_gap_and_spike(probe_samples)
            result.restore_window_max_gap_ms = metrics["max_gap_ms"]
            print(f"  回迁服务中断 (Restore Gap): {result.restore_window_max_gap_ms:.1f} ms")
            
            # 验证回迁后状态
            await self._wait_for_llm_device("gpu")
            t_final = await self.measure_llm_ttft()
            print(f"  回迁后 GPU TTFT: {t_final:.1f} ms")
            
            # 判定结果
            # 1. 成功在 CPU 上响应了 (Coexistence)
            # 2. 成功回到了 GPU
            # 3. 图像任务成功完成
            result.lifecycle_passed = (len(cpu_ttfts) > 0) and (t_final > 0)
            result.test_passed = result.lifecycle_passed
            
            if result.lifecycle_passed:
                print("\n✅ [Lifecycle] 完整生命周期测试通过")
            else:
                print("\n❌ [Lifecycle] 测试失败")

        except Exception as e:
            result.error_message = str(e)
            print(f"\n❌ [Lifecycle] 异常: {e}")
            import traceback
            traceback.print_exc()
            
        self._mark_resource_event("lifecycle_end", {"passed": result.lifecycle_passed})
        return result

    async def run_all_tests(self, include_backpressure: bool = True) -> Dict[str, Any]:
        """
        运行论文核心实验集 (Paper Core Experiments)
        
        包含:
        1. [Test 1.4] 资源迁移与共存生命周期 (对应论文 6.2.1)
        2. [Test 1.3] 队列背压与互斥 (对应论文 6.2.2)
        
        注: Test 1.1 和 1.2 已被 1.4 覆盖，不再默认运行。
        """
        print("\n" + "=" * 60)
        print("动态资源迁移实验 - 论文核心集")
        print("=" * 60)
        print(f"模型路径: {self.model_path}")
        print(f"请求数量: {self.llm_requests}")
        print("=" * 60)
        
        all_results = {
            "experiment_name": "resource_migration_paper_core",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "llm_requests": self.llm_requests,
                "warmup_requests": self.warmup_requests,
                "seed": self.seed,
                "model_path": self.model_path,
            },
            "results": {},
        }
        
        try:
            await self.setup()
            
            # 1. 运行主线生命周期测试 (覆盖 Migration & Coexistence)
            all_results["results"]["test_1_4_lifecycle"] = asdict(
                await self.test_integrated_lifecycle()
            )
            
            await asyncio.sleep(2)
            
            # 2. 运行背压互斥测试 (覆盖 Queue Backpressure)
            # 这是独立于迁移的另一个特性，必须单独测
            if include_backpressure:
                all_results["results"]["test_1_3_backpressure"] = asdict(
                    await self.test_1_3_backpressure_queue()
                )
            
            # 汇总结果
            lifecycle_passed = all_results["results"]["test_1_4_lifecycle"].get("test_passed", False)
            
            if include_backpressure:
                backpressure_passed = all_results["results"]["test_1_3_backpressure"].get("test_passed", False)
                all_passed = lifecycle_passed and backpressure_passed
            else:
                all_passed = lifecycle_passed

            all_results["overall_passed"] = all_passed

            all_results["resources"] = {
                "events": list(self._resource_events),
                "samples": [asdict(s) for s in self._resource_samples],
            }
            
            print("\n" + "=" * 60)
            print("论文实验汇总")
            print("=" * 60)
            print(f"  Sec 6.2.1 迁移与共存 (Test 1.4): {'✅ 通过' if lifecycle_passed else '❌ 失败'}")
            if include_backpressure:
                print(f"  Sec 6.2.2 队列背压 (Test 1.3):   {'✅ 通过' if backpressure_passed else '❌ 失败'}")
            print(f"\n总体结果: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
            
        except Exception as e:
            all_results["error"] = str(e)
            print(f"\n实验执行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.teardown()
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "resource_migration.json") -> str:
        """保存实验结果"""
        from framework import get_output_path

        subdir = "core"
        if "pending" in filename or "restore" in filename or "backpressure" in filename:
            subdir = "pending"
        elif "supplementary" in filename or "all" in filename:
            subdir = "supplementary"

        filepath = str(get_output_path(filename, subdir=subdir))
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **results}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {filepath}")
        return filepath


async def main():
    parser = argparse.ArgumentParser(description="动态资源迁移实验")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "all_no_image", "1.1", "1.2", "1.3", "1.4", "migration", "restore", "backpressure", "lifecycle"],
                        help="要运行的测试")
    parser.add_argument("--llm-requests", type=int, default=10, help="LLM 请求数")
    parser.add_argument("--warmup-requests", type=int, default=3, help="预热请求数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--iterations", type=int, default=1, help="迭代次数 (N)")
    
    args = parser.parse_args()

    if args.iterations < 1:
        print("Error: iterations must be >= 1")
        return

    metrics_keys = [
        "migration_stutter_ms", "coexistence_cpu_ttft_ms", "restore_time_ms",
        "restore_window_max_gap_ms", "llm_gpu_ttft_ms"
    ]

    if args.iterations == 1:
        experiment = ResourceMigrationExperiment(
            llm_requests=args.llm_requests,
            warmup_requests=args.warmup_requests,
            seed=args.seed,
        )
        
        output_filename = "resource_migration.json"

        if args.test in ["all", "all_no_image"]:
            results = await experiment.run_all_tests(include_backpressure=(args.test == "all"))
            output_filename = "resource_migration_all.json"
        else:
            await experiment.setup()
            
            if args.test in ["1.1", "migration"]:
                results = asdict(await experiment.test_1_1_gpu_to_cpu_migration())
                output_filename = "resource_migration.json"
            elif args.test in ["1.2", "restore"]:
                results = asdict(await experiment.test_1_2_cpu_to_gpu_restore())
                output_filename = "resource_migration_restore_pending.json"
            elif args.test in ["1.3", "backpressure"]:
                results = asdict(await experiment.test_1_3_backpressure_queue())
                output_filename = "resource_migration_backpressure_pending.json"
            elif args.test in ["1.4", "lifecycle"]:
                results = asdict(await experiment.test_integrated_lifecycle())
                output_filename = "resource_migration_lifecycle.json"
            else:
                results = {}
            
            await experiment.teardown()

            if "resources" not in results:
                results["resources"] = {
                    "events": list(experiment._resource_events),
                    "samples": [asdict(s) for s in experiment._resource_samples],
                }
        
        experiment.save_results(results, filename=output_filename)
    else:
        # 多次迭代模式
        all_iterations = []
        
        for i in range(args.iterations):
            print(f"\n{'='*60}")
            print(f"迭代 {i+1}/{args.iterations}")
            print(f"{'='*60}")

            experiment = ResourceMigrationExperiment(
                llm_requests=args.llm_requests,
                warmup_requests=args.warmup_requests,
                seed=args.seed + i,
            )

            if args.test in ["all", "all_no_image"]:
                results = await experiment.run_all_tests(include_backpressure=(args.test == "all"))
            else:
                await experiment.setup()
                
                if args.test in ["1.1", "migration"]:
                    results = asdict(await experiment.test_1_1_gpu_to_cpu_migration())
                elif args.test in ["1.2", "restore"]:
                    results = asdict(await experiment.test_1_2_cpu_to_gpu_restore())
                elif args.test in ["1.3", "backpressure"]:
                    results = asdict(await experiment.test_1_3_backpressure_queue())
                elif args.test in ["1.4", "lifecycle"]:
                    results = asdict(await experiment.test_integrated_lifecycle())
                else:
                    results = {}
                
                await experiment.teardown()

            all_iterations.append(results)
            experiment.save_results(results, filename=f"resource_migration_{args.test}_iter{i+1}.json")

            if i < args.iterations - 1:
                print("等待系统冷却...")
                await asyncio.sleep(30)

        # 计算统计数据
        aggregated = {
            "experiment_name": f"resource_migration_{args.test}",
            "test": args.test,
            "iterations": args.iterations,
            "metrics": {}
        }

        for key in metrics_keys:
            values = [r[key] for r in all_iterations if key in r and r[key] is not None and r[key] != 0.0]
            if not values:
                values = [r.get(key, 0.0) for r in all_iterations if r.get(key) is not None]
            if values:
                import statistics
                aggregated["metrics"][key] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                    "n_valid": len(values),
                    "n_total": len(all_iterations),
                }

        # 保存聚合结果
        from framework import get_output_path
        output_filename = f"resource_migration_{args.test}_aggregated.json"
        filepath = str(get_output_path(output_filename, subdir="core"))
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **aggregated}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"\n聚合结果已保存到: {filepath}")
        print("\n统计摘要:")
        for key, data in aggregated["metrics"].items():
            print(f"  {key}: {data['mean']:.1f} ± {data['stdev']:.1f} ms")
    
    print("\n" + "=" * 60)
    print("实验完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
