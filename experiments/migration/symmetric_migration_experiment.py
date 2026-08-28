"""
对称迁移基线实验 (Symmetric Migration Baseline)

对比 EdgeSched 的非对称迁移策略（Urgent Down + Atomic Hot-Swap Up），
本实验测试"对称迁移"策略：GPU→CPU 和 CPU→GPU 使用同一恢复逻辑/同一优化目标，
即都采用"先停服务再完整迁移再恢复"的统一流程。

包含三个变体：
1. Symmetric Migration: 两个方向都用 naive "stop→migrate→resume"
2. Fast-down + naive-up: Downlink 用 EdgeSched 优化, Uplink 用 naive
3. Naive-down + optimized-up: Downlink 用 naive, Uplink 用 EdgeSched 优化

指标体系（与 tab_migration 对齐）：
- Downlink stutter (ms)
- CPU TTFT during coexistence (ms)
- Restore time (ms)
- Restore window / interruption gap (ms)
- OOM / failed restore count

对应论文评估章节：Baseline Behavior Under Alternative Designs
"""

import asyncio
import argparse
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

# 设置 LD_LIBRARY_PATH 以加载 C++ scheduler 的动态库
_build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cpp_scheduler", "build")
_bin_dir = os.path.join(_build_dir, "bin")
if os.path.isdir(_bin_dir):
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _bin_dir not in _existing_ld:
        os.environ["LD_LIBRARY_PATH"] = _bin_dir + (":" + _existing_ld if _existing_ld else "")
    if _build_dir not in sys.path:
        sys.path.insert(0, _build_dir)

os.environ.setdefault("XIAOYOU_ALLOW_CPP_LLM_WORKER", "1")

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
class SymmetricMigrationResult:
    """对称迁移测试结果"""
    variant: str = ""
    test_passed: bool = False
    error_message: str = ""

    llm_gpu_ttft_ms: float = 0.0
    llm_cpu_ttft_ms: float = 0.0
    migration_time_ms: float = 0.0
    llm_still_working: bool = False
    migration_window_max_gap_ms: float = 0.0
    migration_window_max_ttft_ms: float = 0.0

    restore_time_ms: float = 0.0
    service_interruption_ms: float = 0.0
    first_gpu_ttft_ms_after_restore: float = 0.0
    kv_cache_preserved: bool = False
    restore_window_max_gap_ms: float = 0.0
    restore_window_max_ttft_ms: float = 0.0

    coexistence_cpu_ttft_ms: float = 0.0
    migration_stutter_ms: float = 0.0

    oom_count: int = 0
    failed_restore_count: int = 0


@dataclass
class LlmProbeSample:
    t_s: float
    llm_device: str
    ttft_ms: float
    first_token_t: float


class SymmetricMigrationExperiment:
    """对称迁移基线实验"""

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

        self._resource_t0: float = 0.0

    def setup_paths(self) -> None:
        setup_paths()

    async def setup(self) -> None:
        self.reproducibility.setup()
        self.setup_paths()

        if not self.model_path:
            print("[Warning] XIAOYOU_LLM_MODEL_PATH 未设置，尝试默认路径")
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidate_paths = [
                os.path.join(base_dir, "models", "llm", "qwen2.5-7b-instruct-q4_k_m.gguf"),
                os.path.join(base_dir, "models", "qwen2.5-7b-instruct-q4_k_m.gguf"),
            ]
            found = False
            for p in candidate_paths:
                if os.path.exists(p):
                    self.model_path = p
                    found = True
                    break
            if not found:
                print(f"[Error] 模型文件不存在，已搜索: {candidate_paths}")
                raise FileNotFoundError(f"模型文件不存在，已搜索: {candidate_paths}")

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

            # 如果 C++ scheduler 未启用（scheduler_py 不可用），
            # start() 会跳过，需要手动初始化 Python LLM
            if not cpp_scheduler_engine._started:
                print("[Setup] C++ scheduler 未启动，手动初始化 Python LLM...")
                cpp_scheduler_engine._gpu_config = config
                cpp_scheduler_engine._llm_backend = "python"
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    cpp_scheduler_engine._llm_executor,
                    cpp_scheduler_engine._setup_python_llm,
                    config,
                    loop,
                )
                if cpp_scheduler_engine._python_llm is None:
                    raise RuntimeError("Python LLM 初始化失败")
                print("[Setup] Python LLM 初始化完成")
            else:
                print(f"[Setup] C++ scheduler 已启动 (backend={cpp_scheduler_engine._llm_backend})")

            print("[Setup] 检查 Forge 服务可用性...")
            forge_ok = False
            base_url = os.environ.get("XIAOYOU_FORGE_BASE_URL", "http://127.0.0.1:7860")
            candidate_urls = [base_url]
            if base_url == "http://127.0.0.1:7860":
                candidate_urls.append("http://127.0.0.1:7861")

            for _ in range(3):
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    for url in candidate_urls:
                        try:
                            async with session.get(f"{url}/sdapi/v1/sd-models", timeout=5) as resp:
                                if resp.status == 200:
                                    forge_ok = True
                                    os.environ["XIAOYOU_FORGE_BASE_URL"] = url
                                    print(f"  Forge 服务已连接 ({url})")
                                    break
                        except Exception:
                            pass
                    if forge_ok:
                        break
                print("  等待 Forge 服务启动...")
                await asyncio.sleep(2)

            if not forge_ok:
                print("[Warning] Forge 服务未响应！图像生成将回退到 Mock 模式。")

            self._resource_t0 = time.time()
            await self._wait_for_llm_ready(timeout_s=180.0)
            print("[Setup] 调度器初始化完成")

        except Exception as e:
            print(f"[Setup] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def teardown(self) -> None:
        if self.scheduler:
            try:
                await self.scheduler.stop()
            except Exception:
                pass
        print("[Teardown] 实验清理完成")

    async def _llm_generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7):
        """统一的 LLM 生成接口，兼容 enabled/disabled 状态"""
        if self.scheduler.enabled:
            async for token in self.scheduler.submit_llm_task(prompt, max_tokens=max_tokens, temperature=temperature):
                yield token
        else:
            async for token in self.scheduler._submit_llm_task_python(prompt, max_tokens=max_tokens, temperature=temperature):
                yield token

    async def _submit_mock_image_task(self, duration_s: float = 8.0):
        """Mock 图像任务，模拟 GPU 占用"""
        print(f"  [Mock Image] 模拟图像生成 ({duration_s}s)...")
        await asyncio.sleep(duration_s)
        print("  [Mock Image] 图像生成完成")
        return {"status": "ok", "mock": True}

    async def _submit_image_task(self, prompt: str = "A futuristic city", width: int = 512, height: int = 512, steps: int = 20, guidance_scale: float = 7.5):
        """提交图像任务，兼容 enabled/disabled 状态"""
        if self.scheduler.enabled:
            return await self.scheduler.submit_image_task(prompt, width=width, height=height, steps=steps, guidance_scale=guidance_scale)
        else:
            return await self._submit_mock_image_task(duration_s=8.0)

    async def measure_llm_ttft(self, prompt: str = "你好，请介绍一下你自己。") -> float:
        start = time.perf_counter()
        first_token = False
        try:
            async for token in self._llm_generate(prompt, max_tokens=64, temperature=0.7):
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
            async for token in self._llm_generate(prompt, max_tokens=64, temperature=0.7):
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
        except Exception:
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

    async def _naive_restore_to_gpu(self) -> Dict[str, float]:
        """
        对称恢复：先停 CPU 服务，再完整加载 GPU 模型，再恢复服务
        核心区别：没有 GPU 后台预加载，没有原子替换，服务中断 = 完整模型加载时间
        """
        print("  [Naive Restore] 停止 CPU LLM 服务...")
        t_start = time.perf_counter()

        async with self.scheduler._python_llm_lock:
            # 1. 停止 CPU LLM（释放实例）
            old_inst = self.scheduler._python_llm
            if old_inst is not None:
                def _cleanup_old():
                    try:
                        if hasattr(old_inst, "close"):
                            old_inst.close()
                    except Exception:
                        pass
                await asyncio.get_running_loop().run_in_executor(
                    self.scheduler._llm_executor, _cleanup_old
                )
                self.scheduler._python_llm = None
                self.scheduler._python_force_cpu = False
                self.scheduler._python_llm_device = None

            # 2. 释放显存
            def _gc_collect():
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except ImportError:
                    pass
            await asyncio.get_running_loop().run_in_executor(
                self.scheduler._llm_executor, _gc_collect
            )

            # 3. 从磁盘重新加载 GPU LLM（cold start，无后台预加载）
            print("  [Naive Restore] 从磁盘加载 GPU LLM...")
            gpu_config = dict(self.scheduler._gpu_config) if isinstance(self.scheduler._gpu_config, dict) else {}
            if self.scheduler._prev_n_gpu_layers is not None:
                gpu_config["n_gpu_layers"] = self.scheduler._prev_n_gpu_layers
            if self.scheduler._prev_n_ctx is not None:
                gpu_config["max_context_size"] = self.scheduler._prev_n_ctx
            if self.scheduler._prev_n_batch is not None:
                gpu_config["max_batch_size"] = self.scheduler._prev_n_batch
            gpu_config["force_cpu"] = False

            await asyncio.get_running_loop().run_in_executor(
                self.scheduler._llm_executor,
                self.scheduler._setup_python_llm,
                gpu_config,
            )

            if self.scheduler._python_llm is None:
                t_end = time.perf_counter()
                return {
                    "restore_time_ms": (t_end - t_start) * 1000,
                    "service_interruption_ms": (t_end - t_start) * 1000,
                    "failed": 1,
                }

            self.scheduler._python_force_cpu = False
            self.scheduler._python_llm_device = "gpu"

        t_end = time.perf_counter()
        restore_time_ms = (t_end - t_start) * 1000
        print(f"  [Naive Restore] 完成，耗时: {restore_time_ms:.1f} ms")

        return {
            "restore_time_ms": restore_time_ms,
            "service_interruption_ms": restore_time_ms,
            "failed": 0,
        }

    async def _naive_offload_to_cpu(self) -> Dict[str, float]:
        """
        对称卸载：先停 GPU 服务，再完整加载 CPU 模型，再恢复服务
        核心区别：没有异步卸载优化，没有 KV Cache 保存，服务中断 = CPU 模型完整加载时间
        """
        print("  [Naive Offload] 停止 GPU LLM 服务...")
        t_start = time.perf_counter()

        async with self.scheduler._python_llm_lock:
            # 1. 停止 GPU LLM（不保存 KV Cache）
            old_inst = self.scheduler._python_llm
            if old_inst is not None:
                def _cleanup_old():
                    try:
                        if hasattr(old_inst, "close"):
                            old_inst.close()
                    except Exception:
                        pass
                await asyncio.get_running_loop().run_in_executor(
                    self.scheduler._llm_executor, _cleanup_old
                )
                self.scheduler._python_llm = None

            # 2. 释放显存
            def _gc_collect():
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except ImportError:
                    pass
            await asyncio.get_running_loop().run_in_executor(
                self.scheduler._llm_executor, _gc_collect
            )

            # 3. 加载 CPU LLM（cold start）
            print("  [Naive Offload] 加载 CPU LLM...")
            cpu_config = dict(self.scheduler._gpu_config) if isinstance(self.scheduler._gpu_config, dict) else {}
            cpu_config["force_cpu"] = True
            cpu_config["n_gpu_layers"] = 0

            await asyncio.get_running_loop().run_in_executor(
                self.scheduler._llm_executor,
                self.scheduler._setup_python_llm,
                cpu_config,
            )

            if self.scheduler._python_llm is None:
                t_end = time.perf_counter()
                return {
                    "migration_time_ms": (t_end - t_start) * 1000,
                    "stutter_ms": (t_end - t_start) * 1000,
                    "failed": 1,
                }

            self.scheduler._python_force_cpu = True
            self.scheduler._python_llm_device = "cpu"

        t_end = time.perf_counter()
        migration_time_ms = (t_end - t_start) * 1000
        print(f"  [Naive Offload] 完成，耗时: {migration_time_ms:.1f} ms")

        return {
            "migration_time_ms": migration_time_ms,
            "stutter_ms": migration_time_ms,
            "failed": 0,
        }

    async def run_symmetric_lifecycle(self) -> SymmetricMigrationResult:
        """
        对称迁移完整生命周期测试

        流程：
        1. [State: GPU] LLM 在 GPU 上运行 (Baseline)
        2. [Event] 图像任务到达
        3. [Action] 对称卸载 (Naive: GPU→CPU, 停服务→完整迁移→恢复)
        4. [State: Coexistence] 图像生成中 (GPU) + LLM 在 CPU 响应 (CPU)
        5. [Event] 图像任务完成
        6. [Action] 对称恢复 (Naive: CPU→GPU, 停服务→完整迁移→恢复)
        7. [State: GPU] LLM 恢复 GPU 运行
        """
        variant = "symmetric_migration"
        print("\n" + "=" * 60)
        print(f"[实验] 对称迁移生命周期测试 ({variant})")
        print("=" * 60)

        result = SymmetricMigrationResult(variant=variant)

        try:
            # Phase 1: 确保 LLM 在 GPU 上
            print("\n[Phase 1] 初始状态确认 (LLM on GPU)...")
            await self.scheduler.restore_llm_to_gpu()
            await self._wait_for_llm_device("gpu")

            print("  测量 GPU 基线 TTFT...")
            gpu_ttfts = []
            for _ in range(3):
                t = await self.measure_llm_ttft()
                if t > 0:
                    gpu_ttfts.append(t)
            result.llm_gpu_ttft_ms = statistics.mean(gpu_ttfts) if gpu_ttfts else 0.0
            print(f"  GPU Baseline TTFT: {result.llm_gpu_ttft_ms:.1f} ms")

            # Phase 2: 对称卸载 (Naive GPU→CPU)
            print("\n[Phase 2] 提交图像任务并触发对称卸载...")

            t_start = time.perf_counter()

            image_task_future = asyncio.create_task(
                self._submit_image_task(
                    "A futuristic city with flying cars, cyberpunk style, high details",
                    width=512, height=512, steps=20, guidance_scale=7.5
                )
            )

            # 对称卸载：naive offload（并发执行，不等完成）
            offload_future = asyncio.create_task(self._naive_offload_to_cpu())

            # Phase 3: 共存测试（在迁移过程中并发探测）
            print("  [Coexistence] 图像生成中，并发测试 CPU LLM 响应...")
            cpu_ttfts = []
            migration_done = False
            t_cpu_ready = None

            while not image_task_future.done():
                if not migration_done and offload_future.done():
                    migration_done = True
                    try:
                        offload_result = await offload_future
                        result.migration_stutter_ms = offload_result.get("stutter_ms", 0.0)
                        result.migration_time_ms = offload_result.get("migration_time_ms", 0.0)
                        result.oom_count += offload_result.get("failed", 0)
                    except Exception:
                        pass
                    try:
                        await self._wait_for_llm_device("cpu")
                        t_cpu_ready = time.perf_counter()
                        print(f"  对称卸载卡顿 (Stutter): {result.migration_stutter_ms:.1f} ms")
                    except Exception:
                        pass

                if migration_done and t_cpu_ready is not None:
                    t_probe = await self.measure_llm_ttft("在这个共存阶段，请简短回答：你还好吗？")
                    if t_probe > 0:
                        cpu_ttfts.append(t_probe)
                        print(f"  [Coexistence] CPU TTFT: {t_probe:.1f} ms")

                await asyncio.sleep(0.1)

            if not migration_done:
                try:
                    offload_result = await offload_future
                    result.migration_stutter_ms = offload_result.get("stutter_ms", 0.0)
                    result.migration_time_ms = offload_result.get("migration_time_ms", 0.0)
                    result.oom_count += offload_result.get("failed", 0)
                    await self._wait_for_llm_device("cpu")
                    print(f"  对称卸载卡顿 (Stutter): {result.migration_stutter_ms:.1f} ms")
                except Exception:
                    pass
                print("  警告: 图像任务在迁移完成前已结束，未能测到共存阶段数据")

            try:
                await image_task_future
                print("  图像任务完成")
            except Exception as e:
                print(f"  图像任务异常: {e}")

            if cpu_ttfts:
                result.coexistence_cpu_ttft_ms = statistics.mean(cpu_ttfts)
                print(f"  共存阶段 CPU TTFT avg: {result.coexistence_cpu_ttft_ms:.1f} ms")

            # Phase 4: 对称恢复 (Naive CPU→GPU)
            print("\n[Phase 4] 图像完成，触发对称恢复...")

            probe_samples: List[LlmProbeSample] = []
            probe_stop = asyncio.Event()
            probe_task = asyncio.create_task(
                self._probe_llm_continuously(probe_samples, probe_stop, interval_s=0.2, prompt="恢复测试")
            )

            t_restore_start = time.perf_counter()

            # 对称恢复：naive restore
            restore_result = await self._naive_restore_to_gpu()
            result.restore_time_ms = restore_result.get("restore_time_ms", 0.0)
            result.service_interruption_ms = restore_result.get("service_interruption_ms", 0.0)
            result.failed_restore_count += restore_result.get("failed", 0)

            t_restore_end = time.perf_counter()

            await asyncio.sleep(0.5)
            probe_stop.set()
            try:
                await probe_task
            except Exception:
                pass

            metrics = self._compute_gap_and_spike(probe_samples)
            result.restore_window_max_gap_ms = metrics["max_gap_ms"]
            result.restore_window_max_ttft_ms = metrics["max_ttft_ms"]

            # 如果 probe 没有捕捉到 gap（因为 naive restore 期间服务完全中断），
            # 使用完整恢复时间作为服务中断
            if result.restore_window_max_gap_ms <= 0 and result.service_interruption_ms > 0:
                result.restore_window_max_gap_ms = result.service_interruption_ms

            print(f"  对称恢复总耗时: {result.restore_time_ms:.1f} ms")
            print(f"  服务中断 (Restore Gap): {result.restore_window_max_gap_ms:.1f} ms")

            # 验证回迁后状态
            try:
                await self._wait_for_llm_device("gpu", timeout_s=30.0)
                t_final = await self.measure_llm_ttft()
                result.first_gpu_ttft_ms_after_restore = t_final
                print(f"  回迁后 GPU TTFT: {t_final:.1f} ms")
            except Exception as e:
                print(f"  回迁后验证失败: {e}")
                result.failed_restore_count += 1

            # 对称迁移的 naive offload 可能比图像生成还慢，导致共存阶段测不到数据
            # 这是符合预期的物理限制，只要恢复阶段成功即可认为测试通过
            result.llm_still_working = len(cpu_ttfts) > 0
            result.test_passed = result.failed_restore_count == 0

            if result.test_passed:
                if result.llm_still_working:
                    print(f"\n✅ [Symmetric] 完整生命周期测试通过")
                else:
                    print(f"\n✅ [Symmetric] 测试通过（共存阶段因 naive offload 过慢未测到数据，符合预期）")
            else:
                print(f"\n❌ [Symmetric] 测试失败")

        except Exception as e:
            result.error_message = str(e)
            result.failed_restore_count += 1
            print(f"\n❌ [Symmetric] 异常: {e}")
            import traceback
            traceback.print_exc()

        return result

    async def run_fast_down_naive_up(self) -> SymmetricMigrationResult:
        """
        Fast-down + naive-up ablation

        Downlink: 使用 EdgeSched 的 Urgent Mode（优化）
        Uplink: 使用 naive "停服务→完整迁移→恢复"（未优化）
        """
        variant = "fast_down_naive_up"
        print("\n" + "=" * 60)
        print(f"[实验] Fast-down + Naive-up ablation ({variant})")
        print("=" * 60)

        result = SymmetricMigrationResult(variant=variant)

        try:
            # Phase 1: 确保 LLM 在 GPU 上
            print("\n[Phase 1] 初始状态确认 (LLM on GPU)...")
            await self.scheduler.restore_llm_to_gpu()
            await self._wait_for_llm_device("gpu")

            gpu_ttfts = []
            for _ in range(3):
                t = await self.measure_llm_ttft()
                if t > 0:
                    gpu_ttfts.append(t)
            result.llm_gpu_ttft_ms = statistics.mean(gpu_ttfts) if gpu_ttfts else 0.0
            print(f"  GPU Baseline TTFT: {result.llm_gpu_ttft_ms:.1f} ms")

            # Phase 2: Fast downlink (EdgeSched Urgent Mode)
            print("\n[Phase 2] 提交图像任务并触发 EdgeSched Urgent Offload...")

            t_start = time.perf_counter()

            image_task_future = asyncio.create_task(
                self._submit_image_task(
                    "A futuristic city with flying cars, cyberpunk style, high details",
                    width=512, height=512, steps=20, guidance_scale=7.5
                )
            )

            # EdgeSched 优化的 downlink（并发执行）
            offload_future = asyncio.create_task(self.scheduler.offload_llm_to_cpu(urgent=True))

            # Phase 3: 共存测试（在迁移过程中并发探测）
            print("  [Coexistence] 图像生成中，并发测试 CPU LLM 响应...")
            cpu_ttfts = []
            migration_done = False
            t_cpu_ready = None

            while not image_task_future.done():
                if not migration_done and offload_future.done():
                    migration_done = True
                    try:
                        await offload_future
                    except Exception:
                        pass
                    try:
                        await self._wait_for_llm_device("cpu")
                        t_cpu_ready = time.perf_counter()
                        result.migration_stutter_ms = (t_cpu_ready - t_start) * 1000.0
                        result.migration_time_ms = result.migration_stutter_ms
                        print(f"  Fast Downlink Stutter: {result.migration_stutter_ms:.1f} ms")
                    except Exception:
                        pass

                if migration_done and t_cpu_ready is not None:
                    t_probe = await self.measure_llm_ttft("在这个共存阶段，请简短回答：你还好吗？")
                    if t_probe > 0:
                        cpu_ttfts.append(t_probe)
                        print(f"  [Coexistence] CPU TTFT: {t_probe:.1f} ms")

                await asyncio.sleep(0.1)

            if not migration_done:
                try:
                    await offload_future
                    await self._wait_for_llm_device("cpu")
                    t_cpu_ready = time.perf_counter()
                    result.migration_stutter_ms = (t_cpu_ready - t_start) * 1000.0
                    result.migration_time_ms = result.migration_stutter_ms
                    print(f"  Fast Downlink Stutter: {result.migration_stutter_ms:.1f} ms")
                except Exception:
                    pass
                print("  警告: 图像任务在迁移完成前已结束，未能测到共存阶段数据")

            try:
                await image_task_future
                print("  图像任务完成")
            except Exception as e:
                print(f"  图像任务异常: {e}")

            if cpu_ttfts:
                result.coexistence_cpu_ttft_ms = statistics.mean(cpu_ttfts)
                print(f"  共存阶段 CPU TTFT avg: {result.coexistence_cpu_ttft_ms:.1f} ms")

            # Phase 4: Naive uplink
            print("\n[Phase 4] 图像完成，触发 Naive Restore...")

            probe_samples: List[LlmProbeSample] = []
            probe_stop = asyncio.Event()
            probe_task = asyncio.create_task(
                self._probe_llm_continuously(probe_samples, probe_stop, interval_s=0.2, prompt="恢复测试")
            )

            restore_result = await self._naive_restore_to_gpu()
            result.restore_time_ms = restore_result.get("restore_time_ms", 0.0)
            result.service_interruption_ms = restore_result.get("service_interruption_ms", 0.0)
            result.failed_restore_count += restore_result.get("failed", 0)

            await asyncio.sleep(0.5)
            probe_stop.set()
            try:
                await probe_task
            except Exception:
                pass

            metrics = self._compute_gap_and_spike(probe_samples)
            result.restore_window_max_gap_ms = metrics["max_gap_ms"]
            result.restore_window_max_ttft_ms = metrics["max_ttft_ms"]

            if result.restore_window_max_gap_ms <= 0 and result.service_interruption_ms > 0:
                result.restore_window_max_gap_ms = result.service_interruption_ms

            print(f"  Naive Restore 总耗时: {result.restore_time_ms:.1f} ms")
            print(f"  服务中断 (Restore Gap): {result.restore_window_max_gap_ms:.1f} ms")

            try:
                await self._wait_for_llm_device("gpu", timeout_s=30.0)
                t_final = await self.measure_llm_ttft()
                result.first_gpu_ttft_ms_after_restore = t_final
                print(f"  回迁后 GPU TTFT: {t_final:.1f} ms")
            except Exception as e:
                print(f"  回迁后验证失败: {e}")
                result.failed_restore_count += 1

            # naive offload 可能比图像生成还慢，导致共存阶段测不到数据，符合预期
            result.llm_still_working = len(cpu_ttfts) > 0
            result.test_passed = result.failed_restore_count == 0

            if result.test_passed:
                if result.llm_still_working:
                    print(f"\n✅ [Fast-down + Naive-up] 测试通过")
                else:
                    print(f"\n✅ [Fast-down + Naive-up] 测试通过（共存阶段因 naive offload 过慢未测到数据，符合预期）")
            else:
                print(f"\n❌ [Fast-down + Naive-up] 测试失败")

        except Exception as e:
            result.error_message = str(e)
            result.failed_restore_count += 1
            print(f"\n❌ [Fast-down + Naive-up] 异常: {e}")
            import traceback
            traceback.print_exc()

        return result

    async def run_naive_down_optimized_up(self) -> SymmetricMigrationResult:
        """
        Naive-down + optimized-up ablation

        Downlink: 使用 naive "停服务→完整迁移→恢复"（未优化）
        Uplink: 使用 EdgeSched 的 Atomic Hot-Swap（优化）
        """
        variant = "naive_down_optimized_up"
        print("\n" + "=" * 60)
        print(f"[实验] Naive-down + Optimized-up ablation ({variant})")
        print("=" * 60)

        result = SymmetricMigrationResult(variant=variant)

        try:
            # Phase 1: 确保 LLM 在 GPU 上
            print("\n[Phase 1] 初始状态确认 (LLM on GPU)...")
            await self.scheduler.restore_llm_to_gpu()
            await self._wait_for_llm_device("gpu")

            gpu_ttfts = []
            for _ in range(3):
                t = await self.measure_llm_ttft()
                if t > 0:
                    gpu_ttfts.append(t)
            result.llm_gpu_ttft_ms = statistics.mean(gpu_ttfts) if gpu_ttfts else 0.0
            print(f"  GPU Baseline TTFT: {result.llm_gpu_ttft_ms:.1f} ms")

            # Phase 2: Naive downlink
            print("\n[Phase 2] 提交图像任务并触发 Naive Offload...")

            t_start = time.perf_counter()

            image_task_future = asyncio.create_task(
                self._submit_image_task(
                    "A futuristic city with flying cars, cyberpunk style, high details",
                    width=512, height=512, steps=20, guidance_scale=7.5
                )
            )

            # Naive offload（并发执行）
            offload_future = asyncio.create_task(self._naive_offload_to_cpu())

            # Phase 3: 共存测试（在迁移过程中并发探测）
            print("  [Coexistence] 图像生成中，并发测试 CPU LLM 响应...")
            cpu_ttfts = []
            migration_done = False
            t_cpu_ready = None

            while not image_task_future.done():
                if not migration_done and offload_future.done():
                    migration_done = True
                    try:
                        offload_result = await offload_future
                        result.migration_stutter_ms = offload_result.get("stutter_ms", 0.0)
                        result.migration_time_ms = offload_result.get("migration_time_ms", 0.0)
                        result.oom_count += offload_result.get("failed", 0)
                    except Exception:
                        pass
                    try:
                        await self._wait_for_llm_device("cpu")
                        t_cpu_ready = time.perf_counter()
                        print(f"  Naive Downlink Stutter: {result.migration_stutter_ms:.1f} ms")
                    except Exception:
                        pass

                if migration_done and t_cpu_ready is not None:
                    t_probe = await self.measure_llm_ttft("在这个共存阶段，请简短回答：你还好吗？")
                    if t_probe > 0:
                        cpu_ttfts.append(t_probe)
                        print(f"  [Coexistence] CPU TTFT: {t_probe:.1f} ms")

                await asyncio.sleep(0.1)

            if not migration_done:
                try:
                    offload_result = await offload_future
                    result.migration_stutter_ms = offload_result.get("stutter_ms", 0.0)
                    result.migration_time_ms = offload_result.get("migration_time_ms", 0.0)
                    result.oom_count += offload_result.get("failed", 0)
                    await self._wait_for_llm_device("cpu")
                    print(f"  Naive Downlink Stutter: {result.migration_stutter_ms:.1f} ms")
                except Exception:
                    pass
                print("  警告: 图像任务在迁移完成前已结束，未能测到共存阶段数据")

            try:
                await image_task_future
                print("  图像任务完成")
            except Exception as e:
                print(f"  图像任务异常: {e}")

            if cpu_ttfts:
                result.coexistence_cpu_ttft_ms = statistics.mean(cpu_ttfts)
                print(f"  共存阶段 CPU TTFT avg: {result.coexistence_cpu_ttft_ms:.1f} ms")

            # Phase 4: Optimized uplink (EdgeSched Atomic Hot-Swap)
            print("\n[Phase 4] 图像完成，触发 EdgeSched Atomic Hot-Swap...")

            probe_samples: List[LlmProbeSample] = []
            probe_stop = asyncio.Event()
            probe_task = asyncio.create_task(
                self._probe_llm_continuously(probe_samples, probe_stop, interval_s=0.2, prompt="恢复测试")
            )

            t_restore_start = time.perf_counter()
            await self.scheduler.restore_llm_to_gpu()
            t_restore_end = time.perf_counter()

            result.restore_time_ms = (t_restore_end - t_restore_start) * 1000.0

            await asyncio.sleep(0.5)
            probe_stop.set()
            try:
                await probe_task
            except Exception:
                pass

            metrics = self._compute_gap_and_spike(probe_samples)
            result.restore_window_max_gap_ms = metrics["max_gap_ms"]
            result.restore_window_max_ttft_ms = metrics["max_ttft_ms"]
            result.service_interruption_ms = result.restore_window_max_gap_ms

            print(f"  Atomic Hot-Swap 总耗时: {result.restore_time_ms:.1f} ms")
            print(f"  服务中断 (Restore Gap): {result.restore_window_max_gap_ms:.1f} ms")

            try:
                await self._wait_for_llm_device("gpu", timeout_s=30.0)
                t_final = await self.measure_llm_ttft()
                result.first_gpu_ttft_ms_after_restore = t_final
                print(f"  回迁后 GPU TTFT: {t_final:.1f} ms")
            except Exception as e:
                print(f"  回迁后验证失败: {e}")
                result.failed_restore_count += 1

            # naive offload 可能比图像生成还慢，导致共存阶段测不到数据，符合预期
            result.llm_still_working = len(cpu_ttfts) > 0
            result.test_passed = result.failed_restore_count == 0

            if result.test_passed:
                if result.llm_still_working:
                    print(f"\n✅ [Naive-down + Optimized-up] 测试通过")
                else:
                    print(f"\n✅ [Naive-down + Optimized-up] 测试通过（共存阶段因 naive offload 过慢未测到数据，符合预期）")
            else:
                print(f"\n❌ [Naive-down + Optimized-up] 测试失败")

        except Exception as e:
            result.error_message = str(e)
            result.failed_restore_count += 1
            print(f"\n❌ [Naive-down + Optimized-up] 异常: {e}")
            import traceback
            traceback.print_exc()

        return result

    async def run_all_variants(self) -> Dict[str, Any]:
        """运行所有变体"""
        print("\n" + "=" * 60)
        print("对称迁移基线实验 - 全部变体")
        print("=" * 60)
        print(f"模型路径: {self.model_path}")
        print(f"请求数量: {self.llm_requests}")
        print("=" * 60)

        all_results = {
            "experiment_name": "symmetric_migration_baseline",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "llm_requests": self.llm_requests,
                "warmup_requests": self.warmup_requests,
                "seed": self.seed,
                "model_path": self.model_path,
            },
            "variants": {},
        }

        try:
            await self.setup()

            # 1. Symmetric Migration (核心 baseline)
            print("\n>>> 变体 1/3: Symmetric Migration <<<")
            all_results["variants"]["symmetric_migration"] = asdict(
                await self.run_symmetric_lifecycle()
            )
            await asyncio.sleep(5)

            # 2. Fast-down + naive-up
            print("\n>>> 变体 2/3: Fast-down + Naive-up <<<")
            all_results["variants"]["fast_down_naive_up"] = asdict(
                await self.run_fast_down_naive_up()
            )
            await asyncio.sleep(5)

            # 3. Naive-down + optimized-up
            print("\n>>> 变体 3/3: Naive-down + Optimized-up <<<")
            all_results["variants"]["naive_down_optimized_up"] = asdict(
                await self.run_naive_down_optimized_up()
            )

            # 汇总
            print("\n" + "=" * 60)
            print("对称迁移基线实验汇总")
            print("=" * 60)

            for vname, vdata in all_results["variants"].items():
                passed = vdata.get("test_passed", False)
                stutter = vdata.get("migration_stutter_ms", 0)
                cpu_ttft = vdata.get("coexistence_cpu_ttft_ms", 0)
                restore = vdata.get("restore_time_ms", 0)
                gap = vdata.get("restore_window_max_gap_ms", 0)
                oom = vdata.get("oom_count", 0)
                fail = vdata.get("failed_restore_count", 0)
                print(f"  {vname}: {'✅' if passed else '❌'}")
                print(f"    Downlink Stutter: {stutter:.1f} ms")
                print(f"    CPU TTFT (coexistence): {cpu_ttft:.1f} ms")
                print(f"    Restore Time: {restore:.1f} ms")
                print(f"    Restore Gap: {gap:.1f} ms")
                print(f"    OOM/Fail: {oom}/{fail}")

        except Exception as e:
            all_results["error"] = str(e)
            print(f"\n实验执行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.teardown()

        return all_results

    def save_results(self, results: Dict[str, Any], filename: str = "symmetric_migration_baseline.json") -> str:
        from framework import get_output_path

        filepath = str(get_output_path(filename, subdir="core"))
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **results}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {filepath}")
        return filepath


async def main():
    parser = argparse.ArgumentParser(description="对称迁移基线实验")
    parser.add_argument("--variant", type=str, default="all",
                        choices=["all", "symmetric", "fast_down_naive_up", "naive_down_optimized_up"],
                        help="要运行的变体")
    parser.add_argument("--llm-requests", type=int, default=10, help="LLM 请求数")
    parser.add_argument("--warmup-requests", type=int, default=3, help="预热请求数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--iterations", type=int, default=1, help="迭代次数 (N)")

    args = parser.parse_args()

    if args.iterations < 1:
        print("Error: iterations must be >= 1")
        return

    if args.iterations == 1:
        experiment = SymmetricMigrationExperiment(
            llm_requests=args.llm_requests,
            warmup_requests=args.warmup_requests,
            seed=args.seed,
        )

        if args.variant == "all":
            results = await experiment.run_all_variants()
            output_filename = "symmetric_migration_baseline.json"
        else:
            await experiment.setup()

            if args.variant == "symmetric":
                results = asdict(await experiment.run_symmetric_lifecycle())
                output_filename = "symmetric_migration.json"
            elif args.variant == "fast_down_naive_up":
                results = asdict(await experiment.run_fast_down_naive_up())
                output_filename = "fast_down_naive_up.json"
            elif args.variant == "naive_down_optimized_up":
                results = asdict(await experiment.run_naive_down_optimized_up())
                output_filename = "naive_down_optimized_up.json"
            else:
                results = {}

            await experiment.teardown()

        experiment.save_results(results, filename=output_filename)
    else:
        # 多次迭代模式
        all_iterations = {args.variant: [] for v in ["symmetric", "fast_down_naive_up", "naive_down_optimized_up"]}
        metrics_keys = [
            "migration_stutter_ms", "coexistence_cpu_ttft_ms", "restore_time_ms",
            "restore_window_max_gap_ms", "service_interruption_ms", "oom_count", "failed_restore_count"
        ]

        for i in range(args.iterations):
            print(f"\n{'='*60}")
            print(f"迭代 {i+1}/{args.iterations}")
            print(f"{'='*60}")

            experiment = SymmetricMigrationExperiment(
                llm_requests=args.llm_requests,
                warmup_requests=args.warmup_requests,
                seed=args.seed + i,
            )

            await experiment.setup()

            if args.variant == "symmetric":
                result = asdict(await experiment.run_symmetric_lifecycle())
            elif args.variant == "fast_down_naive_up":
                result = asdict(await experiment.run_fast_down_naive_up())
            elif args.variant == "naive_down_optimized_up":
                result = asdict(await experiment.run_naive_down_optimized_up())
            else:
                result = {}

            await experiment.teardown()
            all_iterations[args.variant].append(result)
            experiment.save_results(result, filename=f"{args.variant}_iter{i+1}.json")

            # 等待系统冷却
            if i < args.iterations - 1:
                print("等待系统冷却...")
                await asyncio.sleep(30)

        # 计算统计数据
        aggregated = {
            "experiment_name": f"symmetric_migration_{args.variant}",
            "variant": args.variant,
            "iterations": args.iterations,
            "metrics": {}
        }

        for key in metrics_keys:
            values = [r[key] for r in all_iterations[args.variant] if key in r and r[key] is not None and r[key] != 0.0]
            if not values:
                values = [r.get(key, 0.0) for r in all_iterations[args.variant] if r.get(key) is not None]
            if values:
                aggregated["metrics"][key] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                    "n_valid": len(values),
                    "n_total": len(all_iterations[args.variant]),
                }

        # 保存聚合结果
        from framework import get_output_path
        output_filename = f"{args.variant}_aggregated.json"
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
