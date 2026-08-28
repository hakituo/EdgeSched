"""
Admission Policy Comparison Experiment: Greedy vs Cost-Aware (Real System)

论文 §6.5 — 真实系统实验：对比两种准入策略。

实验方法：
  1. 启动 cpp_scheduler_engine，将 LLM 加载到 GPU
  2. 检查 Forge 服务可用性（图像生成后端）
  3. 对每种策略 (greedy / cost-aware)：
     a. 发送 N 个图像生成 burst 请求，间隔 interval_s
     b. 每个 burst 经过 AdmissionController.decide() 真实决策
     c. 执行真实 GPU 操作：offload LLM / 生成图像 / restore LLM
     d. 通过 RuntimeMetrics EMA 采集真实 offload/restore/burst 时间
  4. 对比两种策略的 burst 完成时间、LLM 中断时间等核心指标

核心指标：
  - burst_completion_s : 从 burst 提交到返回结果的总时间（含 offload）
  - llm_disruption_s   : LLM 被迫离开 GPU 的总时长
  - offload_time_s     : 真实测量的 GPU→CPU offload 耗时
  - restore_time_s     : 真实测量的 CPU→GPU restore 耗时
  - burst_time_s       : 真实图像生成耗时

依赖：
  - XIAOYOU_LLM_MODEL_PATH 指向 GGUF 模型文件
  - Forge UI 在 localhost:7860 或 7861 上运行
  - NVIDIA GPU + CUDA 环境

输出：JSON 文件，存放于 experiment_results/core/ 和 paper/experiment_data/。
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MVP_CORE_DIR = os.path.dirname(_EXPERIMENTS_DIR)

for _p in (_EXPERIMENTS_DIR, _MVP_CORE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure C++ scheduler shared libraries are findable
_SCHEDULER_LIB_DIR = os.path.join(_MVP_CORE_DIR, "cpp_scheduler", "build", "bin")
if os.path.isdir(_SCHEDULER_LIB_DIR):
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _SCHEDULER_LIB_DIR not in _ld:
        os.environ["LD_LIBRARY_PATH"] = _SCHEDULER_LIB_DIR + ":" + _ld if _ld else _SCHEDULER_LIB_DIR

from framework import (
    setup_paths,
    get_output_path,
    ReproducibilityManager,
)

# Import admission controller
try:
    from services.admission_controller import (
        AdmissionController,
        AdmissionConfig,
        AdmissionDecision,
        RuntimeMetrics,
        SystemState,
        get_admission_controller,
        get_runtime_metrics,
    )
except ImportError:
    from mvp_core.services.admission_controller import (  # type: ignore
        AdmissionController,
        AdmissionConfig,
        AdmissionDecision,
        RuntimeMetrics,
        SystemState,
        get_admission_controller,
        get_runtime_metrics,
    )

# Suppress noisy admission controller debug logs
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.disable("services.admission_controller")
    _loguru_logger.disable("mvp_core.services.admission_controller")
except ImportError:
    pass

# Image generation prompts (varied to avoid caching effects)
IMAGE_PROMPTS = [
    "a cat sitting on a windowsill at sunset, watercolor style",
    "a mountain landscape with a lake reflection, oil painting",
    "a futuristic city skyline at night with neon lights",
    "a bouquet of flowers in a ceramic vase, still life",
    "a dog playing in a park with autumn leaves",
    "an underwater scene with coral reef and tropical fish",
    "a cozy cabin in a snowy forest, illustration style",
    "a street food market in Asia, vibrant colors",
]

LLM_PROBE_PROMPT = "用一句话描述今天的天气。"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BurstResult:
    """Per-burst measurement result."""
    burst_id: int
    arrival_time_s: float
    decision: str
    requires_offload: bool
    offload_time_s: float = 0.0
    burst_time_s: float = 0.0
    restore_time_s: float = 0.0
    total_completion_s: float = 0.0
    disruption_s: float = 0.0
    llm_on_gpu_before: bool = True
    llm_on_gpu_after: bool = True
    rejected: bool = False
    gpu_free_mb_before: int = 0
    gpu_free_mb_after: int = 0


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------
class AdmissionPolicyExperiment:
    """Real system admission policy comparison experiment."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reproducibility = ReproducibilityManager(seed=seed)
        self.scheduler = None
        self.model_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH", "")
        # Use local metrics/controller to avoid singleton lock contention
        # with executor thread during LLM preload
        self.local_metrics = RuntimeMetrics()

    def setup_paths(self) -> None:
        setup_paths()

    async def setup(self) -> None:
        """Start C++ scheduler engine with LLM on GPU and verify Forge."""
        self.reproducibility.setup()
        self.setup_paths()

        if not self.model_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidates = [
                os.path.join(base_dir, "models", "llm", "qwen2.5-7b-instruct-q4_k_m.gguf"),
                os.path.join(base_dir, "models", "qwen2.5-7b-instruct-q4_k_m.gguf"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    self.model_path = c
                    break
            if not self.model_path:
                raise FileNotFoundError(f"LLM model not found, searched: {candidates}")

        print(f"[Setup] Model: {self.model_path}")

        from services.cpp_scheduler_engine import cpp_scheduler_engine

        config = {
            "model_path": self.model_path,
            "max_context_size": 4096,
            "n_gpu_layers": -1,
            "max_batch_size": 512,
        }

        cpp_scheduler_engine.start(worker_count=1, gpu_config=config, preload_llm=True)
        self.scheduler = cpp_scheduler_engine

        # Check Forge
        print("[Setup] Checking Forge availability...")
        import aiohttp
        forge_ok = False
        base_url = os.environ.get("XIAOYOU_FORGE_BASE_URL", "http://127.0.0.1:7860")
        candidate_urls = [base_url]
        if base_url == "http://127.0.0.1:7860":
            candidate_urls.append("http://127.0.0.1:7861")

        final_url = None
        for _ in range(15):
            async with aiohttp.ClientSession() as session:
                for url in candidate_urls:
                    try:
                        async with session.get(f"{url}/sdapi/v1/sd-models", timeout=5) as resp:
                            if resp.status == 200:
                                forge_ok = True
                                final_url = url
                                print(f"  Forge connected ({url})")
                                os.environ["XIAOYOU_FORGE_BASE_URL"] = url
                                break
                    except Exception:
                        pass
                if forge_ok:
                    break
            print("  Waiting for Forge...")
            await asyncio.sleep(2)

        if not forge_ok:
            raise RuntimeError(
                "Forge service not available on port 7860/7861. "
                "Start Forge before running this experiment."
            )

        # Wait for LLM ready
        print("[Setup] Waiting for LLM to be ready on GPU...")
        await self._wait_for_llm_ready(timeout_s=180.0)
        print("[Setup] Ready!")

    async def _wait_for_llm_ready(self, timeout_s: float = 180.0) -> None:
        """Wait until LLM can produce tokens."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                first = False
                async for _tok in self.scheduler.submit_llm_task("Hi", max_tokens=5, temperature=0.1):
                    first = True
                    break
                if first:
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("LLM not ready within timeout")

    async def teardown(self) -> None:
        if self.scheduler:
            try:
                await self.scheduler.stop()
            except Exception:
                pass
        print("[Teardown] Done")

    # -- Probe helpers -------------------------------------------------------

    async def probe_llm_ttft(self) -> float:
        """Measure LLM time-to-first-token in ms. Returns -1 on failure.

        IMPORTANT: We must let the async generator run to completion so
        that internal run_in_executor calls finish and aclose() succeeds.
        Early return via `return` inside `async for` causes a deadlock in
        llama_cpp's streaming generator (background thread blocks forever).
        """
        t0 = time.perf_counter()
        first_token_ms = -1.0
        try:
            async for _tok in self.scheduler.submit_llm_task(
                LLM_PROBE_PROMPT, max_tokens=16, temperature=0.3
            ):
                if first_token_ms < 0:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                # do NOT break/return — let generator finish naturally
        except Exception as e:
            print(f"  [Probe] LLM TTFT failed: {e}")
        return first_token_ms

    def _query_gpu_memory(self) -> Dict[str, Optional[float]]:
        try:
            import subprocess
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
            )
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                return {
                    "used_mb": float(parts[0]) if parts[0] else None,
                    "total_mb": float(parts[1]) if len(parts) > 1 and parts[1] else None,
                }
        except Exception:
            pass
        return {"used_mb": None, "total_mb": None}

    def _is_llm_on_gpu(self) -> bool:
        """Check if LLM is loaded on GPU."""
        if not self.scheduler:
            return False
        # Check the internal state
        force_cpu = bool(getattr(self.scheduler, "_python_force_cpu", False))
        if force_cpu:
            return False
        llm_obj = getattr(self.scheduler, "_python_llm", None)
        return llm_obj is not None

    # -- Metrics reset -------------------------------------------------------

    def _reset_metrics(self) -> None:
        """Reset local RuntimeMetrics EMA values for a fresh run."""
        with self.local_metrics._lock:
            self.local_metrics.avg_offload_time_s = None
            self.local_metrics.avg_restore_time_s = None
            self.local_metrics.avg_burst_duration_s = None
            self.local_metrics.avg_burst_vram_mb = None
            self.local_metrics.sample_count = 0
            self.local_metrics.decision_log.clear()

    # -- Core burst execution ------------------------------------------------

    async def run_one_burst(
        self,
        burst_id: int,
        controller: AdmissionController,
        prompt: str,
        image_kwargs: dict,
    ) -> BurstResult:
        """Execute a single burst through the admission controller pipeline."""
        t_arrival = time.perf_counter()

        gpu_info = self._query_gpu_memory()
        llm_gpu = self._is_llm_on_gpu()
        gpu_free = int(gpu_info.get("used_mb") or 0)

        state = SystemState(
            gpu_free_mb=gpu_free,
            gpu_total_mb=int(gpu_info.get("total_mb") or 0),
            resident_on_gpu=llm_gpu,
            queue_depth=0,
            queue_capacity=9,
            burst_running=False,
        )

        decision = controller.decide(state)

        result = BurstResult(
            burst_id=burst_id,
            arrival_time_s=t_arrival,
            decision=decision.decision.value,
            requires_offload=decision.requires_offload,
            llm_on_gpu_before=llm_gpu,
            gpu_free_mb_before=gpu_free,
        )

        if decision.decision == AdmissionDecision.REJECT:
            result.rejected = True
            print(f"    burst {burst_id}: REJECT ({decision.reason})")
            return result

        t_start = time.perf_counter()
        offload_start = None
        restore_end = None

        # Step 1: Offload LLM if required
        if decision.requires_offload and llm_gpu:
            offload_start = time.perf_counter()
            try:
                await self.scheduler.offload_llm_to_cpu()
                result.offload_time_s = time.perf_counter() - offload_start
            except Exception as e:
                print(f"    [Warning] offload failed: {e}")
                result.offload_time_s = time.perf_counter() - offload_start

        # Step 2: Run image generation via Forge (real GPU work)
        try:
            await self.scheduler.submit_image_task(
                prompt,
                skip_auto_offload=decision.requires_offload,
                **image_kwargs,
            )
            result.burst_time_s = time.perf_counter() - t_start
        except Exception as e:
            print(f"    [Warning] image generation failed: {e}")
            result.burst_time_s = time.perf_counter() - t_start

        # Step 3: Restore LLM to GPU (if we offloaded it)
        if decision.requires_offload and offload_start is not None:
            try:
                await self.scheduler.restore_llm_to_gpu()
                restore_end = time.perf_counter()
                result.restore_time_s = restore_end - t_start - result.burst_time_s
            except Exception as e:
                print(f"    [Warning] restore failed: {e}")

        result.total_completion_s = time.perf_counter() - t_start

        # Disruption = total time LLM was off GPU
        if offload_start is not None:
            disruption_end = restore_end if restore_end is not None else time.perf_counter()
            result.disruption_s = disruption_end - offload_start

        result.llm_on_gpu_after = self._is_llm_on_gpu()

        # Update GPU memory info
        gpu_after = self._query_gpu_memory()
        result.gpu_free_mb_after = int(gpu_after.get("used_mb") or 0)

        return result

    # -- Scenario runner -----------------------------------------------------

    async def run_scenario(
        self,
        policy: str,
        n_bursts: int,
        interval_s: float,
        image_kwargs: dict,
    ) -> Dict[str, Any]:
        """Run one policy scenario: sequential bursts with fixed interval."""
        cost_model_enabled = (policy == "cost_aware")
        # Use local metrics/controller to avoid singleton lock contention
        config = AdmissionConfig(enable_cost_model=cost_model_enabled)
        controller = AdmissionController(self.local_metrics, config)

        self._reset_metrics()

        print(f"\n{'=' * 60}")
        print(f"  Policy: {policy} | Bursts: {n_bursts} | Interval: {interval_s}s")
        print(f"{'=' * 60}")

        # Pre-run LLM TTFT probe
        ttft_before = await self.probe_llm_ttft()
        print(f"  LLM TTFT before: {ttft_before:.0f} ms")
        await asyncio.sleep(1)

        results: List[BurstResult] = []
        t_experiment_start = time.perf_counter()

        for i in range(n_bursts):
            prompt = IMAGE_PROMPTS[i % len(IMAGE_PROMPTS)]

            # Wait for interval (except first burst)
            if i > 0:
                await asyncio.sleep(interval_s)

            print(f"\n  [Burst {i + 1}/{n_bursts}] t={time.perf_counter() - t_experiment_start:.1f}s")

            br = await self.run_one_burst(i, controller, prompt, image_kwargs)
            results.append(br)

            print(
                f"    decision={br.decision} offload={br.requires_offload} "
                f"offload={br.offload_time_s:.2f}s burst={br.burst_time_s:.2f}s "
                f"restore={br.restore_time_s:.2f}s total={br.total_completion_s:.2f}s "
                f"disruption={br.disruption_s:.2f}s"
            )

        # Post-run LLM TTFT probe
        await asyncio.sleep(2)
        ttft_after = await self.probe_llm_ttft()
        print(f"\n  LLM TTFT after: {ttft_after:.0f} ms")

        experiment_duration = time.perf_counter() - t_experiment_start
        return self._aggregate(results, ttft_before, ttft_after, experiment_duration)

    # -- Aggregation ---------------------------------------------------------

    def _aggregate(
        self,
        results: List[BurstResult],
        ttft_before: float,
        ttft_after: float,
        total_duration: float,
    ) -> Dict[str, Any]:
        if not results:
            return {}

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
            return {
                "mean": round(statistics.mean(values), 3),
                "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "count": len(values),
            }

        offload_times = [r.offload_time_s for r in results if r.offload_time_s > 0]
        restore_times = [r.restore_time_s for r in results if r.restore_time_s > 0]
        burst_times = [r.burst_time_s for r in results if r.burst_time_s > 0]
        completion_times = [r.total_completion_s for r in results if not r.rejected]
        disruption_times = [r.disruption_s for r in results if r.disruption_s > 0]

        n_rejected = sum(1 for r in results if r.rejected)
        n_offloads = len(offload_times)

        return {
            "runtime_metrics_ema": self.local_metrics.snapshot(),
            "offload_time_s": _stats(offload_times),
            "restore_time_s": _stats(restore_times),
            "burst_time_s": _stats(burst_times),
            "total_completion_s": _stats(completion_times),
            "llm_disruption_s": _stats(disruption_times),
            "n_offloads": n_offloads,
            "n_bursts": len(results),
            "reject_rate": round(n_rejected / len(results), 3) if results else 0.0,
            "total_disruption_s": round(sum(disruption_times), 3),
            "llm_ttft_before_ms": round(ttft_before, 1),
            "llm_ttft_after_ms": round(ttft_after, 1),
            "experiment_duration_s": round(total_duration, 1),
            "throughput": round(len(completion_times) / total_duration, 4) if total_duration > 0 else 0.0,
            "per_burst": [
                {
                    "id": r.burst_id,
                    "decision": r.decision,
                    "offload": r.requires_offload,
                    "offload_s": round(r.offload_time_s, 3),
                    "burst_s": round(r.burst_time_s, 3),
                    "restore_s": round(r.restore_time_s, 3),
                    "total_s": round(r.total_completion_s, 3),
                    "disruption_s": round(r.disruption_s, 3),
                }
                for r in results
            ],
        }

    # -- Summary printer -----------------------------------------------------

    def print_summary(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        print("\n" + "=" * 80)
        print("  ADMISSION POLICY COMPARISON — REAL SYSTEM RESULTS")
        print("=" * 80)

        for policy, data in all_results.items():
            agg = data.get("aggregated", {})
            ema = agg.get("runtime_metrics_ema", {})
            print(f"\n  --- {policy.upper()} ---")
            print(f"  Runtime EMA: offload={ema.get('avg_offload_s')}s  "
                  f"restore={ema.get('avg_restore_s')}s  burst={ema.get('avg_burst_s')}s")
            print(f"  Bursts: {agg.get('n_bursts', 0)}  Offloads: {agg.get('n_offloads', 0)}  "
                  f"Reject rate: {agg.get('reject_rate', 0)}")
            print(f"  Completion (mean): {agg.get('total_completion_s', {}).get('mean', 0):.2f}s")
            print(f"  LLM disruption total: {agg.get('total_disruption_s', 0):.1f}s")
            print(f"  LLM TTFT: before={agg.get('llm_ttft_before_ms', 0):.0f}ms  "
                  f"after={agg.get('llm_ttft_after_ms', 0):.0f}ms")
            print(f"  Duration: {agg.get('experiment_duration_s', 0):.1f}s  "
                  f"Throughput: {agg.get('throughput', 0):.4f} burst/s")

        # Comparison
        greedy = all_results.get("greedy", {}).get("aggregated", {})
        cost = all_results.get("cost_aware", {}).get("aggregated", {})
        if greedy and cost:
            g_comp = greedy.get("total_completion_s", {}).get("mean", 0)
            c_comp = cost.get("total_completion_s", {}).get("mean", 0)
            g_disrupt = greedy.get("total_disruption_s", 0)
            c_disrupt = cost.get("total_disruption_s", 0)
            g_offload = greedy.get("n_offloads", 0)
            c_offload = cost.get("n_offloads", 0)

            print(f"\n  --- COMPARISON ---")
            print(f"  Completion improvement: {((g_comp - c_comp) / g_comp * 100):.1f}%" if g_comp > 0 else "")
            print(f"  Disruption reduction:   {((g_disrupt - c_disrupt) / g_disrupt * 100):.1f}%" if g_disrupt > 0 else f"  Disruption: both 0")
            print(f"  Offload reduction:      {g_offload} → {c_offload}")

        print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Admission Policy Experiment (Real System)")
    parser.add_argument("--n-bursts", type=int, default=5, help="Number of bursts per scenario")
    parser.add_argument("--interval", type=float, default=20.0, help="Inter-burst interval (s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-steps", type=int, default=20)
    parser.add_argument("--warmup-bursts", type=int, default=1, help="Warmup bursts before experiment")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Admission Policy Experiment — REAL SYSTEM")
    print("=" * 60)
    print(f"  Bursts per scenario: {args.n_bursts}")
    print(f"  Inter-burst interval: {args.interval}s")
    print(f"  Seed: {args.seed}")
    print(f"  Image: {args.image_width}x{args.image_height}, {args.image_steps} steps")
    print(f"  Warmup: {args.warmup_bursts} burst(s)")
    print()

    image_kwargs = {
        "width": args.image_width,
        "height": args.image_height,
        "steps": args.image_steps,
        "guidance_scale": 7.5,
    }

    reproducibility = ReproducibilityManager(seed=args.seed)
    reproducibility.setup()

    exp = AdmissionPolicyExperiment(seed=args.seed)

    try:
        await exp.setup()

        # --- Warmup ---
        if args.warmup_bursts > 0:
            print(f"\n[Warmup] Running {args.warmup_bursts} warmup burst(es)...")
            for w in range(args.warmup_bursts):
                try:
                    path = await exp.scheduler.submit_image_task(
                        IMAGE_PROMPTS[w % len(IMAGE_PROMPTS)], **image_kwargs
                    )
                    print(f"  Warmup {w + 1} OK: {path}")
                except Exception as e:
                    print(f"  Warmup {w + 1} failed: {e}")

        # --- LLM baseline probe ---
        print("\n[Baseline] LLM TTFT probe...")
        for i in range(3):
            ttft = await exp.probe_llm_ttft()
            print(f"  Probe {i + 1}: {ttft:.0f} ms")
            await asyncio.sleep(0.5)

        # --- Run scenarios ---
        all_results: Dict[str, Dict[str, Any]] = {}

        for policy in ["greedy", "cost_aware"]:
            data = await exp.run_scenario(
                policy=policy,
                n_bursts=args.n_bursts,
                interval_s=args.interval,
                image_kwargs=image_kwargs,
            )
            all_results[policy] = {"aggregated": data}

        # --- Print summary ---
        exp.print_summary(all_results)

        # --- Save output ---
        output = {
            "name": "admission_policy_comparison_real",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "n_bursts": args.n_bursts,
                "interval_s": args.interval,
                "seed": args.seed,
                "image": image_kwargs,
                "warmup_bursts": args.warmup_bursts,
            },
            "scenarios": all_results,
        }

        output_dir = args.output_dir or get_output_path("core")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "admission_policy_comparison.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[Output] Saved: {output_path}")

    finally:
        await exp.teardown()


if __name__ == "__main__":
    asyncio.run(main())
