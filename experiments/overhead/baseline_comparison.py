"""
基线对比实验

测试目标：
1. 与 Naive Async 对比（最小基线）
2. 与 ThreadPool 对比（常见工程替代）
3. 量化 EdgeSched 的改进幅度

注意：Ray 对比作为可选扩展，但非核心对比对象（定位不同）

对应论文评估章节 5.7

优化版本：
- 使用统一实验框架
- 添加内存监控
- 完善的统计对比
- 公平的对比设置
"""

import asyncio
import json
import os
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework import (
    setup_paths, RESULTS_DIR, ExperimentConfig,
    StatisticsAnalyzer, ReproducibilityManager, MockGPUTask, MockCPUTask,
    import_from_service
)


@dataclass
class ComparisonMetrics:
    """对比指标"""
    throughput_tasks_per_s: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    memory_peak_mb: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "throughput_tasks_per_s": self.throughput_tasks_per_s,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "success_rate": self.success_rate,
        }


class BaselineComparisonExperiment:
    """基线对比实验"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reproducibility = ReproducibilityManager(seed)
        self.analyzer = StatisticsAnalyzer(confidence=0.95)
        self.results: Dict[str, Any] = {}
        
        self._gpu_mock = MockGPUTask(simulate_memory=False)
        self._cpu_mock = MockCPUTask(use_real_computation=True)
    
    def _setup_paths(self) -> None:
        setup_paths()
    
    def _calc_metrics(
        self, 
        latencies: List[float], 
        duration: float,
        memory_peak: float,
        success_count: int,
        total_count: int
    ) -> ComparisonMetrics:
        """计算对比指标"""
        stats = self.analyzer.analyze_latencies(latencies, include_ci=True)
        
        return ComparisonMetrics(
            throughput_tasks_per_s=len(latencies) / duration if duration > 0 else 0,
            avg_latency_ms=stats.avg_ms,
            p50_latency_ms=stats.p50_ms,
            p95_latency_ms=stats.p95_ms,
            p99_latency_ms=stats.p99_ms,
            memory_peak_mb=memory_peak,
            success_rate=success_count / total_count if total_count > 0 else 0,
        )
    
    async def _mock_task(self, task_id: int, duration: float = 0.1):
        """模拟任务"""
        return await self._cpu_mock.execute(task_id, duration)
    
    def _mock_task_sync(self, task_id: int, duration: float = 0.1):
        """同步版本的模拟任务"""
        import time
        time.sleep(duration)
        return f"result_{task_id}"
    
    async def run_naive_async_baseline(
        self,
        n_tasks: int = 100,
        task_duration: float = 0.1
    ) -> ComparisonMetrics:
        """
        Naive Async 基线
        
        最简单的异步并发实现：
        - 无优先级
        - 无背压
        - 无资源隔离
        """
        print(f"=== Naive Async 基线 (n={n_tasks}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        
        tracemalloc.start()
        start_time = time.time()
        
        latencies: List[float] = []
        success_count = 0
        
        async def simple_task(task_id: int):
            nonlocal success_count
            task_start = time.time()
            try:
                await self._mock_task(task_id, task_duration)
                latencies.append(time.time() - task_start)
                success_count += 1
            except Exception:
                pass
        
        # 直接并发执行所有任务
        tasks = [simple_task(i) for i in range(n_tasks)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        metrics = self._calc_metrics(
            latencies, duration, peak / 1024 / 1024, success_count, n_tasks
        )
        
        print(f"  吞吐量: {metrics.throughput_tasks_per_s:.2f} tasks/s")
        print(f"  P99 延迟: {metrics.p99_latency_ms:.2f}ms")
        print(f"  内存峰值: {metrics.memory_peak_mb:.2f}MB")
        
        return metrics
    
    async def run_threadpool_baseline(
        self,
        n_tasks: int = 100,
        task_duration: float = 0.1,
        max_workers: int = 8
    ) -> ComparisonMetrics:
        """
        ThreadPool 基线
        
        常见的工程替代方案：
        - 使用线程池
        - 无优先级
        - 无背压
        """
        print(f"=== ThreadPool 基线 (n={n_tasks}, workers={max_workers}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        
        tracemalloc.start()
        start_time = time.time()
        
        latencies: List[float] = []
        success_count = 0
        lock = asyncio.Lock()
        
        def sync_task(task_id: int):
            nonlocal success_count
            task_start = time.time()
            try:
                time.sleep(task_duration)
                latency = time.time() - task_start
                latencies.append(latency)
                success_count += 1
            except Exception:
                pass
        
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, sync_task, i)
                for i in range(n_tasks)
            ]
            await asyncio.gather(*futures, return_exceptions=True)
        
        duration = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        metrics = self._calc_metrics(
            latencies, duration, peak / 1024 / 1024, success_count, n_tasks
        )
        
        print(f"  吞吐量: {metrics.throughput_tasks_per_s:.2f} tasks/s")
        print(f"  P99 延迟: {metrics.p99_latency_ms:.2f}ms")
        print(f"  内存峰值: {metrics.memory_peak_mb:.2f}MB")
        
        return metrics
    
    async def run_semaphore_controlled_baseline(
        self,
        n_tasks: int = 100,
        task_duration: float = 0.1,
        max_concurrent: int = 10
    ) -> ComparisonMetrics:
        """
        信号量控制的基线
        
        改进的异步实现：
        - 使用信号量限制并发
        - 无优先级
        - 有简单的背压
        """
        print(f"=== Semaphore 控制基线 (n={n_tasks}, max_concurrent={max_concurrent}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        
        tracemalloc.start()
        start_time = time.time()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        latencies: List[float] = []
        success_count = 0
        
        async def limited_task(task_id: int):
            nonlocal success_count
            task_start = time.time()
            try:
                async with semaphore:
                    await self._mock_task(task_id, task_duration)
                latencies.append(time.time() - task_start)
                success_count += 1
            except Exception:
                pass
        
        tasks = [limited_task(i) for i in range(n_tasks)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        metrics = self._calc_metrics(
            latencies, duration, peak / 1024 / 1024, success_count, n_tasks
        )
        
        print(f"  吞吐量: {metrics.throughput_tasks_per_s:.2f} tasks/s")
        print(f"  P99 延迟: {metrics.p99_latency_ms:.2f}ms")
        print(f"  内存峰值: {metrics.memory_peak_mb:.2f}MB")
        
        return metrics
    
    async def run_edgesched(
        self,
        n_tasks: int = 100,
        task_duration: float = 0.1
    ) -> ComparisonMetrics:
        """
        EdgeSched 系统
        
        完整功能：
        - 优先级调度
        - 背压控制
        - 资源隔离
        - 崩溃恢复
        """
        print(f"=== EdgeSched (n={n_tasks}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        
        GlobalTaskScheduler, TaskPriority, TaskType = import_from_service(
            "task_scheduler", "GlobalTaskScheduler", "TaskPriority", "TaskType"
        )
        
        tracemalloc.start()
        start_time = time.time()
        
        scheduler = GlobalTaskScheduler()
        await scheduler.start(worker_count=10)
        
        latencies: List[float] = []
        success_count = 0
        
        try:
            import random
            random.seed(self.seed)
            
            task_ids = []
            task_starts = {}
            
            for i in range(n_tasks):
                r = random.random()
                if r < 0.1:
                    priority = TaskPriority.HIGH
                elif r < 0.3:
                    priority = TaskPriority.MEDIUM
                else:
                    priority = TaskPriority.LOW
                
                try:
                    task_id = await scheduler.schedule_task(
                        func=self._mock_task_sync,
                        args=(i, task_duration),
                        name=f"comp_task_{i}",
                        priority=priority,
                        task_type=TaskType.CPU_BOUND
                    )
                    task_ids.append(task_id)
                    task_starts[task_id] = time.time()
                except Exception:
                    pass
            
            for task_id in task_ids:
                try:
                    future = await scheduler.get_task_future(task_id)
                    if future:
                        await future
                        latency = time.time() - task_starts[task_id]
                        latencies.append(latency)
                        success_count += 1
                except Exception:
                    pass
            
        finally:
            await scheduler.stop()
        
        duration = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        metrics = self._calc_metrics(
            latencies, duration, peak / 1024 / 1024, success_count, n_tasks
        )
        
        print(f"  吞吐量: {metrics.throughput_tasks_per_s:.2f} tasks/s")
        print(f"  P99 延迟: {metrics.p99_latency_ms:.2f}ms")
        print(f"  内存峰值: {metrics.memory_peak_mb:.2f}MB")
        
        return metrics
    
    async def run_full_comparison(
        self,
        n_tasks: int = 100,
        task_duration: float = 0.1
    ) -> Dict[str, Any]:
        """
        完整对比实验
        
        对比系统：
        1. Naive Async - 最小基线
        2. ThreadPool - 常见工程替代
        3. Semaphore - 改进的异步实现
        4. EdgeSched - 完整功能
        """
        print(f"=== 完整基线对比实验 (n={n_tasks}) ===\n")
        
        # 运行各系统
        print("1. Naive Async 基线")
        naive_metrics = await self.run_naive_async_baseline(n_tasks, task_duration)
        print()
        
        await asyncio.sleep(2.0)
        
        print("2. ThreadPool 基线")
        threadpool_metrics = await self.run_threadpool_baseline(n_tasks, task_duration)
        print()
        
        await asyncio.sleep(2.0)
        
        print("3. Semaphore 控制基线")
        semaphore_metrics = await self.run_semaphore_controlled_baseline(n_tasks, task_duration)
        print()
        
        await asyncio.sleep(2.0)
        
        print("4. EdgeSched")
        edgesched_metrics = await self.run_edgesched(n_tasks, task_duration)
        print()
        
        # 计算改进幅度
        def calc_improvement(baseline: float, improved: float, lower_is_better: bool = True) -> float:
            if baseline <= 0:
                return 0.0
            if lower_is_better:
                return ((baseline - improved) / baseline) * 100
            else:
                return ((improved - baseline) / baseline) * 100
        
        comparison = {
            "naive_async": naive_metrics.to_dict(),
            "threadpool": threadpool_metrics.to_dict(),
            "semaphore": semaphore_metrics.to_dict(),
            "edgesched": edgesched_metrics.to_dict(),
        }
        
        # EdgeSched vs Semaphore（最公平的对比）
        improvement = {
            "vs_semaphore": {
                "throughput_improvement_percent": calc_improvement(
                    semaphore_metrics.throughput_tasks_per_s,
                    edgesched_metrics.throughput_tasks_per_s,
                    lower_is_better=False
                ),
                "p99_latency_improvement_percent": calc_improvement(
                    semaphore_metrics.p99_latency_ms,
                    edgesched_metrics.p99_latency_ms,
                    lower_is_better=True
                ),
                "memory_overhead_percent": calc_improvement(
                    semaphore_metrics.memory_peak_mb,
                    edgesched_metrics.memory_peak_mb,
                    lower_is_better=True
                ),
            },
            "vs_naive_async": {
                "throughput_improvement_percent": calc_improvement(
                    naive_metrics.throughput_tasks_per_s,
                    edgesched_metrics.throughput_tasks_per_s,
                    lower_is_better=False
                ),
                "p99_latency_improvement_percent": calc_improvement(
                    naive_metrics.p99_latency_ms,
                    edgesched_metrics.p99_latency_ms,
                    lower_is_better=True
                ),
            },
        }
        
        self.results = {
            "name": "baseline_comparison_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": self.seed,
            "config": {
                "n_tasks": n_tasks,
                "task_duration": task_duration,
            },
            "comparison": comparison,
            "improvement": improvement,
            "summary": {
                "best_throughput": max(
                    naive_metrics.throughput_tasks_per_s,
                    threadpool_metrics.throughput_tasks_per_s,
                    semaphore_metrics.throughput_tasks_per_s,
                    edgesched_metrics.throughput_tasks_per_s,
                ),
                "best_p99_latency": min(
                    naive_metrics.p99_latency_ms,
                    threadpool_metrics.p99_latency_ms,
                    semaphore_metrics.p99_latency_ms,
                    edgesched_metrics.p99_latency_ms,
                ),
                "lowest_memory": min(
                    naive_metrics.memory_peak_mb,
                    threadpool_metrics.memory_peak_mb,
                    semaphore_metrics.memory_peak_mb,
                    edgesched_metrics.memory_peak_mb,
                ),
            }
        }
        
        return self.results
    
    def save_results(self, output_path: str):
        """保存实验结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **self.results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")


def main():
    """主入口"""
    import argparse
    parser = argparse.ArgumentParser(description="基线对比实验")
    parser.add_argument("--tasks", type=int, default=100, help="任务数量")
    parser.add_argument("--duration", type=float, default=0.1, help="单任务耗时")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    exp = BaselineComparisonExperiment(seed=args.seed)
    
    async def run():
        await exp.run_full_comparison(
            n_tasks=args.tasks,
            task_duration=args.duration
        )
        from framework import get_output_path

        exp.save_results(str(get_output_path("baseline_comparison.json", subdir="supplementary")))
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
