"""
大规模压力测试

测试目标：
1. 高并发场景（100+ 并发）
2. 长时间稳定性（1h+）
3. 资源利用率
4. 系统瓶颈分析

对应论文评估章节 5.6

优化版本：
- 使用统一实验框架
- 更真实的 Mock 任务
- 完善的资源监控
- 中间检查点支持
"""

import asyncio
import json
import os
import sys
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework import (
    setup_paths, RESULTS_DIR, ExperimentConfig,
    StatisticsAnalyzer, ReproducibilityManager, MockGPUTask, MockCPUTask,
    ensure_importable
)


@dataclass
class ResourceSnapshot:
    """资源快照"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    thread_count: int
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "thread_count": self.thread_count,
        }


@dataclass
class StressTestMetrics:
    """压力测试指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    
    latencies: List[float] = field(default_factory=list)
    resource_snapshots: List[ResourceSnapshot] = field(default_factory=list)
    
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0
    
    @property
    def throughput(self) -> float:
        return self.completed_tasks / self.duration_s if self.duration_s > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        try:
            import psutil
            self.process = psutil.Process()
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False
    
    def start(self):
        """启动监控"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                if self._psutil_available:
                    snapshot = ResourceSnapshot(
                        timestamp=time.time(),
                        cpu_percent=self.process.cpu_percent(interval=0.1),
                        memory_mb=self.process.memory_info().rss / 1024 / 1024,
                        memory_percent=self.process.memory_percent(),
                        thread_count=self.process.num_threads(),
                    )
                else:
                    import threading
                    snapshot = ResourceSnapshot(
                        timestamp=time.time(),
                        cpu_percent=0.0,
                        memory_mb=0.0,
                        memory_percent=0.0,
                        thread_count=threading.active_count(),
                    )
                self.snapshots.append(snapshot)
            except Exception as e:
                print(f"Resource monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def get_snapshots(self) -> List[ResourceSnapshot]:
        """获取快照"""
        return self.snapshots.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.snapshots:
            return {}
        
        import statistics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        mem_values = [s.memory_mb for s in self.snapshots]
        
        return {
            "cpu": {
                "avg": statistics.mean(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
            },
            "memory": {
                "avg_mb": statistics.mean(mem_values) if mem_values else 0,
                "max_mb": max(mem_values) if mem_values else 0,
                "min_mb": min(mem_values) if mem_values else 0,
            },
            "samples": len(self.snapshots),
        }


class LargeScaleStressTest:
    """大规模压力测试"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reproducibility = ReproducibilityManager(seed)
        self.analyzer = StatisticsAnalyzer(confidence=0.95)
        self.results: Dict[str, Any] = {}
        
        self._gpu_mock = MockGPUTask(simulate_memory=False)
        self._cpu_mock = MockCPUTask(use_real_computation=True)
    
    def _setup_paths(self) -> None:
        setup_paths()
    
    def _calc_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """计算延迟统计指标（带置信区间）"""
        stats = self.analyzer.analyze_latencies(latencies, include_ci=True)
        return stats.to_dict()
    
    async def _mock_task(self, task_id: int, duration: float = 0.1, use_gpu: bool = False):
        """模拟任务（更真实）"""
        if use_gpu:
            return await self._gpu_mock.execute(task_id, duration)
        else:
            return await self._cpu_mock.execute(task_id, duration)
    
    def _mock_task_sync(self, task_id: int, duration: float = 0.1, use_gpu: bool = False):
        """同步版本的模拟任务"""
        import time
        import random
        random.seed(self.seed + task_id)
        
        # 模拟 CPU 工作
        end_time = time.time() + duration
        count = 0
        while time.time() < end_time:
            count += 1
            _ = count * count
        
        return f"result_{task_id}"
    
    async def run_high_concurrency_test(
        self,
        total_tasks: int = 1000,
        concurrency: int = 100,
        task_duration: float = 0.1
    ) -> Dict[str, Any]:
        """
        高并发测试
        
        Args:
            total_tasks: 总任务数
            concurrency: 并发度
            task_duration: 单任务耗时
        """
        print(f"=== 高并发测试 (tasks={total_tasks}, concurrency={concurrency}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        ensure_importable()
        
        from mvp_core.services.task_scheduler import GlobalTaskScheduler, TaskPriority, TaskType
        
        # 启动资源监控
        monitor = ResourceMonitor(interval=1.0)
        monitor.start()
        
        # 启动调度器
        scheduler = GlobalTaskScheduler()
        await scheduler.start(worker_count=min(concurrency, 32))
        
        metrics = StressTestMetrics()
        metrics.start_time = time.time()
        metrics.total_tasks = total_tasks
        
        try:
            # 提交任务
            print(f"提交 {total_tasks} 个任务...")
            task_ids = []
            task_starts = {}
            
            import random
            random.seed(self.seed)
            
            for i in range(total_tasks):
                # 模拟不同优先级分布
                r = random.random()
                if r < 0.05:
                    priority = TaskPriority.HIGH
                elif r < 0.15:
                    priority = TaskPriority.MEDIUM
                else:
                    priority = TaskPriority.LOW
                
                # 模拟不同任务类型
                use_gpu = random.random() < 0.3
                
                try:
                    task_id = await scheduler.schedule_task(
                        func=self._mock_task_sync,
                        args=(i, task_duration, use_gpu),
                        name=f"stress_task_{i}",
                        priority=priority,
                        task_type=TaskType.GPU_BOUND if use_gpu else TaskType.CPU_BOUND
                    )
                    task_ids.append(task_id)
                    task_starts[task_id] = time.time()
                except Exception as e:
                    print(f"  任务 {i} 提交失败: {e}")
                    metrics.failed_tasks += 1
            
            print("等待任务完成...")
            
            # 等待所有任务完成
            for task_id in task_ids:
                try:
                    future = await scheduler.get_task_future(task_id)
                    if future:
                        await future
                        latency = time.time() - task_starts[task_id]
                        metrics.latencies.append(latency)
                        metrics.completed_tasks += 1
                except asyncio.CancelledError:
                    metrics.cancelled_tasks += 1
                except Exception:
                    metrics.failed_tasks += 1
            
            metrics.end_time = time.time()
            
        finally:
            # 停止监控
            monitor.stop()
            metrics.resource_snapshots = monitor.get_snapshots()
            await scheduler.stop()
        
        # 输出结果
        latency_stats = self._calc_metrics(metrics.latencies)
        resource_stats = monitor.get_stats()
        
        print("\n结果:")
        print(f"  完成任务: {metrics.completed_tasks}/{metrics.total_tasks}")
        print(f"  成功率: {metrics.success_rate*100:.1f}%")
        print(f"  吞吐量: {metrics.throughput:.2f} tasks/s")
        print(f"  平均延迟: {latency_stats['avg_ms']:.2f}ms")
        print(f"  P99 延迟: {latency_stats['p99_ms']:.2f}ms")
        if resource_stats:
            print(f"  平均 CPU: {resource_stats.get('cpu', {}).get('avg', 0):.1f}%")
            print(f"  平均内存: {resource_stats.get('memory', {}).get('avg_mb', 0):.1f}MB")
        
        return {
            "test": "high_concurrency",
            "config": {
                "total_tasks": total_tasks,
                "concurrency": concurrency,
                "task_duration": task_duration,
            },
            "metrics": {
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "cancelled_tasks": metrics.cancelled_tasks,
                "success_rate": metrics.success_rate,
                "throughput": metrics.throughput,
                "duration_s": metrics.duration_s,
            },
            "latency": latency_stats,
            "resources": resource_stats,
        }
    
    async def run_long_duration_test(
        self,
        duration_minutes: int = 60,
        arrival_rate: float = 10.0,
        checkpoint_interval_s: float = 60.0
    ) -> Dict[str, Any]:
        """
        长时间稳定性测试
        
        Args:
            duration_minutes: 测试时长（分钟）
            arrival_rate: 任务到达率（tasks/s）
            checkpoint_interval_s: 检查点间隔（秒）
        """
        print(f"=== 长时间稳定性测试 (duration={duration_minutes}min, rate={arrival_rate}tasks/s) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        ensure_importable()
        
        from mvp_core.services.task_scheduler import GlobalTaskScheduler, TaskPriority, TaskType
        
        # 启动资源监控
        monitor = ResourceMonitor(interval=5.0)
        monitor.start()
        
        # 启动调度器
        scheduler = GlobalTaskScheduler()
        await scheduler.start(worker_count=16)
        
        metrics = StressTestMetrics()
        metrics.start_time = time.time()
        
        duration_s = duration_minutes * 60
        deadline = metrics.start_time + duration_s
        
        task_counter = 0
        checkpoints: List[Dict[str, Any]] = []
        last_checkpoint = metrics.start_time
        waiter_tasks: List[asyncio.Task] = []
        
        print(f"开始持续 {duration_minutes} 分钟的压力测试...")
        
        try:
            async def task_generator():
                """任务生成器"""
                nonlocal task_counter
                import random
                random.seed(self.seed)
                
                while time.time() < deadline:
                    try:
                        use_gpu = random.random() < 0.3
                        submitted_at = time.time()
                        task_id = await scheduler.schedule_task(
                            func=self._mock_task_sync,
                            args=(task_counter, 0.1, use_gpu),
                            name=f"long_task_{task_counter}",
                            priority=TaskPriority.MEDIUM,
                            task_type=TaskType.GPU_BOUND if use_gpu else TaskType.CPU_BOUND
                        )
                        task_counter += 1
                        metrics.total_tasks += 1

                        async def _wait_one(tid, t0: float):
                            fut = await scheduler.get_task_future(tid)
                            if fut is None:
                                return
                            try:
                                await fut
                                metrics.completed_tasks += 1
                                metrics.latencies.append(time.time() - t0)
                            except asyncio.CancelledError:
                                return
                            except Exception:
                                metrics.failed_tasks += 1

                        waiter_tasks.append(asyncio.create_task(_wait_one(task_id, float(submitted_at))))
                    except Exception:
                        metrics.failed_tasks += 1
                    
                    await asyncio.sleep(1.0 / arrival_rate)
            
            async def task_monitor():
                """任务监控器（带检查点）"""
                nonlocal last_checkpoint
                
                while time.time() < deadline:
                    await asyncio.sleep(10.0)
                    
                    elapsed = time.time() - metrics.start_time
                    
                    # 检查点记录
                    if time.time() - last_checkpoint >= checkpoint_interval_s:
                        checkpoint = {
                            "elapsed_s": elapsed,
                            "total_tasks": metrics.total_tasks,
                            "completed_tasks": metrics.completed_tasks,
                            "failed_tasks": metrics.failed_tasks,
                        }
                        checkpoints.append(checkpoint)
                        last_checkpoint = time.time()
                    
                    # 每 10 秒报告一次
                    print(f"  [{elapsed/60:.1f}min] 已提交: {metrics.total_tasks}, "
                          f"完成: {metrics.completed_tasks}, 失败: {metrics.failed_tasks}")
            
            # 并发运行生成器和监控器
            await asyncio.gather(
                task_generator(),
                task_monitor(),
                return_exceptions=True
            )
            
        finally:
            try:
                if waiter_tasks:
                    await asyncio.wait_for(
                        asyncio.gather(*waiter_tasks, return_exceptions=True),
                        timeout=60.0,
                    )
            except Exception:
                pass
            metrics.end_time = time.time()
            # 停止监控
            monitor.stop()
            metrics.resource_snapshots = monitor.get_snapshots()
            await scheduler.stop()
        
        # 输出结果
        resource_stats = monitor.get_stats()
        
        print("\n结果:")
        print(f"  总任务: {metrics.total_tasks}")
        print(f"  完成任务: {metrics.completed_tasks}")
        print(f"  失败任务: {metrics.failed_tasks}")
        print(f"  平均吞吐量: {metrics.throughput:.2f} tasks/s")
        if resource_stats:
            print(f"  平均 CPU: {resource_stats.get('cpu', {}).get('avg', 0):.1f}%")
            print(f"  平均内存: {resource_stats.get('memory', {}).get('avg_mb', 0):.1f}MB")
        
        return {
            "test": "long_duration",
            "config": {
                "duration_minutes": duration_minutes,
                "arrival_rate": arrival_rate,
                "checkpoint_interval_s": checkpoint_interval_s,
            },
            "metrics": {
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "throughput": metrics.throughput,
                "duration_s": metrics.duration_s,
            },
            "resources": resource_stats,
            "checkpoints": checkpoints,
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有压力测试"""
        print("=== 大规模压力测试套件 ===\n")
        
        # 测试 1: 高并发（100 并发）
        high_concurrency_100 = await self.run_high_concurrency_test(
            total_tasks=1000,
            concurrency=100,
            task_duration=0.1
        )
        
        await asyncio.sleep(5.0)
        
        # 测试 2: 超高并发（200 并发）
        high_concurrency_200 = await self.run_high_concurrency_test(
            total_tasks=2000,
            concurrency=200,
            task_duration=0.1
        )
        
        await asyncio.sleep(5.0)
        
        # 测试 3: 长时间稳定性（10 分钟，用于快速测试；实际论文可用 60 分钟）
        long_duration = await self.run_long_duration_test(
            duration_minutes=10,
            arrival_rate=10.0
        )
        
        self.results = {
            "name": "large_scale_stress_test",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": self.seed,
            "tests": {
                "high_concurrency_100": high_concurrency_100,
                "high_concurrency_200": high_concurrency_200,
                "long_duration": long_duration,
            },
            "summary": {
                "max_concurrency_tested": 200,
                "max_duration_tested_minutes": 10,
                "total_tasks_processed": (
                    high_concurrency_100["metrics"]["total_tasks"] +
                    high_concurrency_200["metrics"]["total_tasks"] +
                    long_duration["metrics"]["total_tasks"]
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
    parser = argparse.ArgumentParser(description="大规模压力测试")
    parser.add_argument("--test", choices=["concurrency", "duration", "all"], default="all",
                       help="测试类型")
    parser.add_argument("--tasks", type=int, default=1000, help="任务数量（并发测试）")
    parser.add_argument("--concurrency", type=int, default=100, help="并发度")
    parser.add_argument("--duration", type=int, default=10, help="测试时长（分钟）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    exp = LargeScaleStressTest(seed=args.seed)
    
    async def run():
        if args.test == "concurrency":
            results = await exp.run_high_concurrency_test(
                total_tasks=args.tasks,
                concurrency=args.concurrency
            )
            exp.results = {"tests": {"high_concurrency": results}}
        elif args.test == "duration":
            results = await exp.run_long_duration_test(
                duration_minutes=args.duration
            )
            exp.results = {"tests": {"long_duration": results}}
        else:
            await exp.run_all_tests()
        
        from framework import get_output_path

        exp.save_results(str(get_output_path("large_scale_stress_test.json", subdir="supplementary")))
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
