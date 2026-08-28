"""
状态背板恢复实验

测试目标：
1. 验证 mmap 持久化的正确性
2. 测量状态读写性能
3. 验证崩溃恢复后的状态一致性

对应论文章节：6.4 背板数据布局

优化版本：
- 使用统一实验框架
- 更完善的验证逻辑
- 性能基准测试
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework import (
    setup_paths, RESULTS_DIR, ExperimentConfig,
    StatisticsAnalyzer, ReproducibilityManager
)


@dataclass
class BackplaneTestConfig:
    """背板测试配置"""
    n_tasks: int = 32
    n_iterations: int = 100
    backplane_path: str = ""


class StateBackplaneRecoveryExperiment:
    """状态背板恢复实验"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reproducibility = ReproducibilityManager(seed)
        self.analyzer = StatisticsAnalyzer(confidence=0.95)
        self.results: Dict[str, Any] = {}
    
    def _setup_paths(self) -> None:
        setup_paths()
    
    def _calc_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """计算延迟统计指标"""
        stats = self.analyzer.analyze_latencies(latencies, include_ci=True)
        return stats.to_dict()
    
    def test_backplane_persistence(self, n_tasks: int = 32) -> Dict[str, Any]:
        """
        测试背板持久化正确性
        
        步骤：
        1. 创建背板并写入任务状态
        2. 关闭背板
        3. 重新打开背板
        4. 验证状态一致性
        """
        print(f"=== 背板持久化测试 (n_tasks={n_tasks}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        
        try:
            from mvp_core.services.state_backplane import StateBackplane
        except ImportError:
            from services.state_backplane import StateBackplane
        
        backplane_path = os.path.join(
            RESULTS_DIR, 
            f"test_state_backplane_{int(time.time())}.dat"
        )
        
        if os.path.exists(backplane_path):
            os.remove(backplane_path)
        
        # Phase 1: 写入状态
        print("Phase 1: 写入任务状态...")
        bp = StateBackplane(backplane_path)
        
        written_tasks: Dict[str, Dict] = {}
        for i in range(n_tasks):
            task_id = f"task_{i:04d}"
            priority = i % 4
            status = 1  # Pending 状态，不会被复用
            
            bp.register_task(task_id, priority, status)
            written_tasks[task_id] = {
                "task_id": task_id,
                "priority": priority,
                "status": status,
            }
        
        bp.close()
        print(f"  写入 {n_tasks} 个任务")
        
        # Phase 2: 读取并验证
        print("Phase 2: 重新打开并验证...")
        bp = StateBackplane(backplane_path)
        
        read_tasks = bp.get_active_tasks()
        
        # 验证一致性
        matched = 0
        mismatched = 0
        missing = 0
        
        for task_id, expected in written_tasks.items():
            found = False
            for read_task in read_tasks:
                if read_task.get("task_id") == task_id:
                    found = True
                    if (
                        read_task.get("priority") == expected["priority"] and
                        read_task.get("status") == expected["status"]
                    ):
                        matched += 1
                    else:
                        mismatched += 1
                    break
            if not found:
                missing += 1
        
        bp.close()
        
        # 清理
        if os.path.exists(backplane_path):
            os.remove(backplane_path)
        
        consistency_rate = matched / n_tasks if n_tasks > 0 else 0
        
        print(f"  匹配: {matched}, 不匹配: {mismatched}, 缺失: {missing}")
        print(f"  一致性: {consistency_rate*100:.1f}%")
        
        return {
            "test": "backplane_persistence",
            "n_tasks": n_tasks,
            "verification": {
                "matched": matched,
                "mismatched": mismatched,
                "missing": missing,
                "consistency_rate": consistency_rate,
                "is_consistent": (matched == n_tasks and mismatched == 0 and missing == 0),
            }
        }
    
    def test_backplane_performance(self, n_iterations: int = 100) -> Dict[str, Any]:
        """
        测试背板性能
        
        测量：
        1. 任务注册延迟
        2. 状态更新延迟
        3. 批量读取延迟
        """
        print(f"=== 背板性能测试 (n_iterations={n_iterations}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        
        try:
            from mvp_core.services.state_backplane import StateBackplane
        except ImportError:
            from services.state_backplane import StateBackplane
        
        backplane_path = os.path.join(
            RESULTS_DIR,
            f"bench_state_backplane_{int(time.time())}.dat"
        )
        
        if os.path.exists(backplane_path):
            os.remove(backplane_path)
        
        bp = StateBackplane(backplane_path)
        
        # 测试 1: 注册延迟
        print("测试任务注册延迟...")
        register_latencies: List[float] = []
        
        for i in range(n_iterations):
            task_id = f"perf_task_{i:04d}"
            start = time.time()
            bp.register_task(task_id, 1, 0)
            latency = time.time() - start
            register_latencies.append(latency)
        
        # 测试 2: 状态更新延迟
        print("测试状态更新延迟...")
        update_latencies: List[float] = []
        
        for i in range(n_iterations):
            task_id = f"perf_task_{i:04d}"
            start = time.time()
            bp.update_task_status(task_id, 2)
            latency = time.time() - start
            update_latencies.append(latency)
        
        # 测试 3: 批量读取延迟
        print("测试批量读取延迟...")
        read_latencies: List[float] = []
        
        for _ in range(10):
            start = time.time()
            tasks = bp.get_active_tasks()
            latency = time.time() - start
            read_latencies.append(latency)
        
        bp.close()
        
        # 清理
        if os.path.exists(backplane_path):
            os.remove(backplane_path)
        
        register_metrics = self._calc_metrics(register_latencies)
        update_metrics = self._calc_metrics(update_latencies)
        read_metrics = self._calc_metrics(read_latencies)
        
        print(f"  注册延迟: avg={register_metrics['avg_ms']:.3f}ms, "
              f"p99={register_metrics['p99_ms']:.3f}ms")
        print(f"  更新延迟: avg={update_metrics['avg_ms']:.3f}ms, "
              f"p99={update_metrics['p99_ms']:.3f}ms")
        print(f"  读取延迟: avg={read_metrics['avg_ms']:.3f}ms, "
              f"p99={read_metrics['p99_ms']:.3f}ms")
        
        return {
            "test": "backplane_performance",
            "n_iterations": n_iterations,
            "register_latency_ms": register_metrics,
            "update_latency_ms": update_metrics,
            "read_latency_ms": read_metrics,
        }
    
    def test_state_closure_strategy(self) -> Dict[str, Any]:
        """
        测试状态闭合策略
        
        验证：
        1. Running 状态的任务在恢复时被正确识别
        2. 闭合策略能正确将 Running 改为 Failed
        """
        print("=== 状态闭合策略测试 ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        
        try:
            from mvp_core.services.state_backplane import StateBackplane
        except ImportError:
            from services.state_backplane import StateBackplane
        
        backplane_path = os.path.join(
            RESULTS_DIR,
            f"closure_test_{int(time.time())}.dat"
        )
        
        if os.path.exists(backplane_path):
            os.remove(backplane_path)
        
        bp = StateBackplane(backplane_path)
        
        # 写入不同状态的任务
        n_running = 5
        n_completed = 3
        n_pending = 2
        
        for i in range(n_running):
            bp.register_task(f"running_{i}", 1, 2)
        
        for i in range(n_completed):
            bp.register_task(f"completed_{i}", 1, 3)
        
        for i in range(n_pending):
            bp.register_task(f"pending_{i}", 1, 1)
        
        bp.close()
        
        # 恢复并执行闭合
        print("恢复并执行闭合策略...")
        bp = StateBackplane(backplane_path)
        
        tasks_before = bp.get_active_tasks()
        running_before = [t for t in tasks_before if int(t.get("status", 0)) == 2]
        
        closed_count = 0
        for task in running_before:
            task_id = str(task.get("task_id", ""))
            if task_id:
                success = bp.update_task_status(task_id, 4)
                if success:
                    closed_count += 1
        
        tasks_after = bp.get_active_tasks()
        running_after = [t for t in tasks_after if int(t.get("status", 0)) == 2]
        failed_after = [t for t in tasks_after if int(t.get("status", 0)) == 4]
        
        bp.close()
        
        # 清理
        if os.path.exists(backplane_path):
            os.remove(backplane_path)
        
        print(f"  闭合前 Running: {len(running_before)}")
        print(f"  闭合后 Running: {len(running_after)}")
        print(f"  闭合后 Failed: {len(failed_after)}")
        print(f"  成功闭合: {closed_count}")
        
        return {
            "test": "state_closure_strategy",
            "before": {
                "running_count": len(running_before),
                "completed_count": n_completed,
                "pending_count": n_pending,
            },
            "after": {
                "running_count": len(running_after),
                "failed_count": len(failed_after),
            },
            "closed_count": closed_count,
            "verification": {
                "all_running_closed": len(running_after) == 0,
                "correct_failed_count": len(failed_after) == n_running,
            }
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有背板测试"""
        print("=== 状态背板完整测试套件 ===\n")
        
        # 测试 1: 持久化正确性
        persistence_results = self.test_backplane_persistence(n_tasks=32)
        
        # 测试 2: 性能
        performance_results = self.test_backplane_performance(n_iterations=100)
        
        # 测试 3: 状态闭合
        closure_results = self.test_state_closure_strategy()
        
        # 评估测试通过情况
        passed_tests = 0
        if persistence_results.get("verification", {}).get("is_consistent", False):
            passed_tests += 1
        if performance_results.get("register_latency_ms", {}).get("avg_ms", 999) < 1.0:
            passed_tests += 1
        if closure_results.get("verification", {}).get("all_running_closed", False):
            passed_tests += 1
        
        self.results = {
            "name": "state_backplane_recovery_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": self.seed,
            "tests": {
                "persistence": persistence_results,
                "performance": performance_results,
                "state_closure": closure_results,
            },
            "summary": {
                "total_tests": 3,
                "passed_tests": passed_tests,
                "all_passed": passed_tests == 3,
            }
        }
        
        return self.results
    
    def save_results(self, output_path: str):
        """保存实验结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")


def main():
    """主入口"""
    import argparse
    parser = argparse.ArgumentParser(description="状态背板恢复实验")
    parser.add_argument("--test", choices=["persistence", "performance", "closure", "all"], 
                       default="all", help="测试类型")
    parser.add_argument("--tasks", type=int, default=32, help="任务数量")
    parser.add_argument("--iterations", type=int, default=100, help="性能测试迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    exp = StateBackplaneRecoveryExperiment(seed=args.seed)
    
    if args.test == "persistence":
        results = exp.test_backplane_persistence(args.tasks)
        exp.results = {"tests": {"persistence": results}}
    elif args.test == "performance":
        results = exp.test_backplane_performance(args.iterations)
        exp.results = {"tests": {"performance": results}}
    elif args.test == "closure":
        results = exp.test_state_closure_strategy()
        exp.results = {"tests": {"state_closure": results}}
    else:
        asyncio.run(exp.run_all_tests())
    
    from framework import get_output_path

    exp.save_results(str(get_output_path("state_backplane_recovery.json", subdir="core")))


if __name__ == "__main__":
    main()
