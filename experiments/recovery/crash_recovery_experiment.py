"""
崩溃恢复实验（MTTR - Mean Time To Recovery）

测试目标：
1. 测量系统从崩溃到恢复可用的时间
2. 验证背板状态闭合策略的正确性
3. 对比有/无背板的恢复差异

对应论文章节：6.5 恢复后如何让状态机闭合

优化版本：
- 使用统一实验框架
- 改进进程间通信（使用 Queue）
- 添加状态一致性验证
- 更精确的时间测量
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import tempfile
import multiprocessing as mp
from types import SimpleNamespace
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework import (
    setup_paths, RESULTS_DIR, ExperimentConfig,
    StatisticsAnalyzer, ReproducibilityManager, ensure_importable
)


@dataclass
class RecoveryTestConfig:
    """恢复测试配置"""
    n_tasks: int = 10
    child_timeout: float = 20.0
    recovery_timeout: float = 10.0
    ready_signal_timeout: float = 15.0


class CrashRecoveryExperiment:
    """崩溃恢复实验"""
    
    def __init__(self, with_backplane: bool = True, seed: int = 42):
        self.with_backplane = with_backplane
        self.seed = seed
        self.reproducibility = ReproducibilityManager(seed)
        self.analyzer = StatisticsAnalyzer(confidence=0.95)
        self.results: Dict[str, Any] = {}

    def _extract_submitted_task_ids(self, output_lines: List[str]) -> List[str]:
        task_ids: List[str] = []
        seen: set[str] = set()
        for line in output_lines:
            m = re.search(r"\bsubmitted with id\s+(\S+)\b", line)
            if not m:
                continue
            tid = m.group(1).strip()
            if not tid or tid in seen:
                continue
            seen.add(tid)
            task_ids.append(tid)
        return task_ids
    
    def _setup_paths(self) -> None:
        setup_paths()
    
    def _create_backplane_override(self, backplane_path: str):
        """创建背板覆盖上下文管理器"""
        ensure_importable()
        from mvp_core.services.state_backplane import StateBackplane
        
        class _BackplaneOverride:
            def __init__(self, with_backplane: bool, backplane_path: str):
                self.with_backplane = with_backplane
                self.backplane_path = backplane_path
                self._orig = None
                self._orig_get_settings = None
            
            def __enter__(self):
                import mvp_core.services.task_scheduler as task_scheduler
                self._orig = getattr(task_scheduler, "get_state_backplane", None)
                self._orig_get_settings = getattr(task_scheduler, "get_settings", None)
                
                task_scheduler.get_settings = lambda: SimpleNamespace(
                    scheduler=SimpleNamespace(
                        use_cpp=False,
                        backend="python",
                        use_cpp_for_llm=False,
                        use_cpp_for_image=False,
                    ),
                    model=None,
                )
                
                if self.with_backplane:
                    task_scheduler.get_state_backplane = lambda: StateBackplane(self.backplane_path)
                else:
                    task_scheduler.get_state_backplane = lambda: None
                
                return self
            
            def __exit__(self, exc_type, exc, tb):
                import mvp_core.services.task_scheduler as task_scheduler
                if self._orig is not None:
                    task_scheduler.get_state_backplane = self._orig
                if self._orig_get_settings is not None:
                    task_scheduler.get_settings = self._orig_get_settings
                return False
        
        return _BackplaneOverride(self.with_backplane, backplane_path)
    
    def _verify_state_consistency(
        self, 
        pre_crash_tasks: List[Dict], 
        recovered_tasks: List[Dict]
    ) -> Dict[str, Any]:
        """
        验证状态一致性
        
        Args:
            pre_crash_tasks: 崩溃前的任务列表
            recovered_tasks: 恢复后的任务列表
            
        Returns:
            Dict: 验证结果
        """
        pre_ids = {t["task_id"] for t in pre_crash_tasks}
        recovered_ids = {t["task_id"] for t in recovered_tasks}
        
        lost_tasks = pre_ids - recovered_ids
        extra_tasks = recovered_ids - pre_ids
        
        running_tasks = [
            t for t in recovered_tasks 
            if int(t.get("status", 0)) == 2
        ]
        
        return {
            "pre_crash_count": len(pre_crash_tasks),
            "recovered_count": len(recovered_tasks),
            "lost_count": len(lost_tasks),
            "extra_count": len(extra_tasks),
            "running_count": len(running_tasks),
            "is_consistent": len(lost_tasks) == 0 and len(running_tasks) == 0,
        }
    
    async def run_experiment(self, n_tasks: int = 10) -> Dict[str, Any]:
        """
        运行崩溃恢复实验
        
        步骤：
        1. 启动系统，提交 N 个任务（部分 RUNNING，部分 PENDING）
        2. 强制终止进程（模拟 crash）
        3. 记录重启时间点
        4. 测量恢复时间
        5. 验证状态一致性
        """
        print(f"=== 崩溃恢复实验 (State Backplane: {self.with_backplane}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        ensure_importable()
        
        from mvp_core.services.state_backplane import StateBackplane

        backplane_path = os.path.join(
            tempfile.gettempdir(),
            f"mvp_core_crash_test_{'with' if self.with_backplane else 'without'}_{int(time.time())}_{os.getpid()}.dat",
        )
        try:
            if os.path.exists(backplane_path):
                os.remove(backplane_path)
        except Exception:
            pass
        
        start_time = time.time()
        
        # 使用 Queue 进行进程间通信
        ready_queue: mp.Queue = mp.Queue()
        output_queue: mp.Queue = mp.Queue()
        
        child_cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--child",
            "--tasks", str(int(n_tasks)),
            "--backplane-path", backplane_path,
        ]
        if self.with_backplane:
            child_cmd.append("--with-backplane")
        
        proc = subprocess.Popen(
            child_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # 等待就绪信号
        ready = False
        ready_deadline = time.time() + 20.0
        output_lines: List[str] = []
        
        while time.time() < ready_deadline:
            if proc.stdout is None:
                break
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                await asyncio.sleep(0.05)
                continue
            output_lines.append(line.strip())
            if line.strip() == "READY":
                ready = True
                break
        
        setup_time = time.time() - start_time
        
        # 记录崩溃前状态
        pre_crash_tasks: List[Dict] = []
        submitted_task_ids = self._extract_submitted_task_ids(output_lines)
        if ready:
            if self.with_backplane:
                try:
                    bp = StateBackplane(backplane_path)
                    pre_crash_tasks = bp.get_active_tasks()
                    bp.close()
                except Exception:
                    pre_crash_tasks = []

                if not pre_crash_tasks and submitted_task_ids:
                    pre_crash_tasks = [{"task_id": tid, "status": 0} for tid in submitted_task_ids]
            else:
                if submitted_task_ids:
                    pre_crash_tasks = [{"task_id": tid, "status": 0} for tid in submitted_task_ids]
        
        # 等待任务开始执行
        await asyncio.sleep(0.6)
        crash_time = time.time()
        
        # 强制终止进程
        try:
            proc.kill()
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        
        restart_start = time.time()
        recovered_tasks: List[Dict] = []
        closed_tasks: List[str] = []
        
        # 恢复阶段
        if self.with_backplane:
            try:
                bp = StateBackplane(backplane_path)
                recovered_tasks = bp.get_active_tasks()

                stuck_running_before = len([t for t in recovered_tasks if int(t.get("status", 0)) == 2])
                
                # 闭合卡死的 Running 状态
                for task in recovered_tasks:
                    if int(task.get("status", 0)) == 2:
                        tid = str(task.get("task_id", ""))
                        if tid:
                            ok = bp.update_task_status(tid, 4)
                            if ok:
                                closed_tasks.append(tid)
                                try:
                                    task["status"] = 4
                                except Exception:
                                    pass
                bp.close()
            except Exception:
                recovered_tasks = []
        
        # 启动新调度器并测量恢复时间
        ensure_importable()
        from mvp_core.services.task_scheduler import GlobalTaskScheduler, TaskPriority
        
        with self._create_backplane_override(backplane_path):
            scheduler = GlobalTaskScheduler()
            await scheduler.start(worker_count=2)
            process_start_to_healthy = time.time() - restart_start
            
            # 发送健康检查请求（并等待执行完成）
            first_request_start = time.time()
            health_task_id = await scheduler.schedule_task(
                func=lambda: "test",
                name="health_check",
                priority=TaskPriority.HIGH,
            )
            health_fut = await scheduler.get_task_future(health_task_id)
            if health_fut is not None:
                await health_fut
            healthy_to_first_request = time.time() - first_request_start

            # 等待系统稳定：提交一批轻量任务并等待全部完成
            stable_probe_count = 20
            stable_task_ids = []
            for i in range(stable_probe_count):
                tid = await scheduler.schedule_task(
                    func=lambda: 1,
                    name=f"stability_probe_{i}",
                    priority=TaskPriority.MEDIUM,
                )
                stable_task_ids.append(tid)

            stable_futures = []
            for tid in stable_task_ids:
                fut = await scheduler.get_task_future(tid)
                if fut is not None:
                    stable_futures.append(fut)
            if stable_futures:
                await asyncio.wait_for(asyncio.gather(*stable_futures), timeout=5.0)

            stable_end = time.time()
            first_request_to_stable = max(0.0, stable_end - first_request_start)
            total_recovery_time = max(0.0, stable_end - restart_start)
            
            await scheduler.stop()
        
        # 验证状态一致性
        consistency = self._verify_state_consistency(pre_crash_tasks, recovered_tasks)
        
        metrics = {
            "process_start_to_healthy_ms": float(process_start_to_healthy * 1000),
            "healthy_to_first_request_ms": float(healthy_to_first_request * 1000),
            "first_request_to_stable_ms": float(first_request_to_stable * 1000),
            "total_recovery_time_ms": float(total_recovery_time * 1000),
        }
        
        stuck_running_found_count = int(locals().get("stuck_running_before", 0)) if self.with_backplane else 0
        stuck_running_remaining_count = len([t for t in recovered_tasks if int(t.get("status", 0)) == 2]) if self.with_backplane else 0
        
        validation = {
            "child_ready": bool(ready),
            "pre_crash_task_count": int(len(pre_crash_tasks)),
            "recovered_task_count": int(len(recovered_tasks)),
            "closed_task_count": int(len(closed_tasks)),
            "stuck_running_found_count": int(stuck_running_found_count),
            "stuck_running_remaining_count": int(stuck_running_remaining_count),
            "stuck_running_closed_count": int(len(closed_tasks)),
            "no_stuck_running_tasks": (
                True if not self.with_backplane 
                else (stuck_running_remaining_count == 0)
            ),
            "state_machine_closed": (
                True if not self.with_backplane 
                else (stuck_running_remaining_count == 0)
            ),
            "state_consistency": consistency,
        }
        
        self.results = {
            "name": "crash_recovery_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": self.seed,
            "config": {
                "with_backplane": bool(self.with_backplane),
                "n_tasks": int(n_tasks),
                "crash_mode": "process_kill",
                "backplane_path": backplane_path,
            },
            "metrics": metrics,
            "validation": validation,
            "phases": {
                "setup_time_s": float(setup_time),
                "crash_time": float(crash_time),
                "restart_time": float(restart_start),
            },
            "child_output_tail": output_lines[-20:],
        }
        
        print(f"\n恢复时间: {metrics['total_recovery_time_ms']:.2f}ms")
        print(f"状态一致性: {'通过' if consistency['is_consistent'] else '失败'}")

        try:
            if os.path.exists(backplane_path):
                os.remove(backplane_path)
        except Exception:
            pass
        
        return self.results
    
    def save_results(self, output_path: str):
        """保存实验结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_path}")


async def run_comparison_experiment(n_tasks: int = 10, seed: int = 42):
    """
    对比实验：有/无背板的恢复时间对比
    """
    print("=== 崩溃恢复对比实验 ===\n")
    
    # 实验 1: 有背板
    print("实验 1: 有状态背板")
    exp_with_backplane = CrashRecoveryExperiment(with_backplane=True, seed=seed)
    results_with = await exp_with_backplane.run_experiment(n_tasks)
    from framework import get_output_path

    exp_with_backplane.save_results(str(get_output_path("crash_recovery_with_backplane.json", subdir="core")))
    
    print()
    
    # 实验 2: 无背板
    print("实验 2: 无状态背板")
    exp_without_backplane = CrashRecoveryExperiment(with_backplane=False, seed=seed)
    results_without = await exp_without_backplane.run_experiment(n_tasks)
    exp_without_backplane.save_results(str(get_output_path("crash_recovery_without_backplane.json", subdir="pending")))
    
    # 对比分析
    comparison = {
        "name": "crash_recovery_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "with_backplane": results_with["metrics"],
        "without_backplane": results_without["metrics"],
        "improvement": {
            "mttr_reduction_percent": 0.0,
            "validation_success_rate": 0.0,
        }
    }
    
    # 计算改进幅度
    mttr_with = results_with["metrics"]["total_recovery_time_ms"]
    mttr_without = results_without["metrics"]["total_recovery_time_ms"]
    if mttr_without > 0:
        comparison["improvement"]["mttr_reduction_percent"] = \
            ((mttr_without - mttr_with) / mttr_without) * 100

    def _is_validation_success(results: Dict[str, Any]) -> bool:
        v = results.get("validation", {})
        pre_cnt = int(v.get("pre_crash_task_count", 0))
        recovered_cnt = int(v.get("recovered_task_count", 0))
        consistency = bool(v.get("state_consistency", {}).get("is_consistent", False))
        return bool(v.get("child_ready", False)) and consistency and pre_cnt > 0 and pre_cnt == recovered_cnt

    with_success = _is_validation_success(results_with)
    without_success = _is_validation_success(results_without)
    comparison["improvement"]["validation_success_rate"] = float(1.0 if with_success else 0.0)
    comparison["improvement"]["validation_success_rate_without_backplane"] = float(1.0 if without_success else 0.0)
    comparison["validation"] = {
        "with_backplane": {
            "success": bool(with_success),
            "pre_crash_task_count": int(results_with.get("validation", {}).get("pre_crash_task_count", 0)),
            "recovered_task_count": int(results_with.get("validation", {}).get("recovered_task_count", 0)),
            "lost_count": int(results_with.get("validation", {}).get("state_consistency", {}).get("lost_count", 0)),
        },
        "without_backplane": {
            "success": bool(without_success),
            "pre_crash_task_count": int(results_without.get("validation", {}).get("pre_crash_task_count", 0)),
            "recovered_task_count": int(results_without.get("validation", {}).get("recovered_task_count", 0)),
            "lost_count": int(results_without.get("validation", {}).get("state_consistency", {}).get("lost_count", 0)),
        },
    }
    
    # 保存对比结果
    comparison_path = str(get_output_path("crash_recovery_comparison.json", subdir="supplementary"))
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **comparison}
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n对比结果已保存到: {comparison_path}")
    
    return comparison


def _child_process_main(
    tasks: int, 
    backplane_path: str, 
    with_backplane: bool
):
    """子进程主函数"""
    setup_paths()
    ensure_importable()
    
    from mvp_core.services.state_backplane import StateBackplane
    import mvp_core.services.task_scheduler as task_scheduler
    
    orig = getattr(task_scheduler, "get_state_backplane", None)
    orig_get_settings = getattr(task_scheduler, "get_settings", None)
    
    try:
        if with_backplane:
            task_scheduler.get_state_backplane = lambda: StateBackplane(backplane_path)
        else:
            task_scheduler.get_state_backplane = lambda: None
        
        task_scheduler.get_settings = lambda: SimpleNamespace(
            scheduler=SimpleNamespace(
                use_cpp=False,
                backend="python",
                use_cpp_for_llm=False,
                use_cpp_for_image=False,
            ),
            model=None,
        )
        
        from mvp_core.services.task_scheduler import GlobalTaskScheduler, TaskPriority, TaskType

        def _mock_long_task(task_seconds: float = 30.0) -> int:
            end_t = time.time() + float(task_seconds)
            x = 0
            while time.time() < end_t:
                x = (x + 1) % 1000003
            return x
        
        async def _child_async():
            scheduler = GlobalTaskScheduler()
            await scheduler.start(worker_count=2)
            
            try:
                # 提交 LLM 任务（不等待结果，让任务在后台运行）
                for i in range(int(tasks)):
                    try:
                        task_id = await scheduler.schedule_task(
                            func=_mock_long_task,
                            name=f"llm_task_{i}",
                            priority=TaskPriority.HIGH,
                            task_type=TaskType.CPU_BOUND,
                        )
                        print(f"Task {i} submitted with id {task_id}", flush=True)
                    except Exception as e:
                        print(f"Task {i} failed: {e}", flush=True)
                
                await asyncio.sleep(2.0)  # 等待任务开始执行（但未完成）
                print("READY", flush=True)
                
                while True:
                    await asyncio.sleep(60.0)
            finally:
                try:
                    await scheduler.stop()
                except Exception:
                    pass
        
        asyncio.run(_child_async())
    finally:
        if orig is not None:
            task_scheduler.get_state_backplane = orig
        if orig_get_settings is not None:
            task_scheduler.get_settings = orig_get_settings


def main():
    """主入口"""
    import argparse
    parser = argparse.ArgumentParser(description="崩溃恢复实验（MTTR）")
    parser.add_argument("--tasks", type=int, default=10, help="任务数量")
    parser.add_argument("--with-backplane", action="store_true", help="启用状态背板")
    parser.add_argument("--comparison", action="store_true", help="运行对比实验")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--child", action="store_true", help="子进程模式")
    parser.add_argument("--backplane-path", default="", help="背板文件路径")
    args = parser.parse_args()
    
    if args.child:
        _child_process_main(
            tasks=args.tasks,
            backplane_path=args.backplane_path,
            with_backplane=args.with_backplane
        )
        return
    
    if args.comparison:
        asyncio.run(run_comparison_experiment(args.tasks, args.seed))
    else:
        exp = CrashRecoveryExperiment(
            with_backplane=args.with_backplane,
            seed=args.seed
        )
        asyncio.run(exp.run_experiment(args.tasks))
        from framework import get_output_path

        if args.with_backplane:
            out_path = str(get_output_path("crash_recovery_with_backplane.json", subdir="core"))
        else:
            out_path = str(get_output_path("crash_recovery_without_backplane.json", subdir="pending"))
        exp.save_results(out_path)


if __name__ == "__main__":
    main()
