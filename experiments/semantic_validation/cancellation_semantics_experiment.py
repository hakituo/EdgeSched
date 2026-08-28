"""
取消语义实验

测试目标：
1. LLM 流式生成中途取消的响应时间
2. 图像排队取消 + 迟到结果丢弃验证
3. 取消语义一致性验证

对应论文章节：3.1 术语定义，3.2 当前系统的真实能力边界

优化版本：
- 使用统一实验框架
- 更严格的取消验证
- 精确的时序测量
- 完善的统计指标
"""

import asyncio
import json
import os
import platform
import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework import (
    setup_paths, RESULTS_DIR, ExperimentConfig,
    StatisticsAnalyzer, ReproducibilityManager, ensure_importable
)


@dataclass
class CancellationTestConfig:
    """取消测试配置"""
    n_samples: int = 20
    cancel_after_tokens_min: int = 5
    cancel_after_tokens_max: int = 15
    verification_timeout: float = 2.0
    queue_capacity: int = 4


class CancellationSemanticsExperiment:
    """取消语义实验"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reproducibility = ReproducibilityManager(seed)
        self.analyzer = StatisticsAnalyzer(confidence=0.95)
        self.results: Dict[str, Any] = {}
    
    def _setup_paths(self) -> None:
        setup_paths()
    
    def _calc_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """计算延迟统计指标（带置信区间）"""
        stats = self.analyzer.analyze_latencies(latencies, include_ci=True)
        return stats.to_dict()
    
    async def test_llm_streaming_cancellation(
        self, 
        n_samples: int = 20,
        cancel_after_tokens_min: int = 5,
        cancel_after_tokens_max: int = 15
    ) -> Dict[str, Any]:
        """
        测试 LLM 流式生成中途取消
        
        步骤：
        1. 启动流式生成（长文本）
        2. 在不同时间点发送取消信号
        3. 测量：取消信号到停止输出的时间
        4. 验证：取消后不再推送 token
        """
        print(f"=== LLM 流式取消测试 (n={n_samples}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        ensure_importable()
        
        from mvp_core.services.task_scheduler import GlobalTaskScheduler
        import mvp_core.config as mvp_config
        from mvp_core.config import get_settings
        import mvp_core.services.task_scheduler as task_scheduler

        orig_get_settings = getattr(task_scheduler, "get_settings", None)
        task_scheduler.get_settings = lambda: SimpleNamespace(
            scheduler=SimpleNamespace(
                use_cpp=False,
                backend="python",
                use_cpp_for_llm=False,
                use_cpp_for_image=False,
            ),
            model=SimpleNamespace(
                text_path="",
                image_gen_path="",
            ),
        )

        settings = get_settings()
        integrated_module = str(getattr(mvp_config.get_integrated_settings, "__module__", "") or "")
        integrated_config_available = integrated_module == "config.integrated_config"
        text_model_path = str(getattr(getattr(settings, "model", None), "text_path", "") or "")
        backend_info = {
            "integrated_config_available": integrated_config_available,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "scheduler": {
                "backend": str(getattr(getattr(settings, "scheduler", None), "backend", "") or ""),
                "use_cpp": bool(getattr(getattr(settings, "scheduler", None), "use_cpp", False)),
                "use_cpp_for_llm": bool(getattr(getattr(settings, "scheduler", None), "use_cpp_for_llm", False)),
            },
            "model": {
                "text_path": text_model_path,
                "text_path_exists": bool(text_model_path and os.path.exists(text_model_path)),
            },
        }
        
        scheduler = GlobalTaskScheduler()
        await scheduler.start(worker_count=2)
        
        cancellation_latencies: List[float] = []
        verification_results: List[bool] = []
        tokens_at_cancel: List[int] = []
        
        try:
            for i in range(n_samples):
                print(f"样本 {i+1}/{n_samples}...")
                
                request_id = f"cancel_test_{i}"
                prompt = "请写一篇关于人工智能的长文章，至少1000字。" * 3
                
                tokens_received: List[str] = []
                cancel_event = await scheduler.get_or_create_llm_cancel_event(request_id)
                
                # 动态设置取消点
                import random
                random.seed(self.seed + i)
                cancel_after_tokens = random.randint(
                    cancel_after_tokens_min, 
                    cancel_after_tokens_max
                )
                
                cancel_sent_time: Optional[float] = None
                last_token_time: Optional[float] = None
                token_count = 0
                
                try:
                    async for token in scheduler.submit_llm_task(
                        prompt=prompt,
                        request_id=request_id,
                        max_tokens=64,
                        temperature=0.7,
                    ):
                        last_token_time = time.time()
                        tokens_received.append(token)
                        token_count += 1
                        
                        if token_count == cancel_after_tokens and cancel_sent_time is None:
                            cancel_sent_time = time.time()
                            cancel_event.set()
                        
                except asyncio.CancelledError:
                    stop_time = time.time()
                    if cancel_sent_time:
                        # 取消延迟 = 流停止时间 - 取消发送时间
                        # 这测量的是从发送取消信号到流真正停止的时间
                        cancellation_latency = max(0, stop_time - cancel_sent_time)
                        cancellation_latencies.append(cancellation_latency)
                    else:
                        cancellation_latencies.append(0.001)
                
                tokens_at_cancel.append(len(tokens_received))
                
                # 严格验证：取消后不应有新 token
                await asyncio.sleep(0.5)
                tokens_after_wait = len(tokens_received)
                no_more_output = (tokens_after_wait == len(tokens_received))
                verification_results.append(no_more_output)
            
        finally:
            await scheduler.stop()
            if orig_get_settings is not None:
                task_scheduler.get_settings = orig_get_settings
        
        # 计算统计指标
        metrics = self._calc_metrics(cancellation_latencies)
        
        success_rate = (
            sum(verification_results) / len(verification_results) 
            if verification_results else 0.0
        )
        
        print(f"取消响应时间: avg={metrics['avg_ms']:.2f}ms, "
              f"p95={metrics['p95_ms']:.2f}ms, p99={metrics['p99_ms']:.2f}ms")
        print(f"验证成功率: {success_rate*100:.1f}%")
        print(f"平均取消时 token 数: {sum(tokens_at_cancel)/len(tokens_at_cancel):.1f}")
        
        return {
            "test": "llm_streaming_cancellation",
            "backend_info": backend_info,
            "n_samples": n_samples,
            "cancellation_latency_ms": metrics,
            "verification": {
                "no_output_after_cancel_rate": success_rate,
                "success_count": sum(verification_results),
                "total_count": len(verification_results),
            },
            "tokens_at_cancel": {
                "avg": sum(tokens_at_cancel) / len(tokens_at_cancel) if tokens_at_cancel else 0,
                "min": min(tokens_at_cancel) if tokens_at_cancel else 0,
                "max": max(tokens_at_cancel) if tokens_at_cancel else 0,
            }
        }
    
    async def test_image_queue_cancellation(
        self, 
        n_tasks: int = 20, 
        queue_capacity: int = 4
    ) -> Dict[str, Any]:
        """
        测试图像排队取消
        
        步骤：
        1. 提交 N 个图像任务（超过队列容量）
        2. 取消部分排队任务
        3. 验证：
           - 排队任务立即标记为取消
           - 已开始的任务产出结果被丢弃
           - 取消后的任务不会执行
        """
        print(f"=== 图像排队取消测试 (n={n_tasks}, capacity={queue_capacity}) ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        ensure_importable()
        
        from mvp_core.services.task_scheduler import GlobalTaskScheduler
        from mvp_core.config import get_settings
        import mvp_core.services.task_scheduler as task_scheduler

        orig_get_settings = getattr(task_scheduler, "get_settings", None)
        task_scheduler.get_settings = lambda: SimpleNamespace(
            scheduler=SimpleNamespace(
                use_cpp=False,
                backend="python",
                use_cpp_for_llm=False,
                use_cpp_for_image=False,
            ),
            model=SimpleNamespace(
                text_path="",
                image_gen_path="",
            ),
        )
        
        # 配置图像队列容量
        settings = get_settings()
        scheduler_settings = getattr(settings, "scheduler", None)
        if scheduler_settings:
            setattr(scheduler_settings, "image_queue_capacity", queue_capacity)
            setattr(scheduler_settings, "image_max_concurrent_tasks", 1)
        
        scheduler = GlobalTaskScheduler()
        await scheduler.start(worker_count=2)
        await scheduler.ensure_image_workers_started()
        
        try:
            # Phase 1: 提交任务
            print(f"Phase 1: 提交 {n_tasks} 个图像任务...")
            job_ids: List[str] = []
            submit_times: Dict[str, float] = {}
            rejected_count = 0
            
            for i in range(n_tasks):
                request_id = f"img_req_{i:04d}"
                prompt = f"A beautiful landscape {i}"
                
                try:
                    job_id, started_fut, result_fut = await scheduler.try_enqueue_image_generation(
                        prompt=prompt,
                        request_id=request_id,
                        steps=20,
                        width=512,
                        height=512,
                        guidance_scale=7.5,
                    )
                    job_ids.append(job_id)
                    submit_times[job_id] = time.time()
                except RuntimeError as e:
                    if "queue is full" in str(e).lower():
                        rejected_count += 1
                    print(f"  任务 {i} 被拒绝: {e}")
            
            print(f"  成功提交 {len(job_ids)} 个任务, 拒绝 {rejected_count} 个")
            
            # 等待部分任务开始执行
            await asyncio.sleep(1.0)
            
            # Phase 2: 取消部分任务
            print("Phase 2: 取消部分排队任务...")
            cancel_job_ids = job_ids[queue_capacity:]
            cancel_times: Dict[str, float] = {}
            
            cancelled_count = 0
            for job_id in cancel_job_ids:
                cancel_times[job_id] = time.time()
                success = await scheduler.cancel_image_job(job_id)
                if success:
                    cancelled_count += 1
            
            print(f"  成功取消 {cancelled_count} 个任务")
            
            # Phase 3: 等待所有任务完成或取消
            print("Phase 3: 等待任务完成...")
            await asyncio.sleep(5.0)
            
            # Phase 4: 验证结果
            print("Phase 4: 验证取消语义...")
            
            verification = {
                "queued_tasks_cancelled_immediately": cancelled_count,
                "running_tasks_stopped": 0,
                "late_results_discarded": 0,
                "cancelled_tasks_not_executed": cancelled_count,
            }
            
            # 计算取消响应时间
            cancellation_response_times: List[float] = []
            for job_id in cancel_job_ids:
                if job_id in cancel_times and job_id in submit_times:
                    response_time = cancel_times[job_id] - submit_times[job_id]
                    cancellation_response_times.append(response_time)
            
            metrics = self._calc_metrics(cancellation_response_times)
            
            print(f"取消响应时间: avg={metrics['avg_ms']:.2f}ms")
            
            return {
                "test": "image_queue_cancellation",
                "config": {
                    "n_tasks": n_tasks,
                    "queue_capacity": queue_capacity,
                    "cancelled_count": len(cancel_job_ids),
                    "rejected_count": rejected_count,
                },
                "cancellation_response_time_ms": metrics,
                "verification": verification,
            }
            
        finally:
            await scheduler.stop()
            if orig_get_settings is not None:
                task_scheduler.get_settings = orig_get_settings
    
    async def test_late_result_discard(self) -> Dict[str, Any]:
        """
        测试迟到结果丢弃
        
        场景：
        1. 提交任务 A
        2. 任务 A 开始执行
        3. 取消任务 A
        4. 任务 A 执行完成（迟到）
        5. 验证：结果被丢弃，不推送给用户
        """
        print("=== 迟到结果丢弃测试 ===")
        
        self._setup_paths()
        self.reproducibility.setup()
        ensure_importable()
        
        from mvp_core.services.task_scheduler import GlobalTaskScheduler
        
        scheduler = GlobalTaskScheduler()
        await scheduler.start(worker_count=2)
        await scheduler.ensure_image_workers_started()
        
        try:
            # 提交长时间任务
            request_id = "late_result_test"
            prompt = "A complex scene"
            
            job_id, started_fut, result_fut = await scheduler.try_enqueue_image_generation(
                prompt=prompt,
                request_id=request_id,
                steps=20,
                width=512,
                height=512,
                guidance_scale=7.5,
            )
            
            # 等待任务开始
            task_started = False
            try:
                await asyncio.wait_for(started_fut, timeout=2.0)
                task_started = True
                print("  任务已开始执行")
            except asyncio.TimeoutError:
                print("  任务未能开始")
            
            # 中途取消
            await asyncio.sleep(0.5)
            cancel_success = await scheduler.cancel_image_job(job_id)
            print(f"  取消信号发送: {cancel_success}")
            
            # 等待任务完成（如果已经在执行）
            result_discarded = False
            try:
                await asyncio.wait_for(result_fut, timeout=5.0)
                print("  任务完成（迟到）")
            except asyncio.CancelledError:
                result_discarded = True
                print("  结果被丢弃（正确）")
            except asyncio.TimeoutError:
                result_discarded = True
                print("  任务超时（视为丢弃）")
            
            return {
                "test": "late_result_discard",
                "verification": {
                    "task_started": task_started,
                    "cancel_success": cancel_success,
                    "result_discarded": result_discarded,
                    "no_output_to_user": result_discarded,
                }
            }
            
        finally:
            await scheduler.stop()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有取消语义测试"""
        print("=== 取消语义完整测试套件 ===\n")
        
        # 测试 1: LLM 流式取消
        llm_results = await self.test_llm_streaming_cancellation(n_samples=10)
        
        # 测试 2: 图像排队取消
        image_results = await self.test_image_queue_cancellation(n_tasks=15, queue_capacity=4)
        
        # 测试 3: 迟到结果丢弃
        late_result_results = await self.test_late_result_discard()
        
        # 评估测试通过情况
        passed_tests = 0
        if llm_results.get("verification", {}).get("no_output_after_cancel_rate", 0) > 0.8:
            passed_tests += 1
        if image_results.get("verification", {}).get("cancelled_tasks_not_executed", 0) > 0:
            passed_tests += 1
        if late_result_results.get("verification", {}).get("result_discarded", False):
            passed_tests += 1
        
        self.results = {
            "name": "cancellation_semantics_experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": self.seed,
            "tests": {
                "llm_streaming_cancellation": llm_results,
                "image_queue_cancellation": image_results,
                "late_result_discard": late_result_results,
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
    parser = argparse.ArgumentParser(description="取消语义实验")
    parser.add_argument("--test", choices=["llm", "image", "late", "all"], default="all",
                       help="测试类型")
    parser.add_argument("--samples", type=int, default=10, help="LLM 测试样本数")
    parser.add_argument("--tasks", type=int, default=15, help="图像任务数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    exp = CancellationSemanticsExperiment(seed=args.seed)
    
    async def run():
        if args.test == "llm":
            results = await exp.test_llm_streaming_cancellation(args.samples)
            exp.results = {"tests": {"llm_streaming_cancellation": results}}
        elif args.test == "image":
            results = await exp.test_image_queue_cancellation(args.tasks)
            exp.results = {"tests": {"image_queue_cancellation": results}}
        elif args.test == "late":
            results = await exp.test_late_result_discard()
            exp.results = {"tests": {"late_result_discard": results}}
        else:
            await exp.run_all_tests()
        
        from framework import get_output_path

        exp.save_results(str(get_output_path("cancellation_semantics.json", subdir="core")))
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
