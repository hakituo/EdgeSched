import os
import sys
import time
import json
import asyncio
from asyncio import Event

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 同时也把 legacy 目录加进去，这样 import mvp_core 就能找到了
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 这里故意保持“Python 控制面 + CPPSchedulerEngine 执行面”的标准链路：
# GlobalTaskScheduler -> CPPSchedulerEngine(backend=python) -> llama.cpp state migration
os.environ.setdefault("XIAOYOU_USE_CPP_SCHEDULER", "1")
os.environ.setdefault("XIAOYOU_USE_CPP_FOR_LLM", "1")
os.environ.setdefault("XIAOYOU_USE_CPP_FOR_IMAGE", "0")
os.environ.setdefault("XIAOYOU_CPP_LLM_BACKEND", "python")
os.environ.setdefault("XIAOYOU_CPP_WORKER_COUNT", "1")
os.environ.setdefault("XIAOYOU_BENCH_N_CTX", "2048")
os.environ.setdefault("XIAOYOU_BENCH_N_GPU_LAYERS", "-1")
os.environ.setdefault("XIAOYOU_BENCH_PRELOAD_LLM", "1")
os.environ.setdefault("XIAOYOU_MVP_IMAGE_BACKEND", "forge_then_cpp")

from mvp_core.services.task_scheduler import GlobalTaskScheduler
from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine

class RealWorkloadExperiment:
    """
    真实负载 (Real Workload Trace) 实验
    模拟真实用户交互：连续对话（LCRT任务）中，突然夹杂着生图请求（TITT任务）。
    用于收集并展示在 EdgeSched 调度下，用户的 LLM 首字延迟 (TTFT) 在突发任务前后的平滑表现。
    """
    def __init__(self, llm_path: str, image_id: str):
        # 强制使用本地模型路径，保持与 legacy config 的读取口径一致
        os.environ["XIAOYOU_BENCH_LLM_PATH"] = str(llm_path)
        os.environ["XIAOYOU_BENCH_SD_PATH"] = str(image_id)
        try:
            import config.integrated_config as integrated_config
            integrated_config.reset_settings_cache()
        except Exception:
            pass

        self.scheduler = GlobalTaskScheduler()
        self.metrics = []
        self.start_time = time.time()
        self.llm_path = llm_path
        self.image_id = image_id
        self._image_waiters = []
        self._image_started = Event()
        self._image_finished = Event()
        self._migration_done = Event()
        self._restore_done = Event()
        self._migration_started_at = None
        self._migration_finished_at = None
        self._restore_started_at = None
        self._restore_finished_at = None
        self._image_result_paths = []

    async def _record_llm_round(self, round_no: int, phase: str):
        print(f"[Time: {time.time() - self.start_time:.1f}s] User: 发送第 {round_no} 轮对话请求...")

        chat_start = time.time()
        task_id = f"chat_task_{round_no}"
        first_token_time = None

        async for _chunk in self.scheduler.submit_llm_task(
            prompt=f"这是第 {round_no} 轮聊天",
            request_id=task_id,
            max_tokens=24,
            temperature=0.3,
            first_token_timeout=60.0,
            total_timeout=180.0,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                latency = first_token_time - chat_start
                self.metrics.append({
                    "timestamp_s": time.time() - self.start_time,
                    "task_type": "LLM",
                    "latency_s": latency,
                    "round": round_no,
                    "phase": phase,
                })
                print(f"[Time: {time.time() - self.start_time:.1f}s] Assistant: 获取首字 (阶段={phase}, 延迟: {latency:.3f}s)")
                break

    async def simulate_user_chat(self):
        """模拟用户持续发送聊天消息"""
        # GPU 常驻阶段
        await self._record_llm_round(1, "gpu_before_image")
        await asyncio.sleep(2.0)
        await self._record_llm_round(2, "gpu_before_image")
        await asyncio.sleep(2.0)
        await self._record_llm_round(3, "gpu_before_image")

        # 等待图片任务真正触发迁移
        await self._migration_done.wait()
        await asyncio.sleep(0.5)

        # CPU 服务阶段
        await self._record_llm_round(4, "cpu_during_image")
        await asyncio.sleep(2.0)
        await self._record_llm_round(5, "cpu_during_image")
            
    async def simulate_burst_image_requests(self):
        """在特定的聊天回合触发突发的画图请求"""
        # 等待前三轮聊天完成后再触发图片，确保先采到 GPU 基线
        await asyncio.sleep(9.0)
        
        print(f"\n[Time: {time.time() - self.start_time:.1f}s] ⚠️ 突发事件: 用户请求生成两张高清图片！")
        self._image_started.set()

        self._migration_started_at = time.time()
        await cpp_scheduler_engine.offload_llm_to_cpu(urgent=True)
        self._migration_finished_at = time.time()
        self._migration_done.set()
        self.metrics.append({
            "timestamp_s": self._migration_finished_at - self.start_time,
            "task_type": "MIGRATION",
            "latency_s": self._migration_finished_at - self._migration_started_at,
            "phase": "gpu_to_cpu",
        })
        
        for i in range(2):
            try:
                # 提交 IMAGE 任务到真实调度器
                img_start = time.time()
                task_id, started_fut, result_fut = await self.scheduler.try_enqueue_image_generation(
                    prompt=f"生成第 {i+1} 张图片",
                    request_id=f"img_task_{i}",
                    steps=8,
                    width=1024,
                    height=1024,
                )
                
                # 异步等待结果
                async def wait_for_img(tid=task_id, sf=started_fut, rf=result_fut, start_time=img_start, idx=i):
                    try:
                        await sf
                    except Exception:
                        pass
                    result = await rf
                    
                    latency = time.time() - start_time
                    if result:
                        self._image_result_paths.append(str(result))
                    self.metrics.append({
                        "timestamp_s": time.time() - self.start_time,
                        "task_type": "IMAGE",
                        "latency_s": latency,
                        "round": idx + 1,
                        "path": str(result) if result else "",
                    })
                    print(f"[Time: {time.time() - self.start_time:.1f}s] Image Task {idx+1} 完成 (耗时: {latency:.3f}s)")
                
                self._image_waiters.append(asyncio.create_task(wait_for_img()))
                
            except Exception as e:
                print(f"提交画图任务失败: {e}")
                self.metrics.append({
                    "timestamp_s": time.time() - self.start_time,
                    "task_type": "IMAGE_ERROR",
                    "latency_s": 0.0,
                    "round": i + 1,
                    "error": str(e),
                })
                
            # 两个画图请求稍微错开
            await asyncio.sleep(1.0)

        if self._image_waiters:
            done, pending = await asyncio.wait(
                self._image_waiters,
                timeout=600.0,
            )
            for task in pending:
                task.cancel()

        self._image_finished.set()
        self._restore_started_at = time.time()
        restored = await cpp_scheduler_engine.restore_llm_to_gpu()
        self._restore_finished_at = time.time()
        self._restore_done.set()
        self.metrics.append({
            "timestamp_s": self._restore_finished_at - self.start_time,
            "task_type": "RESTORE",
            "latency_s": self._restore_finished_at - self._restore_started_at,
            "phase": "cpu_to_gpu",
            "success": bool(restored),
        })
            
    async def run(self):
        print(f"{'='*60}")
        print("Baseline 3: 真实交互负载 (Real Workload Trace) 实验")
        print("模拟: 用户在持续聊天的过程中，穿插了耗时的画图任务。")
        print(f"{'='*60}")
        
        # 启动真实调度器
        await self.scheduler.start()
        
        # 并发运行聊天流和突发画图流
        await asyncio.gather(
            self.simulate_user_chat(),
            self.simulate_burst_image_requests()
        )

        if self._restore_done.is_set():
            await asyncio.sleep(1.0)
            await self._record_llm_round(6, "gpu_after_restore")
        
        await self.scheduler.stop()
        
        # 保存结果
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_results")
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "real_workload_trace.json")
        payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **self.metrics}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            
        print(f"\n实验完成！结果已保存至: {out_file}")

if __name__ == "__main__":
    llm_path = os.environ.get("XIAOYOU_LLM_MODEL_PATH", "/home/leslie/mvp_core/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf")
    if not os.path.exists(llm_path):
        llm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/qwen2.5-7b-instruct-q4_k_m.gguf"))
        
    image_id = "/home/leslie/mvp_core/models/Image/check_point/SDXL/sd_xl_turbo_1.0_fp16.safetensors"

    experiment = RealWorkloadExperiment(llm_path, image_id)
    asyncio.run(experiment.run())
