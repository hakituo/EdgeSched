import asyncio
import time
import argparse
import logging
import json
import statistics
import os
import sys
import subprocess
import aiohttp
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(current_dir)
mvp_core_path = os.path.dirname(experiments_dir)
legacy_root = os.path.dirname(mvp_core_path)
repo_root = os.path.dirname(legacy_root)
sys.path.append(repo_root)
sys.path.append(legacy_root)
sys.path.append(mvp_core_path) # Add mvp_core to path for 'domain' and 'data' imports

# Configure Logging
from mvp_core.utils.logger import setup_logger, TraceFormatter  # noqa: E402
from mvp_core.utils.trace_context import TraceContext  # noqa: E402

# 实验日志属于本项目运行产物，避免写入父仓库日志目录。
log_dir = os.path.join(mvp_core_path, "experiment_results", "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"benchmark_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Setup File Logging for mvp_core and Benchmark
def add_file_handler(logger_name, file_path):
    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(logging.DEBUG) # Ensure logger captures DEBUG
    
    # Check if file handler already exists to avoid duplicates
    for h in target_logger.handlers:
        if isinstance(h, logging.FileHandler):
            return

    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # File gets detailed traces
    
    # Use TraceFormatter
    fmt_str = '[%(asctime)s] [%(levelname)s] [TraceID:%(trace_id)s] [%(name)s] %(message)s'
    formatter = TraceFormatter(fmt_str, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    target_logger.addHandler(file_handler)

# Configure loggers
logger = setup_logger("ComprehensiveBenchmark", level=logging.INFO) # Console stays INFO
add_file_handler("ComprehensiveBenchmark", log_file)
add_file_handler("mvp_core", log_file)

print(f"Logging traces to: {log_file}")

from mvp_core.config import get_settings  # noqa: E402

# ================= 1. 配置区 (Config) =================
settings = get_settings()

def get_abs_path(path: Optional[str]) -> str:
    if not path:
        return "dummy_path"
    if os.path.isabs(path):
        return path
    
    # Check in repo root first
    path_in_root = os.path.join(repo_root, path)
    if os.path.exists(path_in_root):
        return path_in_root
        
    # Check in mvp_core path (if self-contained)
    path_in_mvp = os.path.join(mvp_core_path, path)
    if os.path.exists(path_in_mvp):
        return path_in_mvp
        
    # Default to project root if neither found (or let it fail later)
    return path_in_root

CONFIG = {
    "llm_path": get_abs_path(settings.model.text_path or "models/llm/Qwen2___5-7B-Instruct-Q4_K_M.gguf"),
    "sd_path": get_abs_path(os.environ.get("XIAOYOU_BENCH_SD_PATH") or settings.model.sd_path),
    "vl_path": get_abs_path(settings.model.vl_path),
    "tts_api": settings.model.tts_api
}

def cpu_bound_task_blocking(duration: float):
    """
    Simulates a CPU-intensive task (e.g., Matrix Mul) that BLOCKS the thread.
    Used for Mock LLM/VL simulation.
    """
    end_time = time.time() + duration
    count = 0
    while time.time() < end_time:
        count += 1
        _ = count * count
    return count

# ================= 2. 模拟组件区 (Mock Adapters) =================

class MockLLMAdapter:
    async def generate(self, prompt: str, **kwargs):
        # Simulate CPU blocking (Inference is CPU heavy on edge if not fully offloaded)
        # 0.2s blocking to simulate token generation lag on main thread if naive
        cpu_bound_task_blocking(0.1) 
        await asyncio.sleep(0.05) # Some non-blocking wait
        return "Mock LLM Response"

class MockVLAdapter:
    async def analyze_image(self, image_path: str, prompt: str):
        # Simulate Preprocessing (CPU blocking) + Inference (GPU/Wait)
        cpu_bound_task_blocking(0.05)
        await asyncio.sleep(0.1)
        return "Mock Image Description"

class MockTTSAdapter:
    async def synthesize(self, text: str):
        # Simulate Network IO latency
        await asyncio.sleep(0.2)
        return b"fake_audio_bytes"

class MockSDAdapter:
    def generate_image(self, prompt: str, **kwargs):
        # Simulate Heavy Blocking (Traditional Architecture Pain Point)
        # Blocks the Main Thread!
        cpu_bound_task_blocking(0.5) 
        return {"status": "success", "images": []}

# ================= 3. 真实组件加载区 (Real Imports) =================

class RealTTSAdapter:
    def __init__(self, api_url):
        self.api_url = api_url

    async def synthesize(self, text: str):
        try:
            # Try to find ref audio in repo root or mvp_core
            ref_audio_path = os.path.join(repo_root, "ref_audio", "female", "ref_calm.wav")
            if not os.path.exists(ref_audio_path):
                 ref_audio_path = os.path.join(mvp_core_path, "ref_audio", "female", "ref_calm.wav")
            
            # If still not found, use a dummy path (API might fail if strict, but better than crashing here)
            if not os.path.exists(ref_audio_path):
                logger.warning(f"Reference audio not found at {ref_audio_path}")

            params = {
                "text": text,
                "text_language": "zh",
                "refer_wav_path": ref_audio_path,
                "prompt_language": "zh",
                "prompt_text": "这是中文纯语音测试，不包含英文内容",
                "media_type": "wav"
            }
            
            timeout = aiohttp.ClientTimeout(total=15, connect=2, sock_connect=2, sock_read=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # GPT-SoVITS default API
                # Ensure URL is correct
                url = self.api_url
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        # Use read() which handles reading the whole body
                        # If TransferEncodingError occurs, it might be due to server closing connection early
                        data = await resp.read()
                        if not data:
                            logger.error("TTS returned empty data")
                            return None
                        return data
                    else:
                        logger.error(f"TTS Error {resp.status}: {await resp.text()}")
        except aiohttp.ClientPayloadError as e:
            logger.error(f"TTS Payload Error (TransferEncoding?): {e}")
        except Exception as e:
            logger.error(f"TTS Request Failed: {e}")
        return None

def load_real_adapters():
    """延迟加载：只有在需要时才 import 那些巨大的库""" 
    logger.info("正在加载真实模型驱动 (Local MVP)...") 
    try:
        # Import LLM
        from mvp_core.data.adapters.gguf_llm_adapter import GGUFLLMAdapter
        
        # Import SD
        from mvp_core.data.adapters.sd_adapter import SDAdapter
        
        # Import VL
        from mvp_core.data.adapters.vl_adapter import VLAdapter

        return GGUFLLMAdapter, SDAdapter, VLAdapter
            
    except ImportError as e:
        logger.error(f"无法加载真实模型依赖: {e}") 
        logger.error("请检查环境或使用 --workload mock") 
        return None, None, None

# ================= 4. 上下文/工厂模式 (Context/Factory) =================

class ExperimentContext:
    def __init__(self, mode: str, workload: str):
        self.mode = mode
        self.workload = workload
        self.llm = None
        self.vl = None
        self.tts = None
        self.sd = None
        self.scheduler = None
        self.resource_manager = None
        self.TaskType = None
        self.TaskPriority = None
        self.tts_mode = "unknown"

    async def setup(self):
        # 1. Initialize Adapters
        if self.workload == "mock":
            logger.info("\n=== 初始化 MOCK 模拟环境 ===") 
            self.llm = MockLLMAdapter()
            self.vl = MockVLAdapter()
            self.tts = MockTTSAdapter()
            self.tts_mode = "mock"
            try:
                if bool(getattr(getattr(settings, "scheduler", None), "use_cpp_for_image", False)):
                    from mvp_core.data.adapters.synthetic_image_adapter import SyntheticImageAdapter

                    self.sd = SyntheticImageAdapter(use_cpp_scheduler=True)
                else:
                    self.sd = MockSDAdapter()
            except Exception:
                self.sd = MockSDAdapter()
        
        elif self.workload == "real":
            logger.info("\n=== 初始化 REAL 真实环境 (全量化模型) ===") 
            GGUFLLMAdapter, SDAdapter, RealQwen2VLAdapter = load_real_adapters()

            if GGUFLLMAdapter is not None:
                try:
                    self.llm = GGUFLLMAdapter(CONFIG['llm_path'], n_gpu_layers=0)
                except Exception as e:
                    logger.error(f"真实 LLM 加载失败，回退到 MockLLM: {e}")
                    self.llm = MockLLMAdapter()
            else:
                self.llm = MockLLMAdapter()
            self.tts = RealTTSAdapter(CONFIG['tts_api'])
            self.tts_mode = "real"
            
            # Ensure test image exists for VL
            if not os.path.exists("test.jpg"):
                try:
                    from PIL import Image
                    img = Image.new('RGB', (224, 224), color = 'red')
                    img.save("test.jpg")
                except ImportError:
                    pass

            if RealQwen2VLAdapter is not None:
                try:
                    self.vl = RealQwen2VLAdapter(CONFIG['vl_path'])
                except Exception:
                    self.vl = MockVLAdapter()
            else:
                self.vl = MockVLAdapter()
                
            # Config for SDAdapter
            sd_conf = {
                'model_type': 'stable_diffusion',
                'sd_model_path': CONFIG['sd_path'],
                'device': 'auto', # Use auto to enable GPU if available
                'quantization': {
                    'enabled': True,
                    'precision_level': 'fp16'
                },
                'generation': {
                    'low_vram_mode': True, # Enable low vram mode (uses model offload)
                    'width': 512,
                    'height': 512,
                    'num_inference_steps': 4,
                    'local_files_only': True
                }
            }
            if SDAdapter is not None:
                try:
                    self.sd = SDAdapter(sd_conf)
                except Exception:
                    self.sd = MockSDAdapter()
            else:
                self.sd = MockSDAdapter()

            try:
                import aiohttp

                timeout = aiohttp.ClientTimeout(total=2, connect=1, sock_connect=1, sock_read=2)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(CONFIG['tts_api'], params={"text": "ping"}) as resp:
                        _ = resp.status
            except Exception:
                self.tts = MockTTSAdapter()
                self.tts_mode = "mock_fallback"
        
        logger.info("模型加载完成")

        try:
            from mvp_core.services.resource_manager import ResourceManager

            self.resource_manager = ResourceManager(llm=self.llm, vl=self.vl, sd=self.sd)
        except Exception:
            self.resource_manager = None

        # 2. Initialize Scheduler (if needed)
        if self.mode == 'xy_core':
            try:
                from mvp_core.services.task_scheduler import GlobalTaskScheduler, TaskPriority, TaskType
                self.scheduler = GlobalTaskScheduler()
                self.TaskType = TaskType
                self.TaskPriority = TaskPriority
                await self.scheduler.start()
                logger.info("调度器已启动")
            except ImportError:
                logger.error("GlobalTaskScheduler not found!")
                sys.exit(1)

    async def teardown(self):
        if self.scheduler:
            await self.scheduler.stop()
            logger.info("调度器已停止")

# ================= 5. 实验逻辑区 (Experiment Runner) =================

class ExperimentRunner:
    def __init__(self, context: ExperimentContext):
        self.ctx = context
        self.results = {}

    # --- Task Logic Helpers ---

    async def _run_mixed_task_naive(self, task_id: int) -> float:
        """Naive Async: Everything runs in main loop. Returns duration in seconds."""
        start = time.time()
        if self.ctx.resource_manager:
            await self.ctx.resource_manager.prepare_for_heavy_task("llm")
        await self.ctx.llm.generate("Hello", max_tokens=64)
        await self.ctx.tts.synthesize("Hello world")
        if self.ctx.resource_manager:
            await self.ctx.resource_manager.prepare_for_heavy_task("vl")
        await self.ctx.vl.analyze_image("test.jpg", "Describe")
        
        # Polymorphic handling for SD (Sync Mock vs Async Real)
        if self.ctx.resource_manager:
            await self.ctx.resource_manager.prepare_for_heavy_task("sd")
        res = self.ctx.sd.generate_image("A cat")
        if asyncio.iscoroutine(res):
            await res
            
        return time.time() - start

    async def _run_mixed_task_serial(self, task_id: int) -> float:
        """Serial: Blocking calls. Returns duration in seconds."""
        start = time.time()
        
        if self.ctx.workload == 'mock':
            cpu_bound_task_blocking(0.1) # LLM
            time.sleep(0.2) # TTS
            cpu_bound_task_blocking(0.1) # VL
            cpu_bound_task_blocking(0.3) # SD
        else:
            # Real workload - run sequentially but using async adapters
            if self.ctx.resource_manager:
                await self.ctx.resource_manager.prepare_for_heavy_task("llm")
            await self.ctx.llm.generate("Hello", max_tokens=64)
            await self.ctx.tts.synthesize("Hello world")
            if self.ctx.resource_manager:
                await self.ctx.resource_manager.prepare_for_heavy_task("vl")
            await self.ctx.vl.analyze_image("test.jpg", "Describe")
            
            if self.ctx.resource_manager:
                await self.ctx.resource_manager.prepare_for_heavy_task("sd")
            res = self.ctx.sd.generate_image("A cat")
            if asyncio.iscoroutine(res):
                await res
                
        return time.time() - start

    async def _run_mixed_task_xycore(self, task_id: int, trace_id: Optional[str] = None) -> float:
        """xy-core: Offloaded to Scheduler. Returns duration in seconds."""
        start = time.time()
        
        # Check if C++ is enabled for routing
        settings = get_settings()
        scheduler_settings = getattr(settings, "scheduler", None)
        use_cpp_for_image = bool(getattr(scheduler_settings, "use_cpp_for_image", False))
        use_cpp_for_llm = bool(getattr(scheduler_settings, "use_cpp_for_llm", False))

        # 1. LLM
        if use_cpp_for_llm:
            # Use High-Level API for C++ Routing
            await self.ctx.scheduler.schedule_llm_generation("Hello", max_tokens=64)
        else:
            # Use generic schedule_task for Python backend
            async def run_llm():
                if self.ctx.resource_manager:
                    await self.ctx.resource_manager.prepare_for_heavy_task("llm")
                return await self.ctx.llm.generate("Hello", max_tokens=64)
                
            t1 = await self.ctx.scheduler.schedule_task(
                func=run_llm if self.ctx.workload == 'real' else cpu_bound_task_blocking,
                args=() if self.ctx.workload == 'real' else (0.1,),
                name=f"llm_{task_id}",
                priority=self.ctx.TaskPriority.HIGH,
                task_type=self.ctx.TaskType.DEFAULT if self.ctx.workload == 'real' else self.ctx.TaskType.CPU_BOUND,
                trace_id=trace_id
            )
            await (await self.ctx.scheduler.get_task_future(t1))

        # 2. TTS (IO_BOUND)
        async def run_tts():
            if self.ctx.resource_manager:
                # Simulate preparation if needed, or just log
                pass
            return await self.ctx.tts.synthesize("Hello world")

        t2 = await self.ctx.scheduler.schedule_task(
            func=run_tts,
            name=f"tts_{task_id}",
            priority=self.ctx.TaskPriority.MEDIUM,
            task_type=self.ctx.TaskType.DEFAULT,
            trace_id=trace_id
        )
        await (await self.ctx.scheduler.get_task_future(t2))

        # 3. VL (GPU_BOUND) - Always generic Python for now as C++ doesn't support VL yet
        async def run_vl():
            if self.ctx.resource_manager:
                await self.ctx.resource_manager.prepare_for_heavy_task("vl")
            return await self.ctx.vl.analyze_image("test.jpg", "Describe")

        t3 = await self.ctx.scheduler.schedule_task(
            func=run_vl if self.ctx.workload == 'real' else cpu_bound_task_blocking,
            args=() if self.ctx.workload == 'real' else (0.05,),
            name=f"vl_{task_id}",
            priority=self.ctx.TaskPriority.MEDIUM,
            task_type=self.ctx.TaskType.DEFAULT if self.ctx.workload == 'real' else self.ctx.TaskType.GPU_BOUND,
            trace_id=trace_id
        )
        await (await self.ctx.scheduler.get_task_future(t3))

        # 4. SD (GPU_BOUND / CPU_BOUND if Mock)
        if use_cpp_for_image:
            # Use High-Level API for C++ Routing
            await self.ctx.scheduler.schedule_image_generation("A cat", steps=4)
        else:
            # Original Generic Scheduling Logic
            is_async_sd = asyncio.iscoroutinefunction(self.ctx.sd.generate_image) or self.ctx.workload == 'real'
            
            if is_async_sd:
                async def run_sd():
                    if self.ctx.resource_manager:
                        await self.ctx.resource_manager.prepare_for_heavy_task("sd")
                    res = self.ctx.sd.generate_image("A cat")
                    if asyncio.iscoroutine(res):
                        res = await res
                    
                    # Save image for verification
                    try:
                        if isinstance(res, dict) and res.get('status') == 'success' and res.get('images'):
                            img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiment_results', 'generated_images')
                            os.makedirs(img_dir, exist_ok=True)
                            timestamp = int(time.time() * 1000)
                            for i, img in enumerate(res['images']):
                                img_path = os.path.join(img_dir, f"sd_task_{task_id}_{timestamp}_{i}.png")
                                img.save(img_path)
                                logger.info(f"Saved generated image to {img_path}")
                    except Exception as e:
                        logger.error(f"Failed to save image: {e}")
                    return res
                    
                t4 = await self.ctx.scheduler.schedule_task(
                    func=run_sd,
                    args=(),
                    name=f"sd_{task_id}",
                    priority=self.ctx.TaskPriority.LOW, 
                    task_type=self.ctx.TaskType.DEFAULT, # Scheduler runs async func in loop
                    trace_id=trace_id
                )
            else:
                # Sync Mock - Schedule as CPU_BOUND to offload to ThreadPool
                t4 = await self.ctx.scheduler.schedule_task(
                    func=self.ctx.sd.generate_image,
                    args=("A cat",),
                    name=f"sd_{task_id}",
                    priority=self.ctx.TaskPriority.LOW, 
                    task_type=self.ctx.TaskType.CPU_BOUND, # Scheduler runs sync func in executor
                    trace_id=trace_id
                )

            await (await self.ctx.scheduler.get_task_future(t4))
        
        return time.time() - start

    async def _run_mixed_task(self, task_id: int) -> float:
        """Unified Task Runner. Returns duration."""
        # Start a new trace for this request
        new_id = TraceContext.generate_trace_id()
        trace_token = TraceContext.set_trace_id(new_id)
        try:
            # logger.info(f"Starting request {task_id}")
            if self.ctx.mode == 'single_thread':
                return await self._run_mixed_task_serial(task_id)
            elif self.ctx.mode == 'naive_async':
                return await self._run_mixed_task_naive(task_id)
            elif self.ctx.mode == 'xy_core':
                return await self._run_mixed_task_xycore(task_id, trace_id=new_id)
            return 0.0
        finally:
            TraceContext.reset_trace_id(trace_token)

    # --- Experiments ---

    def _calc_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate detailed metrics from a list of latencies (seconds)."""
        if not latencies:
            return {}
        
        n = len(latencies)
        avg = statistics.mean(latencies)
        
        try:
            if n < 2:
                # Handle single data point case
                p50 = latencies[0]
                p95 = latencies[0]
                p99 = latencies[0]
            else:
                # statistics.quantiles requires Python 3.8+ and at least 2 data points
                quantiles = statistics.quantiles(latencies, n=100)
                p50 = quantiles[49]
                p95 = quantiles[94]
                p99 = quantiles[98]
        except (AttributeError, statistics.StatisticsError):
            # Fallback for older python or insufficient data points
            sorted_l = sorted(latencies)
            p50 = sorted_l[int(n * 0.5)]
            p95 = sorted_l[min(int(n * 0.95), n-1)]
            p99 = sorted_l[min(int(n * 0.99), n-1)]
            
        return {
            "n": n,
            "avg_ms": avg * 1000,
            "p50_ms": p50 * 1000,
            "p95_ms": p95 * 1000,
            "p99_ms": p99 * 1000,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000
        }

    async def run_experiment_1_concurrency(self, concurrencies=[1, 5, 10]):
        logger.info(f"=== Exp 1: Concurrency ({self.ctx.mode}/{self.ctx.workload}) ===")
        exp_results = []
        
        # Warmup
        logger.info("Warming up...")
        try:
            await self._run_mixed_task(-1)
        except Exception:
            pass
            
        for c in concurrencies:
            logger.info(f"Running concurrency: {c}")
            start = time.time()
            task_latencies = []
            
            if self.ctx.mode == 'single_thread':
                for i in range(c): 
                    dur = await self._run_mixed_task_serial(i)
                    task_latencies.append(dur)
            else:
                # For naive/xy_core, we use asyncio.gather
                tasks = []
                for i in range(c):
                    if self.ctx.mode == 'naive_async':
                        tasks.append(self._run_mixed_task_naive(i))
                    elif self.ctx.mode == 'xy_core':
                        tasks.append(self._run_mixed_task_xycore(i))
                task_latencies = await asyncio.gather(*tasks)
            
            total_dur = time.time() - start
            rps = c / total_dur
            
            metrics = self._calc_metrics(task_latencies)
            logger.info(f"Concur: {c} | Total Time: {total_dur:.2f}s | RPS: {rps:.2f} | Avg Latency: {metrics['avg_ms']:.2f}ms")
            
            result_entry = {
                "concurrency": c, 
                "rps": rps, 
                "total_time": total_dur,
                "metrics": metrics
            }
            exp_results.append(result_entry)
        
        self.results['exp1'] = exp_results

    async def run_experiment_2_blocking(self):
        logger.info(f"=== Exp 2: Blocking Latency ({self.ctx.mode}) ===")
        lags = []
        running = True
        
        async def monitor():
            while running:
                s = time.time()
                await asyncio.sleep(0.1)
                lags.append(time.time() - s - 0.1)
        
        asyncio.create_task(monitor())
        await asyncio.sleep(0.5)
        
        # Fire a heavy SD task
        logger.info("Triggering Heavy SD Task...")
        if self.ctx.mode == 'xy_core':
            async def run_sd(): return await self.ctx.sd.generate_image("Heavy")
            tid = await self.ctx.scheduler.schedule_task(
                func=run_sd if self.ctx.workload == 'real' else lambda: cpu_bound_task_blocking(2.0),
                name="heavy_sd",
                priority=self.ctx.TaskPriority.LOW,
                task_type=self.ctx.TaskType.DEFAULT if self.ctx.workload == 'real' else self.ctx.TaskType.GPU_BOUND
            )
            await (await self.ctx.scheduler.get_task_future(tid))
        else:
            if self.ctx.workload == 'real':
                # Real SD adapter is async and handles offloading internally
                res = self.ctx.sd.generate_image("Heavy")
                if asyncio.iscoroutine(res):
                    await res
            else:
                cpu_bound_task_blocking(2.0)
                
        running = False
        
        if lags:
            metrics = self._calc_metrics(lags)
            logger.info(f"Max Lag: {metrics['max_ms']:.2f}ms | Avg Lag: {metrics['avg_ms']:.2f}ms")
            self.results['exp2'] = {"max_lag": metrics['max_ms'], "avg_lag": metrics['avg_ms'], "metrics": metrics}
        else:
            self.results['exp2'] = {"max_lag": 0, "avg_lag": 0}

    async def run_experiment_3_distribution(self, n_requests=5):
        logger.info(f"=== Exp 3: Latency Distribution ({self.ctx.mode}) ===")
        latencies = []
        
        # Warmup
        logger.info("Warming up...")
        try:
            await self._run_mixed_task(-1)
        except Exception:
            pass
            
        for i in range(n_requests):
            dur = await self._run_mixed_task(i)
            latencies.append(dur)
            await asyncio.sleep(0.05)
            
        metrics = self._calc_metrics(latencies)
        logger.info(f"Collected {len(latencies)} samples. P50: {metrics['p50_ms']:.2f}ms, P99: {metrics['p99_ms']:.2f}ms")
        
        # Explicitly unload models to demonstrate memory recovery
        if self.ctx.resource_manager:
            logger.info("TraceEvent:Type=MODEL_UNLOAD_START")
            await self.ctx.resource_manager.unload_all()
            logger.info("TraceEvent:Type=MODEL_UNLOAD_COMPLETE")
            # Wait for VRAM to drop and be captured
            await asyncio.sleep(3.0)

        self.results['exp3'] = latencies
        self.results['exp3_metrics'] = metrics

    async def run_experiment_4_stability(self, duration=30):
        logger.info(f"=== Exp 4: Stability Test ({duration}s) ===")
        start_time = time.time()
        errors = 0
        completed = 0
        latencies = []
        
        async def stress_worker(wid):
            nonlocal errors, completed
            while time.time() - start_time < duration:
                try:
                    dur = await self._run_mixed_task(wid)
                    latencies.append(dur)
                    completed += 1
                except Exception as e:
                    errors += 1
                    logger.error(f"Error in worker {wid}: {e}")
                await asyncio.sleep(0.1)

        # Run 3 concurrent workers to stress
        await asyncio.gather(*[stress_worker(i) for i in range(3)])
        
        metrics = self._calc_metrics(latencies)
        self.results['exp4'] = {
            "duration": duration, 
            "completed": completed, 
            "errors": errors,
            "metrics": metrics
        }
        logger.info(f"Stability: {completed} tasks, {errors} errors. Avg Latency: {metrics.get('avg_ms', 0):.2f}ms")

    async def run_experiment_5_vram_elasticity(self):
        """
        Experiment 5: VRAM Elasticity & Conflict Resolution.
        Simulates: LLM is active -> High priority Image request arrives -> LLM Offloads -> Image Finishes -> LLM Restores.
        """
        logger.info(f"=== Exp 5: VRAM Elasticity ({self.ctx.mode}/{self.ctx.workload}) ===")
        if self.ctx.mode != 'xy_core':
            logger.info("Skipping Exp 5: Only applicable to xy_core.")
            return {"skipped": True}

        def get_vram():
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
                return int(out.decode().split('\n')[0].strip())
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                return 0

        # 1. Warmup & Load LLM
        logger.info(f"Initial VRAM: {get_vram()}MB")
        await self._run_mixed_task(-1) 
        logger.info(f"VRAM after LLM Load: {get_vram()}MB")
        
        # 2. Start LLM Task
        logger.info("Step 2: Starting background LLM task...")
        llm_task = asyncio.create_task(self.ctx.llm.generate("Tell me a long story...", max_tokens=128))
        await asyncio.sleep(0.5) 
        
        # 3. Trigger Image Task
        logger.info("Step 3: Triggering heavy Image task (Expect VRAM Offload)...")
        start_img = time.time()
        img_res = self.ctx.sd.generate_image("A high resolution masterpiece")
        if asyncio.iscoroutine(img_res):
            img_res = await img_res
        img_dur = time.time() - start_img
        logger.info(f"VRAM during Image Task: {get_vram()}MB")
        
        # 4. Wait for LLM
        llm_res = await llm_task
        logger.info(f"VRAM after Restoration: {get_vram()}MB")
        
        result = {
            "img_duration_s": img_dur,
            "llm_success": len(llm_res) > 0 if llm_res else False,
            "img_success": img_res is not None
        }
        self.results['exp5'] = result
        return result

    def save_report(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "mode": self.ctx.mode, 
                "workload": self.ctx.workload,
                "adapters": {
                    "tts_mode": self.ctx.tts_mode,
                },
                "scheduler": {
                    "use_cpp_scheduler": bool(getattr(getattr(settings, "scheduler", None), "use_cpp", False)),
                    "use_cpp_for_llm": bool(getattr(getattr(settings, "scheduler", None), "use_cpp_for_llm", False)),
                    "use_cpp_for_image": bool(getattr(getattr(settings, "scheduler", None), "use_cpp_for_image", False)),
                    "llm_backend": str(getattr(getattr(settings, "scheduler", None), "llm_backend", "python") or "python"),
                    "image_engine_type": str(getattr(getattr(settings, "scheduler", None), "image_engine_type", "SD15_TURBO") or "SD15_TURBO"),
                    "preload_llm": bool(getattr(getattr(settings, "scheduler", None), "preload_llm", False)),
                },
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
        logger.info(f"Report saved to {filename}")

# ================= 6. 入口 (Main) =================

if __name__ == "__main__":
    # Default output path
    default_output_dir = os.path.join(os.path.dirname(__file__), '..', 'experiment_results')
    os.makedirs(default_output_dir, exist_ok=True)
    default_output_path = os.path.join(default_output_dir, 'comprehensive_results.json')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='xy_core', choices=['single_thread', 'naive_async', 'xy_core'])
    parser.add_argument('--workload', default='mock', choices=['mock', 'real'])
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--n_requests', type=int, default=5, help="Number of requests for Exp 3")
    parser.add_argument('--use_cpp_scheduler', action='store_true')
    parser.add_argument('--cpp_llm_backend', default='python', choices=['python', 'cpp'])
    parser.add_argument('--cpp_preload_llm', action='store_true')
    parser.add_argument('--use_cpp_image', action='store_true')
    parser.add_argument('--cpp_image_engine', default='SD15_TURBO', choices=['SD15_TURBO', 'SDXL_TURBO', 'MOBILE_DIFFUSION', 'SVD'])
    parser.add_argument('--stability_duration', type=int, default=30)
    parser.add_argument('--output', default=default_output_path)
    args = parser.parse_args()

    try:
        settings.scheduler.backend = "cpp" if args.use_cpp_scheduler else "python"
        settings.scheduler.use_cpp = bool(args.use_cpp_scheduler)
        settings.scheduler.use_cpp_for_llm = bool(args.use_cpp_scheduler)
        settings.scheduler.llm_backend = args.cpp_llm_backend
        settings.scheduler.preload_llm = bool(args.cpp_preload_llm)
        settings.scheduler.use_cpp_for_image = bool(args.use_cpp_scheduler) and bool(args.use_cpp_image)
        settings.scheduler.image_engine_type = str(args.cpp_image_engine or "SD15_TURBO")
    except Exception:
        pass

    if bool(args.use_cpp_scheduler) and str(args.cpp_llm_backend).lower() == "cpp":
        os.environ["XIAOYOU_ALLOW_CPP_LLM_WORKER"] = "1"

    # 1. Initialize Context
    ctx = ExperimentContext(args.mode, args.workload)
    
    # 2. Initialize Runner
    runner = ExperimentRunner(ctx)
    
    async def main():
        try:
            await ctx.setup()
            
            if args.exp in [0, 1]:
                await runner.run_experiment_1_concurrency()
            if args.exp in [0, 2]:
                await runner.run_experiment_2_blocking()
            if args.exp in [0, 3]:
                await runner.run_experiment_3_distribution(n_requests=args.n_requests)
            if args.exp in [0, 4]:
                await runner.run_experiment_4_stability(duration=int(args.stability_duration))
            if args.exp in [0, 5]:
                await runner.run_experiment_5_vram_elasticity()
            
            runner.save_report(args.output)
        finally:
            await ctx.teardown()
        
    asyncio.run(main())
