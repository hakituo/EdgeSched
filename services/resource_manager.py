import asyncio
import time
import psutil
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from loguru import logger

try:
    import torch
except ImportError:
    torch = None

from mvp_core.config import get_settings

class ResourcePriority(Enum):
    CRITICAL = 0  # 核心服务 (System)
    HIGH = 1      # 直接交互 (LLM/TTS)
    MEDIUM = 2    # 辅助服务 (Vision)
    LOW = 3       # 后台任务 (Training/Indexing)

class ResourceState(Enum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3
    LOADING = 10
    READY = 11
    OFFLOADING = 12
    OFFLOADED = 13
    UNLOADED = 14
    ERROR = 15

class ResourceType(Enum):
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    CPU = "cpu"
    GPU_COMPUTE = "gpu_compute"

@dataclass
class ResourceThreshold:
    warning: float = 80.0
    critical: float = 90.0
    emergency: float = 95.0

@dataclass
class ModelResource:
    model_id: str
    model_type: str  # 'llm', 'stt', 'tts', 'image_gen', 'vision'
    priority: ResourcePriority
    
    # 资源控制回调
    load_func: Callable
    unload_func: Callable
    offload_func: Optional[Callable] = None  # 显存->内存
    
    # 状态追踪
    is_loaded: bool = False
    is_offloaded: bool = False  # 是否处于CPU Offload状态
    last_used_time: float = 0.0
    
    # 资源占用估算 (MB)
    memory_usage_mb: int = 0
    vram_usage_mb: int = 0
    
    # 实例引用 (用于直接操作)
    instance: Any = None
    
    # 当前设备
    device: str = "CPU"  # "CPU" or "GPU"

    def update_usage(self):
        self.last_used_time = time.time()

class SystemMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._thresholds = {
            ResourceType.MEMORY: ResourceThreshold(),
            ResourceType.GPU_MEMORY: ResourceThreshold(warning=75.0, critical=85.0, emergency=92.0),
            ResourceType.CPU: ResourceThreshold(warning=85.0, critical=95.0)
        }
        
    def get_memory_usage(self) -> float:
        return psutil.virtual_memory().percent

    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=None)

    def get_process_memory_usage(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    async def get_gpu_memory_usage_async(self) -> Optional[tuple[int, int]]:
        """
        Asynchronously get GPU memory usage.
        Returns: (used_mb, total_mb) or None if no GPU
        """
        if not torch or not torch.cuda.is_available():
            return None
            
        try:
            # Running torch.cuda calls in a thread executor to avoid blocking the event loop
            # strictly speaking torch.cuda calls might be blocking but usually fast.
            # For better async hygiene, wrap in to_thread.
            def _get_gpu():
                try:
                    # torch.cuda.mem_get_info returns (free, total) in bytes
                    free, total = torch.cuda.mem_get_info()
                    used = total - free
                    return int(used / 1024 / 1024), int(total / 1024 / 1024)
                except Exception:
                    return None
            
            return await asyncio.to_thread(_get_gpu)
        except Exception:
            return None

    async def get_gpu_compute_process_usage_async(self) -> Optional[Dict[int, int]]:
        """
        获取所有进程的GPU显存占用 (MB)
        Returns: {pid: used_mb}
        """
        # Note: This is hard to get purely from pytorch without pynvml.
        # We will return None if pynvml is not available or logic not implemented.
        # For MVP, we skip complex NVML integration unless necessary.
        return None

    def get_current_process_gpu_used_mb(self, pid_usage: Optional[Dict[int, int]] = None) -> int:
        # Placeholder
        return 0

    def get_resource_state(self, resource_type: ResourceType) -> ResourceState:
        usage = 0.0
        if resource_type == ResourceType.MEMORY:
            usage = self.get_memory_usage()
        elif resource_type == ResourceType.CPU:
            usage = self.get_cpu_usage()
        elif resource_type == ResourceType.GPU_MEMORY:
            # This requires async, so we might need a stored value or assume caller handles async
            # For synchronous check, we might skip or use cached
            return ResourceState.NORMAL 
            
        threshold = self._thresholds.get(resource_type)
        if not threshold:
            return ResourceState.NORMAL
            
        if usage >= threshold.emergency:
            return ResourceState.EMERGENCY
        elif usage >= threshold.critical:
            return ResourceState.CRITICAL
        elif usage >= threshold.warning:
            return ResourceState.WARNING
        return ResourceState.NORMAL

class ResourceManager:
    _instance = None
    
    TYPICAL_VRAM_USAGE = {
        "llm_engine": 12000,   # Qwen2.5-7B-Q4 ~6GB + KV Cache
        "image_gen_module": 4000, # SD1.5/SDXL Turbo ~4GB
        "vision_module": 2000, # Qwen-VL/JoyTag ~2GB
        "tts_engine": 500,     # GPT-SoVITS ~500MB
        "stt_engine": 1000,    # Whisper-large ~1GB
        "svc_engine": 500      # RVC ~500MB
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config = {
            'monitor_interval': 2.0,
            'model_unload_timeout': 300,  # 5分钟未使用自动卸载
            'cache_size_limit_mb': 1024
        }
        
        self.monitor = SystemMonitor(self.config)
        self._models: Dict[str, ModelResource] = {}
        self._resource_handlers: List[Dict] = []
        self._memory_cleanup_callbacks: List[Callable] = []
        
        self._is_running = False
        self._monitor_task = None
        self._lock = asyncio.Lock()
        self._restore_lock = asyncio.Lock()
        
        self._cache_stats = {'size': 0}
        
        # 上次清理时间
        self._last_cleanup_ts = {}
        self._cleanup_cooldown_seconds = {
            "emergency": 12.0,
            "critical": 6.0,
            "regular": 3.0
        }
        
        self._restore_tasks: Dict[str, asyncio.Task] = {}
        
        self._initialized = True
        logger.info("ResourceManager initialized (MVP Legacy Version)")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ResourceManager()
        return cls._instance

    async def start(self):
        """启动资源管理器"""
        if self._is_running:
            return
            
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        logger.info("资源管理器已启动")

    async def stop(self):
        """停止资源管理器"""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # 卸载所有模型
        await self.unload_all()

    def register_model(self, 
                      model_id: str, 
                      model_type: str = "llm", 
                      priority: ResourcePriority = ResourcePriority.MEDIUM,
                      load_func: Optional[Callable] = None,
                      unload_func: Optional[Callable] = None,
                      memory_usage_mb: int = 0,
                      vram_usage_mb: int = 0,
                      offload_func: Optional[Callable] = None,
                      instance: Any = None,
                      **_kwargs):
        """注册模型资源"""
        if load_func is None:
            def _noop_load():
                return None
            load_func = _noop_load
        if unload_func is None:
            def _noop_unload():
                return None
            unload_func = _noop_unload
        model = ModelResource(
            model_id=model_id,
            model_type=model_type,
            priority=priority,
            load_func=load_func,
            unload_func=unload_func,
            offload_func=offload_func,
            memory_usage_mb=memory_usage_mb,
            vram_usage_mb=vram_usage_mb,
            instance=instance
        )
        self._models[model_id] = model
        logger.info(f"已注册模型资源: {model_id} (优先级: {priority.name})")

    async def update_model_state(self, model_id: str, state: ResourceState, **kwargs) -> None:
        model = self._models.get(model_id)
        if model is None:
            self.register_model(model_id=model_id, model_type="unknown", priority=ResourcePriority.MEDIUM)
            model = self._models.get(model_id)
            if model is None:
                return

        device = kwargs.get("device", None)
        if device is not None:
            try:
                model.device = "CPU" if "cpu" in str(device).lower() else "GPU"
            except Exception:
                pass

        if "memory_usage_mb" in kwargs:
            try:
                model.memory_usage_mb = int(kwargs.get("memory_usage_mb") or 0)
            except Exception:
                pass
        if "vram_usage_mb" in kwargs:
            try:
                model.vram_usage_mb = int(kwargs.get("vram_usage_mb") or 0)
            except Exception:
                pass
        if "instance" in kwargs:
            model.instance = kwargs.get("instance")

        if state == ResourceState.LOADING:
            model.is_loaded = False
        elif state == ResourceState.READY:
            model.is_loaded = True
            model.is_offloaded = bool(str(model.device).upper() == "CPU" or model.vram_usage_mb == 0)
        elif state == ResourceState.OFFLOADING:
            model.is_loaded = True
        elif state == ResourceState.OFFLOADED:
            model.is_loaded = True
            model.is_offloaded = True
            model.device = "CPU"
        elif state == ResourceState.UNLOADED:
            model.is_loaded = False
            model.is_offloaded = False
            model.device = "CPU"
            model.vram_usage_mb = 0
            model.instance = None
        elif state == ResourceState.ERROR:
            model.is_loaded = False

        try:
            model.update_usage()
        except Exception:
            pass

    def register_resource_handler(self, handler_type: str, priority: ResourcePriority, handler_func: Callable):
        """注册通用资源处理回调 (如清理缓存)"""
        self._resource_handlers.append({
            "type": handler_type,
            "priority": priority,
            "handler": handler_func
        })

    async def request_resource(self, model_id: str, priority: Optional[ResourcePriority] = None) -> bool:
        """
        请求加载模型资源
        如果资源不足，会尝试卸载低优先级模型
        """
        model = self._models.get(model_id)
        if not model:
            logger.error(f"模型未注册: {model_id}")
            return False
            
        # 如果模型已加载且在GPU上，直接返回
        if model.is_loaded and not model.is_offloaded and model.device == "GPU":
            model.update_usage()
            return True
            
        async with self._lock:
            # 检查系统资源状态
            memory_state = self.monitor.get_resource_state(ResourceType.MEMORY)
            
            # 如果是紧急状态，先清理
            if memory_state == ResourceState.EMERGENCY:
                await self._emergency_cleanup()
            elif memory_state == ResourceState.CRITICAL:
                await self._critical_cleanup()
                
            # 检查显存是否足够
            if torch and torch.cuda.is_available():
                gpu_usage = await self.monitor.get_gpu_memory_usage_async()
                if gpu_usage:
                    used, total = gpu_usage
                    # 预留 buffer
                    required_mb = self.TYPICAL_VRAM_USAGE.get(model_id, 2000)
                    available_mb = total - used
                    
                    if available_mb < required_mb:
                        logger.info(f"显存不足 (可用: {available_mb}MB, 需要: {required_mb}MB)，尝试清理资源...")
                        # 尝试卸载低优先级模型
                        await self._free_up_vram(required_mb, model.priority)
            
            # 尝试加载/回迁
            try:
                # 如果是Offload状态，尝试回迁
                if model.is_offloaded:
                    gpu_free_mb = await self._get_gpu_free_mb_async()
                    success = await self._try_restore_model_to_gpu(model_id, gpu_free_mb)
                    if success:
                        model.update_usage()
                        return True

                # 否则执行加载
                logger.info(f"开始加载模型: {model_id}")
                if asyncio.iscoroutinefunction(model.load_func):
                    await model.load_func()
                else:
                    model.load_func()
                
                self.mark_model_loaded(model_id, True)
                return True
            except Exception as e:
                logger.error(f"加载模型 {model_id} 失败: {str(e)}")
                return False

    async def _free_up_vram(self, required_mb: int, min_priority: ResourcePriority):
        """尝试释放指定大小的显存"""
        # 1. 先清理缓存
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 2. 检查是否够了
        gpu_free = await self._get_gpu_free_mb_async() or 0
        if gpu_free >= required_mb:
            return

        # 3. 按优先级卸载/Offload
        # 排序：优先级低 -> 高
        sorted_models = sorted(
            [m for m in self._models.values() if m.is_loaded and not m.is_offloaded and m.device == "GPU"],
            key=lambda x: x.priority.value,
            reverse=True # value越大优先级越低 (LOW=3)
        )
        
        for model in sorted_models:
            # 不动比当前请求优先级高或相等的模型
            if model.priority.value <= min_priority.value:
                continue
            
            # 尝试 Offload
            await self._try_offload_model_to_cpu(model.model_id, reason="resource_request")
            
            gpu_free = await self._get_gpu_free_mb_async() or 0
            if gpu_free >= required_mb:
                return

    async def prepare_for_heavy_task(self, task_type: str, urgent: bool = False):
        """
        为重负载任务预留资源
        
        Args:
            task_type: 'image_gen', 'training'
            urgent: 是否使用紧急模式（快速让位）
        """
        logger.info(f"准备重负载任务资源: {task_type} (紧急模式: {urgent})")
        if task_type == 'image_gen':
            # 必须卸载/Offload LLM
            await self._try_offload_model_to_cpu("llm_engine", reason="image_gen", urgent=urgent)
            # 卸载 TTS/STT
            await self._offload_voice_services()
            # 清理缓存
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def unload_model(self, model_id: str):
        """卸载指定模型"""
        model = self._models.get(model_id)
        if not model:
            return
            
        if model.is_loaded:
            try:
                if asyncio.iscoroutinefunction(model.unload_func):
                    await model.unload_func()
                else:
                    model.unload_func()
                self.mark_model_loaded(model_id, False)
                logger.info(f"模型已卸载: {model_id}")
            except Exception as e:
                logger.error(f"卸载模型 {model_id} 失败: {str(e)}")

    async def unload_all(self):
        """卸载所有模型"""
        tasks = []
        for model_id in self._models:
            tasks.append(self.unload_model(model_id))
        if tasks:
            await asyncio.gather(*tasks)

    def get_model(self, model_id: str) -> Optional[ModelResource]:
        return self._models.get(model_id)

    def _determine_model_device(self, model: ModelResource) -> str:
        try:
            # 1. 检查 is_offloaded 标志
            if model.is_offloaded:
                return "CPU"
                
            # 2. 检查 instance provider
            provider = getattr(model.instance, "provider", None)
            if provider and str(provider).strip().lower() != "local":
                return "CPU"
                
            # 3. 针对不同模型类型的特定逻辑
            if model.model_id == "llm_engine":
                cfg = getattr(model.instance, "config", None) or getattr(model.instance, "_gpu_config", None)
                if isinstance(cfg, dict):
                    if int(cfg.get("n_gpu_layers", -1)) == 0:
                        return "CPU"
                return "GPU"
                
            if model.model_id == "vision_module":
                device = getattr(model.instance, "device", None)
                if device and "cpu" in str(device).lower():
                    return "CPU"
                return "GPU"
                
            # 4. 通用 Torch 设备检查
            if torch and torch.cuda.is_available():
                device = getattr(model.instance, "device", None)
                if device and "cuda" in str(device).lower():
                    return "GPU"
            
            return "CPU" if model.vram_usage_mb == 0 else "GPU"
        except Exception as e:
            logger.debug(f"Error determining device for {model.model_id}: {e}")
            return "GPU" if model.vram_usage_mb > 0 else "CPU"

    async def _offload_voice_services(self):
        """Proactively offload voice services (TTS/STT) to CPU"""
        logger.info("ResourceManager: Offloading voice services to CPU...")
        voice_models = ["tts_engine", "stt_engine", "svc_engine"]
        
        for mid in voice_models:
            model = self.get_model(mid)
            if model and model.is_loaded:
                if model.offload_func:
                    try:
                        if asyncio.iscoroutinefunction(model.offload_func):
                            await model.offload_func("release")
                        else:
                            await asyncio.to_thread(model.offload_func, "release")
                    except Exception as e:
                        logger.warning(f"Failed to offload {mid} via offload_func: {e}")
                
                elif model.instance:
                    if hasattr(model.instance, "move_to_cpu"):
                        try:
                            if asyncio.iscoroutinefunction(model.instance.move_to_cpu):
                                await model.instance.move_to_cpu()
                            else:
                                await asyncio.to_thread(model.instance.move_to_cpu)
                            model.device = "CPU"
                            model.vram_usage_mb = 0
                        except Exception as e:
                            logger.warning(f"Failed to move {mid} to CPU: {e}")

    def mark_model_loaded(self, model_id: str, loaded: bool):
        """标记模型加载状态并更新设备信息"""
        model = self._models.get(model_id)
        if not model:
            return
            
        model.is_loaded = bool(loaded)
        if loaded:
            model.update_usage()
            if model.vram_usage_mb == 0:
                model.vram_usage_mb = self.TYPICAL_VRAM_USAGE.get(model_id, 0)
            model.device = self._determine_model_device(model)
        else:
            model.device = "CPU"
            model.is_offloaded = False
            model.vram_usage_mb = 0
            
        # 触发异步更新指标 (Legacy 简化: 不依赖 EventBus)
        if self._is_running:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._update_model_resource_metrics())
            except Exception:
                pass
    
    def unregister_model(self, model_id: str):
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"模型已注销: {model_id}")
    
    def register_memory_cleanup_callback(self, callback: Callable):
        self._memory_cleanup_callbacks.append(callback)
    
    async def _monitor_resources(self):
        """资源监控任务"""
        try:
            while self._is_running:
                await asyncio.sleep(self.config['monitor_interval'])
                
                memory_state = self.monitor.get_resource_state(ResourceType.MEMORY)
                gpu_usage = await self.monitor.get_gpu_memory_usage_async()
                gpu_memory_state = ResourceState.NORMAL
                if gpu_usage:
                    used, total = gpu_usage
                    percent = (used / total) * 100 if total > 0 else 0
                    threshold = self.monitor._thresholds[ResourceType.GPU_MEMORY]
                    if percent >= threshold.emergency:
                        gpu_memory_state = ResourceState.EMERGENCY
                    elif percent >= threshold.critical:
                        gpu_memory_state = ResourceState.CRITICAL
                    elif percent >= threshold.warning:
                        gpu_memory_state = ResourceState.WARNING
                
                # 记录资源使用情况 (Debug)
                # logger.debug(f"Memory: {self.monitor.get_memory_usage():.1f}%")
                
                now_ts = time.time()
                if memory_state == ResourceState.EMERGENCY or gpu_memory_state == ResourceState.EMERGENCY:
                    last_ts = self._last_cleanup_ts.get("emergency", 0.0)
                    cooldown = float(self._cleanup_cooldown_seconds.get("emergency", 12.0))
                    if (now_ts - last_ts) >= cooldown:
                        self._last_cleanup_ts["emergency"] = now_ts
                        await self._emergency_cleanup()
                elif memory_state == ResourceState.CRITICAL or gpu_memory_state == ResourceState.CRITICAL:
                    last_ts = self._last_cleanup_ts.get("critical", 0.0)
                    cooldown = float(self._cleanup_cooldown_seconds.get("critical", 6.0))
                    if (now_ts - last_ts) >= cooldown:
                        self._last_cleanup_ts["critical"] = now_ts
                        await self._critical_cleanup()
                elif memory_state == ResourceState.WARNING or gpu_memory_state == ResourceState.WARNING:
                    last_ts = self._last_cleanup_ts.get("regular", 0.0)
                    cooldown = float(self._cleanup_cooldown_seconds.get("regular", 3.0))
                    if (now_ts - last_ts) >= cooldown:
                        self._last_cleanup_ts["regular"] = now_ts
                        await self._regular_cleanup()
                
                await self._update_model_resource_metrics(gpu_usage)
                await self._cleanup_unused_models()
                await self._cleanup_cache()

                try:
                    await self._auto_recover_gpu_models()
                except Exception:
                    pass
                
        except Exception as e:
            logger.error(f"资源监控任务异常: {str(e)}")
    
    async def _release_low_priority_resources(self, max_priority: ResourcePriority):
        for handler_info in self._resource_handlers:
             if handler_info["priority"].value < max_priority.value:
                 try:
                     func = handler_info["handler"]
                     if asyncio.iscoroutinefunction(func):
                         await func("release")
                     else:
                         func("release")
                 except Exception as e:
                     logger.error(f"Error releasing resource: {e}")

    async def _emergency_cleanup(self):
        logger.warning("进入紧急清理模式")
        try:
            await asyncio.wait_for(self._offload_voice_services(), timeout=3.0)
        except Exception:
            pass
        await self._release_low_priority_resources(ResourcePriority.HIGH)
        await self._unload_models_by_priority(ResourcePriority.HIGH)
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def _critical_cleanup(self):
        logger.warning("进入临界清理模式")
        try:
            await asyncio.wait_for(self._offload_voice_services(), timeout=3.0)
        except Exception:
            pass
        await self._release_low_priority_resources(ResourcePriority.HIGH)
        await self._unload_models_by_priority(ResourcePriority.MEDIUM)
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def _regular_cleanup(self):
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def _unload_models_by_priority(self, min_priority: ResourcePriority):
        async with self._lock:
            models_to_unload = sorted(
                [(model_id, model) for model_id, model in self._models.items() 
                 if model.is_loaded and model.priority.value < min_priority.value],
                key=lambda x: (x[1].priority.value, x[1].last_used_time)
            )
            
            for model_id, model in models_to_unload:
                try:
                    logger.info(f"卸载模型: {model_id} (优先级: {model.priority.name})")
                    if asyncio.iscoroutinefunction(model.unload_func):
                        await model.unload_func()
                    else:
                        model.unload_func()
                    model.is_loaded = False
                except Exception as e:
                    logger.error(f"卸载模型 {model_id} 失败: {str(e)}")
    
    async def _cleanup_unused_models(self):
        current_time = time.time()
        timeout = self.config['model_unload_timeout']
        
        async with self._lock:
            for model_id, model in self._models.items():
                if model.priority == ResourcePriority.HIGH:
                    continue
                if model.is_loaded and (current_time - model.last_used_time) > timeout:
                    try:
                        logger.info(f"卸载超时模型: {model_id}")
                        if asyncio.iscoroutinefunction(model.unload_func):
                            await model.unload_func()
                        else:
                            model.unload_func()
                        model.is_loaded = False
                    except Exception as e:
                        logger.error(f"卸载模型 {model_id} 失败: {str(e)}")
    
    async def _cleanup_cache(self):
        if self._cache_stats['size'] > self.config['cache_size_limit_mb']:
            target_size = int(self.config['cache_size_limit_mb'] * 0.8)
            self._cache_stats['size'] = target_size
    
    async def _update_model_resource_metrics(self, gpu_info: Optional[tuple[int, int]] = None):
        # Simplified for Legacy: just update vram_usage if loaded
        async with self._lock:
            if gpu_info is None:
                gpu_info = await self.monitor.get_gpu_memory_usage_async()
            
            # Logic simplified from core - we trust self-reported or typical usage mostly for legacy
            # unless we implement full process tracking
            pass 

    async def _get_gpu_free_mb_async(self) -> Optional[int]:
        gpu_info = await self.monitor.get_gpu_memory_usage_async()
        if not gpu_info:
            return None
        used, total = gpu_info
        return max(0, int(total) - int(used))

    def _get_vram_reserve_mb(self) -> int:
        try:
            settings = get_settings()
            model_settings = getattr(settings, "model", None)
            if model_settings is not None:
                return int(getattr(model_settings, "vram_reserve_mb", 0) or 0)
        except Exception:
            return 0
        return 0
    
    async def _call_maybe_async(self, func: Callable, *args, timeout: float = 10.0, **kwargs) -> bool:
        try:
            if asyncio.iscoroutinefunction(func):
                await asyncio.wait_for(asyncio.shield(func(*args, **kwargs)), timeout=timeout)
            else:
                await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=timeout)
            return True
        except Exception:
            return False

    async def _call_maybe_async_detail(self, func: Callable, *args, timeout: float = 10.0, **kwargs) -> tuple[bool, Optional[str], bool]:
        try:
            if asyncio.iscoroutinefunction(func):
                await asyncio.wait_for(asyncio.shield(func(*args, **kwargs)), timeout=timeout)
            else:
                await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=timeout)
            return True, None, False
        except asyncio.TimeoutError:
            return False, "timeout", True
        except Exception as e:
            return False, str(e), False

    async def _try_offload_model_to_cpu(self, model_id: str, reason: str = "", urgent: bool = False):
        """
        将模型迁移到 CPU
        
        Args:
            model_id: 模型ID
            reason: 迁移原因
            urgent: 是否使用紧急模式（快速让位）
        """
        model = self._models.get(model_id)
        if not model or not bool(model.is_loaded):
            return

        if model.is_offloaded or str(model.device).upper() == "CPU":
            return

        # Legacy logic adaptation
        if model_id == "llm_engine":
            instance = model.instance
            # Try atomic offload if available
            quick_release = getattr(instance, "release_llm_vram_for_image_gen", None)
            offload_to_cpu = getattr(instance, "offload_llm_to_cpu", None)
            
            # 优先使用 offload_llm_to_cpu（支持原子热替换）
            if callable(offload_to_cpu):
                try:
                    await self._call_maybe_async(offload_to_cpu, urgent, timeout=30.0)
                    model.device = "CPU"
                    model.vram_usage_mb = 0
                    model.is_offloaded = True
                    logger.info(f"LLM 已迁移到 CPU (模式: {'紧急' if urgent else '原子'})")
                    return
                except Exception as e:
                    logger.warning(f"offload_llm_to_cpu 失败: {e}")
            
            # Fallback: release_llm_vram_for_image_gen（紧急模式）
            if callable(quick_release) and str(reason or "").lower() == "image_gen":
                try:
                    await self._call_maybe_async(quick_release, timeout=8.0)
                    model.device = "CPU"
                    model.vram_usage_mb = 0
                    model.is_offloaded = True
                    return
                except Exception:
                    pass
            
            # Fallback to offload_func
            if model.offload_func:
                if await self._call_maybe_async(model.offload_func, timeout=10.0):
                    model.device = "CPU"
                    model.vram_usage_mb = 0
                    model.is_offloaded = True
                    return

        # Generic move_to_cpu
        instance = model.instance
        move_to_cpu = getattr(instance, "move_to_cpu", None)
        if callable(move_to_cpu):
            if await self._call_maybe_async(move_to_cpu, timeout=12.0):
                model.device = "CPU"
                model.vram_usage_mb = 0
                model.is_offloaded = True
                return

        # Fallback to unload
        if model.unload_func:
            await self.unload_model(model_id)

    async def _try_restore_model_to_gpu(self, model_id: str, gpu_free_mb: Optional[int]) -> bool:
        model = self._models.get(model_id)
        if not model or not bool(model.is_loaded):
            return False

        if gpu_free_mb is not None:
            need = int(self.TYPICAL_VRAM_USAGE.get(model_id, 0) or 0)
            if need > 0 and int(gpu_free_mb) < need:
                return False

        instance = model.instance
        logger.info(f"正在尝试将模型 {model_id} 回迁至 GPU...")
        
        # LLM specific restore
        if model_id == "llm_engine":
            restore = getattr(instance, "restore_llm_to_gpu", None)
            if callable(restore):
                ok, _, _ = await self._call_maybe_async_detail(restore, timeout=120.0)
                if ok:
                    model.device = "GPU"
                    model.is_offloaded = False
                    logger.info("LLM 引擎已成功回迁至 GPU")
                return bool(ok)

        # Generic move_to_gpu
        move_to_gpu = getattr(instance, "move_to_gpu", None)
        if callable(move_to_gpu):
            ok, _, _ = await self._call_maybe_async_detail(move_to_gpu, timeout=30.0)
            if ok:
                model.device = "GPU"
                model.is_offloaded = False
                self.mark_model_loaded(model_id, True)
            return bool(ok)

        # Fallback to reload
        if model.load_func:
            ok = await self._call_maybe_async(model.load_func, timeout=15.0)
            if ok:
                model.device = "GPU"
                model.is_offloaded = False
                self.mark_model_loaded(model_id, True)
            return bool(ok)

        return False

    async def _auto_recover_gpu_models(self):
        # Simplified auto recover
        gpu_usage = await self.monitor.get_gpu_memory_usage_async()
        if not gpu_usage:
            return
        
        used, total = gpu_usage
        percent = (used / total) * 100 if total > 0 else 0
        if percent >= 80: # Warning threshold
            return

        # Don't recover if image gen is loaded
        image = self._models.get("image_gen_module")
        if image and bool(image.is_loaded):
            return

        targets = ["llm_engine"]
        for model_id in targets:
            model = self._models.get(model_id)
            if model and model.is_loaded and model.is_offloaded:
                # Avoid concurrent restore
                if model_id in self._restore_tasks and not self._restore_tasks[model_id].done():
                    continue
                    
                async def _do_restore():
                    async with self._restore_lock:
                        fresh_free = await self._get_gpu_free_mb_async()
                        await self._try_restore_model_to_gpu(model_id, fresh_free)

                self._restore_tasks[model_id] = asyncio.create_task(_do_restore())

def get_resource_manager():
    return ResourceManager.get_instance()
