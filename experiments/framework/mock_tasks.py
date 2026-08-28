"""
Mock 任务模块

提供更真实的模拟任务，用于无 GPU 环境下的测试
"""

import asyncio
import time
from typing import Any, Optional
from abc import ABC, abstractmethod


class BaseMockTask(ABC):
    """Mock 任务基类"""
    
    @abstractmethod
    async def execute(self, task_id: int, duration: float) -> Any:
        """执行任务"""
        pass


class MockCPUTask(BaseMockTask):
    """CPU 密集型 Mock 任务"""
    
    def __init__(self, use_real_computation: bool = True):
        """
        初始化 CPU Mock 任务
        
        Args:
            use_real_computation: 是否使用真实计算（矩阵乘法）
        """
        self.use_real_computation = use_real_computation
    
    async def execute(self, task_id: int, duration: float = 0.1) -> str:
        """
        执行 CPU 密集型任务
        
        Args:
            task_id: 任务 ID
            duration: 目标持续时间
            
        Returns:
            str: 结果标识
        """
        if self.use_real_computation:
            result = await asyncio.to_thread(
                self._cpu_bound_work, duration
            )
        else:
            await asyncio.sleep(duration)
            result = 0
        
        return f"cpu_result_{task_id}_{result}"
    
    def _cpu_bound_work(self, duration: float) -> int:
        """CPU 密集型工作负载"""
        end_time = time.time() + duration
        count = 0
        while time.time() < end_time:
            count += 1
            _ = count * count
        return count


class MockGPUTask(BaseMockTask):
    """GPU 密集型 Mock 任务"""
    
    def __init__(self, simulate_memory: bool = True):
        """
        初始化 GPU Mock 任务
        
        Args:
            simulate_memory: 是否模拟 GPU 内存分配
        """
        self.simulate_memory = simulate_memory
        self._torch_available = None
    
    @property
    def torch_available(self) -> bool:
        if self._torch_available is None:
            try:
                import torch
                self._torch_available = torch.cuda.is_available()
            except ImportError:
                self._torch_available = False
        return self._torch_available
    
    async def execute(self, task_id: int, duration: float = 0.1) -> str:
        """
        执行 GPU 密集型任务
        
        Args:
            task_id: 任务 ID
            duration: 目标持续时间
            
        Returns:
            str: 结果标识
        """
        if self.torch_available and self.simulate_memory:
            result = await self._gpu_work_real(task_id, duration)
        else:
            result = await self._gpu_work_simulated(task_id, duration)
        
        return result
    
    async def _gpu_work_real(self, task_id: int, duration: float) -> str:
        """使用真实 GPU 的工作负载"""
        import torch
        
        dummy = torch.zeros(1, device='cuda')
        start = time.time()
        
        while time.time() - start < duration:
            dummy = dummy + 1
            await asyncio.sleep(0)
        
        result = int(dummy.item())
        del dummy
        torch.cuda.empty_cache()
        
        return f"gpu_result_{task_id}_{result}"
    
    async def _gpu_work_simulated(self, task_id: int, duration: float) -> str:
        """模拟 GPU 工作负载（无 GPU 时）"""
        try:
            import numpy as np
            
            def matrix_work():
                start = time.time()
                count = 0
                while time.time() - start < duration:
                    _ = np.random.rand(100, 100) @ np.random.rand(100, 100)
                    count += 1
                return count
            
            count = await asyncio.to_thread(matrix_work)
            return f"gpu_simulated_result_{task_id}_{count}"
        except ImportError:
            await asyncio.sleep(duration)
            return f"gpu_mock_result_{task_id}"


class MockIOTask(BaseMockTask):
    """I/O 密集型 Mock 任务（模拟网络请求）"""
    
    def __init__(self, latency_range: tuple = (0.05, 0.2)):
        """
        初始化 I/O Mock 任务
        
        Args:
            latency_range: 延迟范围（最小，最大）
        """
        self.latency_range = latency_range
    
    async def execute(self, task_id: int, duration: float = 0.1) -> str:
        """
        执行 I/O 密集型任务
        
        Args:
            task_id: 任务 ID
            duration: 基础持续时间
            
        Returns:
            str: 结果标识
        """
        import random
        latency = random.uniform(*self.latency_range)
        await asyncio.sleep(latency)
        return f"io_result_{task_id}"


class MockLLMAdapter:
    """Mock LLM 适配器"""
    
    def __init__(self, tokens_per_second: float = 50.0):
        """
        初始化 Mock LLM
        
        Args:
            tokens_per_second: 模拟的生成速度
        """
        self.tokens_per_second = tokens_per_second
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 64,
        **kwargs
    ) -> str:
        """生成文本"""
        token_interval = 1.0 / self.tokens_per_second
        result = []
        for i in range(min(max_tokens, 100)):
            await asyncio.sleep(token_interval)
            result.append(f"token_{i}")
        return " ".join(result)
    
    async def stream_generate(
        self, 
        prompt: str, 
        max_tokens: int = 64,
        cancel_event: Optional[asyncio.Event] = None,
        **kwargs
    ):
        """流式生成文本"""
        token_interval = 1.0 / self.tokens_per_second
        for i in range(min(max_tokens, 100)):
            if cancel_event and cancel_event.is_set():
                break
            await asyncio.sleep(token_interval)
            yield f"token_{i}"


class MockImageAdapter:
    """Mock 图像生成适配器"""
    
    def __init__(self, generation_time: float = 2.0):
        """
        初始化 Mock 图像生成器
        
        Args:
            generation_time: 生成时间
        """
        self.generation_time = generation_time
    
    async def generate_image(
        self, 
        prompt: str, 
        steps: int = 20,
        cancel_event: Optional[asyncio.Event] = None,
        **kwargs
    ) -> str:
        """生成图像"""
        step_time = self.generation_time / steps
        for i in range(steps):
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError("Image generation cancelled")
            await asyncio.sleep(step_time)
        return f"mock_image_{hash(prompt) % 10000}.png"


class MockTTSAdapter:
    """Mock TTS 适配器"""
    
    def __init__(self, synthesis_time: float = 0.2):
        """
        初始化 Mock TTS
        
        Args:
            synthesis_time: 合成时间
        """
        self.synthesis_time = synthesis_time
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """合成语音"""
        await asyncio.sleep(self.synthesis_time)
        return b"mock_audio_data"
