"""LLM 后端 Mixin：Python (llama.cpp) / C++ 原生 Worker 的加载与推理提交。

从 cpp_scheduler_engine.py 的 CPPSchedulerEngine 拆出。
- _setup_python_llm: 资源自适应加载（候选参数降级重试）
- _setup_cpp_llm_worker: C++ GPULLMWorker 装配
- submit_llm_task: 熔断感知的 LLM 提交入口（cpp 失败回退 python）
状态（self._python_llm / _gpu_config / _breaker 等）仍由 CPPSchedulerEngine 管理。
"""
import asyncio
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import psutil

try:
    from mvp_core.services.cpp_engine_utils import (
        GPULLMWorker,
        Llama,
        LLMModelConfig,
        ResourcePriority,
        ResourceState,
        scheduler_py,
        _set_attr,
        _patch_llama_cpp_internals,
        _is_oom_error,
        _is_cuda_backend_error,
        _nvidia_smi_total_used_mb,
    )
except ImportError:
    from services.cpp_engine_utils import (  # type: ignore
        GPULLMWorker,
        Llama,
        LLMModelConfig,
        ResourcePriority,
        ResourceState,
        scheduler_py,
        _set_attr,
        _patch_llama_cpp_internals,
        _is_oom_error,
        _is_cuda_backend_error,
        _nvidia_smi_total_used_mb,
    )

logger = logging.getLogger(__name__)


class LLMBackendMixin:
    """LLM 后端装配与推理提交（Python / C++ 双后端）"""

    def _setup_python_llm(self, config: Dict[str, Any], loop: Optional[asyncio.AbstractEventLoop] = None, return_instance: bool = False):
        """
        初始化 Python 侧 Llama 实例 (增强型：资源自适应 + 容错重试)

        Args:
            config: 模型配置
            loop: 事件循环（用于更新资源管理器状态）
            return_instance: 如果为 True，返回实例而不赋值给 self._python_llm
                            用于原子热替换场景

        Returns:
            如果 return_instance=True，返回 Llama 实例；否则返回 None
        """
        if self.resource_manager:
            self.resource_manager.register_model(
                "llm_engine",
                priority=ResourcePriority.CRITICAL,
                device="gpu" # Intent
            )
            if loop:
                asyncio.run_coroutine_threadsafe(
                    self.resource_manager.update_model_state("llm_engine", ResourceState.LOADING),
                    loop
                )

        if not Llama:
            logger.error("llama_cpp 未安装，无法初始化 LLM")
            return

        # 应用内部补丁，修复内存访问冲突
        _patch_llama_cpp_internals()

        model_path = config.get("model_path")
        self._last_llm_load_error = None
        self._last_llm_load_ts = time.time()

        if not model_path or not isinstance(model_path, str):
            self._last_llm_load_error = "本地模型路径为空"
            logger.error("Python侧Llama模型加载失败：model_path为空")
            self._python_llm = None
            return

        model_path = str(model_path)
        if not os.path.exists(model_path):
            self._last_llm_load_error = f"本地模型文件不存在: {model_path}"
            logger.error("Python侧Llama模型加载失败：模型文件不存在: %s", model_path)
            self._python_llm = None
            return

        # GGUF 格式检查
        if model_path.lower().endswith(".gguf"):
            try:
                with open(model_path, "rb") as f:
                    header = f.read(4)
                if header != b"GGUF":
                    self._last_llm_load_error = "模型文件不是有效的 GGUF 格式"
                    logger.error("Python侧Llama模型加载失败：GGUF文件头异常: %s", model_path)
                    self._python_llm = None
                    return
            except Exception as e:
                self._last_llm_load_error = f"读取模型文件失败: {e}"
                logger.error("Python侧Llama模型加载失败：无法读取文件: %s", e)
                self._python_llm = None
                return

        # 打印传入的配置
        logger.info("_setup_python_llm 配置: force_cpu=%s, n_gpu_layers=%s", config.get("force_cpu"), config.get("n_gpu_layers"))

        # 资源自适应逻辑
        mem_percent = 0.0
        mem_available_gb = 0.0
        try:
            mem_info = psutil.virtual_memory()
            mem_percent = float(mem_info.percent)
            mem_available_gb = mem_info.available / (1024**3)
        except Exception:
            mem_percent = 0.0
            mem_available_gb = 0.0

        mem_pressure = bool(mem_available_gb < 1.0)
        has_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
        except Exception:
            pass

        if mem_pressure:
            if has_gpu:
                logger.warning("系统内存压力较大 (可用 %.1fGB / %.1f%%)，检测到 GPU 可用，优先显存加载以减轻压力", mem_available_gb, 100 - mem_percent)
            else:
                logger.warning("系统内存压力较大 (可用 %.1fGB / %.1f%%) 且无 GPU，将使用保守参数加载模型", mem_available_gb, 100 - mem_percent)

        desired_n_ctx = int(config.get("max_context_size") or config.get("n_ctx") or 4096)
        desired_n_batch = int(config.get("max_batch_size", 512) or 512)
        desired_n_gpu_layers = int(config.get("n_gpu_layers", -1) or -1)

        if bool(config.get("force_cpu")):
            desired_n_gpu_layers = 0
            logger.info("force_cpu=True，设置 n_gpu_layers=0")

        logger.info("最终加载参数: n_ctx=%s, n_gpu_layers=%s, n_batch=%s", desired_n_ctx, desired_n_gpu_layers, desired_n_batch)

        if mem_pressure:
            if not has_gpu:
                desired_n_ctx = min(desired_n_ctx, 2048)
                desired_n_batch = min(desired_n_batch, 256)
                desired_n_gpu_layers = 0
            else:
                # 有 GPU 时，稍微限制 batch 避免爆发性显存占用
                desired_n_batch = min(desired_n_batch, 512)

        # 线程数设置
        desired_n_threads = 0
        cpu_count = os.cpu_count() or 4
        desired_n_threads = max(1, min(8, cpu_count // 2))

        # 候选参数序列 (n_ctx, n_gpu_layers, n_batch)
        candidates = []
        def _expand_gpu_layer_candidates(raw_value: int) -> List[int]:
            if raw_value == 0:
                return [0]
            if raw_value < 0:
                # 优先只尝试 -1（全部层），如果失败再尝试更少的层
                return [-1]
            return [raw_value]

        for layers in _expand_gpu_layer_candidates(desired_n_gpu_layers):
            candidates.append((desired_n_ctx, int(layers), desired_n_batch))
            if desired_n_ctx > 2048:
                candidates.append((2048, int(layers), min(desired_n_batch, 256)))

        # 最后的兜底，只有在 -1 失败时才会到这里
        if desired_n_gpu_layers < 0:
            candidates.append((desired_n_ctx, 64, desired_n_batch))
            candidates.append((desired_n_ctx, 32, desired_n_batch))
            candidates.append((desired_n_ctx, 0, desired_n_batch))
        candidates.append((1024, 0, 128))

        logger.info("正在尝试加载 Llama 模型 (Python)...")
        tried = set()
        loaded_instance = None
        for n_ctx, n_gpu_layers, n_batch in candidates:
            attempt_key = (n_ctx, n_gpu_layers, n_batch)
            if attempt_key in tried:
                continue
            tried.add(attempt_key)

            try:
                gpu_used_before = None
                if has_gpu and n_gpu_layers != 0:
                    gpu_used_before = _nvidia_smi_total_used_mb()
                logger.info("尝试初始化 llama_cpp: n_ctx=%s, n_gpu_layers=%s, n_batch=%s", n_ctx, n_gpu_layers, n_batch)
                loaded_instance = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_batch=n_batch,
                    n_threads=desired_n_threads,
                    verbose=bool(config.get("verbose", False))
                )
                logger.info("Llama 模型加载成功 (n_ctx=%s, n_gpu_layers=%s)", n_ctx, n_gpu_layers)

                effective_device = "cpu"
                if n_gpu_layers != 0:
                    effective_device = "gpu"
                    gpu_used_after = _nvidia_smi_total_used_mb()
                    if gpu_used_before is not None and gpu_used_after is not None:
                        if (gpu_used_after - gpu_used_before) < 64:
                            effective_device = "cpu"
                try:
                    setattr(loaded_instance, "_xy_effective_device", effective_device)
                except Exception:
                    pass

                # 检查 GPU 层是否真的加载了
                if loaded_instance:
                    try:
                        # 尝试获取实际层数信息
                        if hasattr(loaded_instance, "model") and hasattr(loaded_instance.model, "n_layers"):
                            actual_layers = loaded_instance.model.n_layers
                            logger.info(f"实际模型层数: {actual_layers}, 加载到 GPU: {n_gpu_layers if n_gpu_layers >=0 else 'all'}")
                    except Exception:
                        pass

                if return_instance:
                    # 原子热替换模式：返回实例，不赋值
                    return loaded_instance
                else:
                    # 正常模式：赋值给 self._python_llm
                    self._python_llm = loaded_instance
                    self._python_llm_device = effective_device

                if loop and self.resource_manager:
                    asyncio.run_coroutine_threadsafe(
                        self.resource_manager.update_model_state(
                            "llm_engine",
                            ResourceState.READY,
                            device=effective_device,
                            instance=self._python_llm
                        ),
                        loop
                    )
                return
            except Exception as e:
                msg = str(e)
                logger.warning("模型加载尝试失败 (%s): %s", attempt_key, e)
                if _is_oom_error(msg) or _is_cuda_backend_error(msg):
                    continue
                # 如果是非资源类错误（如文件损坏、非法参数），记录并跳出
                self._last_llm_load_error = msg
                break

        if not return_instance:
            self._python_llm = None
            self._python_llm_device = None
        logger.error("所有模型加载尝试均已失败")
        if loop and self.resource_manager:
            asyncio.run_coroutine_threadsafe(
                self.resource_manager.update_model_state(
                    "llm_engine",
                    ResourceState.ERROR,
                    error=self._last_llm_load_error or "All attempts failed"
                ),
                loop
            )
        return None

    def _setup_cpp_llm_worker(self, config: Dict[str, Any]):
        if not self.scheduler or not scheduler_py:
            return
        if GPULLMWorker is None or LLMModelConfig is None:
            return
        if self._cpp_worker_ready:
            return

        model_path = str(config.get("model_path") or "")
        if not model_path:
            return

        worker = GPULLMWorker("gpu-worker-0")
        cfg = LLMModelConfig()
        cfg.modelPath = model_path
        cfg.modelType = str(config.get("model_type") or "llama")
        _set_attr(cfg, ["gpu_device_id", "gpuDeviceId"], int(config.get("gpu_device_id", 0) or 0))
        _set_attr(cfg, ["n_gpu_layers", "nGpuLayers"], int(config.get("n_gpu_layers", -1) or -1))
        _set_attr(cfg, ["max_context_size", "maxContextSize"], int(config.get("max_context_size", 4096) or 4096))
        cfg.maxBatchSize = int(config.get("max_batch_size", 256) or 256)
        cfg.temperature = float(config.get("temperature", 0.7) or 0.7)
        cfg.topK = int(config.get("top_k", 40) or 40)
        cfg.topP = float(config.get("top_p", 0.95) or 0.95)
        cfg.repetitionPenalty = float(config.get("repetition_penalty", 1.1) or 1.1)
        worker.setModelConfig(cfg)
        worker.initialize()
        self.scheduler.addWorker(worker)
        self._cpp_worker_ready = True

    async def submit_llm_task(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not (self.enabled and self.scheduler):
            raise RuntimeError("C++ Scheduler not available")

        backend = (self._llm_backend or "python").strip().lower()
        if backend == "cpp":
            if self._breaker_is_open("llm"):
                async for t in self._submit_llm_task_python(prompt, **kwargs):
                    yield t
                return

            yielded_any = False
            try:
                async for t in self._submit_llm_task_cpp(prompt, **kwargs):
                    yielded_any = True
                    yield t
                self._breaker_on_success("llm")
                return
            except Exception:
                self._breaker_on_failure("llm")
                if yielded_any:
                    raise
                async for t in self._submit_llm_task_python(prompt, **kwargs):
                    yield t
                return

        async for t in self._submit_llm_task_python(prompt, **kwargs):
            yield t

    async def _submit_llm_task_python(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not Llama:
            raise RuntimeError("llama_cpp 未安装")

        first_token_timeout = float(kwargs.get("first_token_timeout", 30.0) or 30.0)
        max_tokens = int(kwargs.get("max_tokens", 512) or 512)
        temperature = float(kwargs.get("temperature", 0.7) or 0.7)

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                async with self._python_llm_lock:
                    if self._python_llm is None:
                        if not isinstance(self._gpu_config, dict):
                            raise RuntimeError("C++ 引擎 Python 后端缺失配置")

                        load_config = dict(self._gpu_config)
                        if self._python_force_cpu:
                            load_config["force_cpu"] = True
                            load_config["n_gpu_layers"] = 0
                        else:
                            load_config["force_cpu"] = False

                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            self._llm_executor, self._setup_python_llm, load_config, loop
                        )
                        if self._python_llm is None:
                            error_msg = self._last_llm_load_error or "模型加载失败"
                            raise RuntimeError(f"Python LLM 加载失败: {error_msg}")

                    llm = self._python_llm

                start = time.time()
                # 在专用线程中创建补全流
                try:
                    stream = await asyncio.get_running_loop().run_in_executor(
                        self._llm_executor,
                        lambda: llm.create_chat_completion(
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )
                    )
                except Exception as e:
                    if _is_oom_error(str(e)) and attempt < max_retries:
                        logger.warning("LLM 推理任务触发 OOM，正在卸载并重试...")
                        await self.unload_llm()
                        continue
                    logger.error("创建 LLM 补全流失败: %s", e)
                    raise RuntimeError(f"生成失败: {e}")

                yielded = False

                # 迭代生成器
                def _get_next_chunk(it):
                    try:
                        return next(it)
                    except StopIteration:
                        return None
                    except Exception as e:
                        return e

                need_retry = False
                it = iter(stream)
                while True:
                    # 在专用线程中获取下一块
                    chunk = await asyncio.get_running_loop().run_in_executor(
                        self._llm_executor, _get_next_chunk, it
                    )

                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        if _is_oom_error(str(chunk)) and (not yielded) and attempt < max_retries:
                            logger.warning("LLM 推理流迭代触发 OOM，正在卸载并重试...")
                            await self.unload_llm()
                            need_retry = True
                            break
                        logger.error("流式推理过程中出错: %s", chunk)
                        raise chunk

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yielded = True
                        yield content
                        # 稍微让出控制权
                        await asyncio.sleep(0)

                    if (not yielded) and (time.time() - start) > first_token_timeout:
                        raise TimeoutError("首字响应超时，请检查模型状态或资源占用")
                if need_retry:
                    continue
                return

            except Exception as e:
                if _is_oom_error(str(e)) and attempt < max_retries:
                    logger.warning("LLM 任务执行中发生 OOM (%s)，正在卸载并重试...", e)
                    await self.unload_llm()
                    continue
                raise

    async def _submit_llm_task_cpp(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not scheduler_py:
            raise RuntimeError("scheduler_py not available")
        if not self.scheduler:
            raise RuntimeError("C++ scheduler not started")

        async with self._cpp_llm_lock:
            if not self._cpp_worker_ready:
                if not isinstance(self._gpu_config, dict):
                    raise RuntimeError("C++ engine cpp backend missing gpu_config")
                await asyncio.to_thread(self._setup_cpp_llm_worker, self._gpu_config)
                if not self._cpp_worker_ready:
                    raise RuntimeError("C++ LLM worker not ready")

        req = scheduler_py.LLMInferenceRequest()
        req.prompt = prompt
        conv_id = (
            kwargs.get("conversation_id")
            or kwargs.get("conversationId")
            or kwargs.get("request_id")
            or kwargs.get("trace_id")
        )
        req.conversationId = str(conv_id) if conv_id else uuid.uuid4().hex
        req.maxTokens = int(kwargs.get("max_tokens", 512) or 512)
        req.temperature = float(kwargs.get("temperature", 0.7) or 0.7)
        req.topK = int(kwargs.get("top_k", 40) or 40)
        req.topP = float(kwargs.get("top_p", 0.95) or 0.95)
        req.repetitionPenalty = float(kwargs.get("repetition_penalty", 1.1) or 1.1)
        stream = kwargs.get("stream")
        if stream is None:
            stream = kwargs.get("stream_output")
        stream = True if stream is None else bool(stream)

        req.streamOutput = stream

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue = asyncio.Queue()

        if stream:
            def on_token(t: str):
                try:
                    loop.call_soon_threadsafe(token_queue.put_nowait, t)
                except Exception:
                    pass

            req.onTokenGenerated = on_token

        task = scheduler_py.LLMTask(req)
        self.scheduler.submitTask(task)

        first_token_timeout = float(kwargs.get("first_token_timeout", 10.0) or 10.0)
        deadline = time.time() + float(kwargs.get("total_timeout", 180.0) or 180.0)

        if not stream:
            while True:
                if time.time() > deadline:
                    raise TimeoutError("LLM task timeout")

                status = None
                try:
                    status = task.getStatus()
                except Exception:
                    status = None

                if status == scheduler_py.TaskStatus.COMPLETED:
                    resp = task.getResponse()
                    if (not getattr(resp, "success", True)) and getattr(resp, "errorMessage", None):
                        raise RuntimeError(resp.errorMessage)
                    text = getattr(resp, "generatedText", "") or ""
                    self._last_llm_stats = {
                        "backend": "cpp",
                        "generated_tokens": int(getattr(resp, "generatedTokens", 0) or 0),
                        "inference_time_s": float(getattr(resp, "inferenceTime", 0.0) or 0.0),
                    }
                    if text:
                        yield text
                    return
                if status == scheduler_py.TaskStatus.FAILED:
                    try:
                        resp = task.getResponse()
                        raise RuntimeError(resp.errorMessage or "LLMTask failed")
                    except Exception as e:
                        raise RuntimeError(str(e))

                await asyncio.sleep(0.02)

        yielded_any = False
        while True:
            if time.time() > deadline:
                raise TimeoutError("LLM task timeout")

            try:
                t = await asyncio.wait_for(token_queue.get(), timeout=0.2 if yielded_any else first_token_timeout)
                if t:
                    yielded_any = True
                    yield t
                continue
            except asyncio.TimeoutError:
                pass

            status = None
            try:
                status = task.getStatus()
            except Exception:
                status = None

            if status == scheduler_py.TaskStatus.COMPLETED:
                break
            if status == scheduler_py.TaskStatus.FAILED:
                try:
                    resp = task.getResponse()
                    raise RuntimeError(resp.errorMessage or "LLMTask failed")
                except Exception as e:
                    raise RuntimeError(str(e))
