"""LLM 动态显存迁移 Mixin：GPU↔CPU 非对称迁移与卸载。

从 cpp_scheduler_engine.py 的 CPPSchedulerEngine 拆出。
状态（self._python_llm / _python_force_cpu / _saved_llm_state 等）仍由
CPPSchedulerEngine.__init__ 初始化，本 Mixin 只提供方法。
- 紧急模式 (urgent=True): save KV → 释放 GPU → 后台加载 CPU 实例
- 原子模式 (urgent=False): 预加载 CPU 实例 → 原子替换 → 延迟释放旧实例
"""
import asyncio
import gc
import time

from mvp_core.utils.logger import logger

try:
    from mvp_core.services.cpp_engine_utils import (
        ResourceState,
        get_runtime_metrics,
    )
except ImportError:
    from services.cpp_engine_utils import (  # type: ignore
        ResourceState,
        get_runtime_metrics,
    )


class MigrationMixin:
    """LLM GPU↔CPU 动态迁移与卸载"""

    async def release_llm_vram_for_image_gen(self):
        """
        专门用于图像生成的显存快速释放接口
        由 ResourceManager 调用
        """
        logger.info("收到图像生成显存释放请求，正在执行...")
        await self.offload_llm_to_cpu(urgent=True)

    async def offload_llm_to_cpu(self, urgent: bool = False):
        """
        将 LLM 迁移到 CPU 或卸载以释放显存 (原子热切换)

        支持两种模式：
        - 紧急模式 (urgent=True): 快速让位，先卸载再后台加载 CPU 实例
        - 原子模式 (urgent=False): 无感迁移，先加载 CPU 实例再替换，服务不中断
        """
        logger.info(f"执行 LLM 设备迁移 (GPU -> CPU), 模式: {'紧急' if urgent else '原子热切换'}...")
        _t0_offload = time.time()
        # 1. 如果是 Python 后端
        if self._llm_backend == "python":
            async with self._python_llm_lock:
                if not self._python_llm:
                    return

                # 检查是否已经在 CPU 模式
                if self._python_force_cpu:
                    logger.info("LLM 已在 CPU 模式，跳过迁移")
                    return

                # 更新资源管理器状态
                if self.resource_manager:
                    await self.resource_manager.update_model_state(
                        "llm_engine",
                        ResourceState.OFFLOADING,
                        vram_usage_mb=0
                    )

                # 保存当前 GPU 配置以便恢复
                if isinstance(self._gpu_config, dict):
                    try:
                        self._prev_n_gpu_layers = int(self._gpu_config.get("n_gpu_layers", -1) or -1)
                        self._prev_n_ctx = int(self._gpu_config.get("max_context_size") or self._gpu_config.get("n_ctx") or 4096)
                        self._prev_n_batch = int(self._gpu_config.get("max_batch_size") or self._gpu_config.get("n_batch") or 512)
                    except Exception:
                        pass

                if urgent:
                    # 紧急模式：闪电让位，优先释放显存
                    logger.info("紧急模式：立即释放 GPU 显存...")

                    old_inst = self._python_llm
                    if old_inst:
                        try:
                            logger.info("紧急模式：正在保存 KV Cache...")
                            state = await asyncio.get_running_loop().run_in_executor(
                                self._llm_executor, old_inst.save_state
                            )
                            self._saved_llm_state = state
                            self._saved_llm_state_ts = time.time()
                            logger.info("紧急模式：KV Cache 保存完成")
                        except Exception as e:
                            logger.warning(f"紧急模式：KV Cache 保存失败（非致命）: {e}")
                            self._saved_llm_state = None
                            self._saved_llm_state_ts = 0.0

                    def _do_offload():
                        if self._python_llm:
                            try:
                                if hasattr(self._python_llm, "close"):
                                    self._python_llm.close()
                            except Exception:
                                pass
                            del self._python_llm
                            self._python_llm = None
                        gc.collect()

                    await asyncio.get_running_loop().run_in_executor(self._llm_executor, _do_offload)

                    # 强制清理显存
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    except ImportError:
                        pass

                    self._python_force_cpu = True
                    self._python_llm_device = "cpu"

                    # 后台异步加载 CPU 实例
                    async def background_cpu_load():
                        try:
                            async with self._python_llm_lock:
                                if self._python_force_cpu and self._python_llm is None:
                                    cpu_config = dict(self._gpu_config) if isinstance(self._gpu_config, dict) else {}
                                    cpu_config["force_cpu"] = True
                                    cpu_config["n_gpu_layers"] = 0
                                    await asyncio.get_running_loop().run_in_executor(
                                        self._llm_executor, self._setup_python_llm, cpu_config
                                    )
                                    inst = self._python_llm
                                    state = self._saved_llm_state
                                    if inst is not None and state is not None:
                                        try:
                                            logger.info("紧急模式：正在恢复 KV Cache 到 CPU 实例...")
                                            await asyncio.get_running_loop().run_in_executor(
                                                self._llm_executor, inst.load_state, state
                                            )
                                            self._saved_llm_state = None
                                            self._saved_llm_state_ts = 0.0
                                            logger.info("紧急模式：KV Cache 恢复完成")
                                        except Exception as e:
                                            logger.warning(f"紧急模式：KV Cache 恢复失败（非致命）: {e}")
                                    logger.info("LLM 后台迁移到 CPU 完成")
                        except Exception as e:
                            logger.error(f"LLM 后台迁移到 CPU 失败: {e}")

                    asyncio.create_task(background_cpu_load())

                else:
                    # 原子热替换模式：先加载 CPU 实例，再替换
                    logger.info("原子模式：准备 CPU 实例...")
                    old_inst = self._python_llm

                    # 1. 准备 CPU 配置
                    cpu_config = dict(self._gpu_config) if isinstance(self._gpu_config, dict) else {}
                    cpu_config["force_cpu"] = True
                    cpu_config["n_gpu_layers"] = 0

                    # 2. 加载新实例 (CPU)
                    new_inst = await asyncio.get_running_loop().run_in_executor(
                        self._llm_executor,
                        lambda: self._setup_python_llm(cpu_config, return_instance=True)
                    )

                    if new_inst:
                        # 3. 迁移 KV Cache（关键：保持对话上下文）
                        if old_inst:
                            try:
                                logger.info("正在迁移 KV Cache...")
                                state = await asyncio.get_running_loop().run_in_executor(
                                    self._llm_executor, old_inst.save_state
                                )
                                await asyncio.get_running_loop().run_in_executor(
                                    self._llm_executor, new_inst.load_state, state
                                )
                                logger.info("KV Cache 迁移完成")
                            except Exception as e:
                                logger.warning(f"KV Cache 迁移失败（非致命）: {e}")

                        # 4. 原子替换
                        self._python_llm = new_inst
                        self._python_force_cpu = True
                        self._python_llm_device = getattr(new_inst, "_xy_effective_device", "cpu")
                        logger.info("LLM 已原子切换到 CPU 实例，服务保持")

                        # 5. 后台释放旧 GPU 实例
                        if old_inst:
                            async def delayed_unload(inst=old_inst):
                                try:
                                    logger.info("正在清理旧的 GPU 实例...")
                                    def _cleanup():
                                        try:
                                            if hasattr(inst, "close"):
                                                inst.close()
                                        except Exception:
                                            pass
                                        gc.collect()
                                        try:
                                            import torch
                                            if torch.cuda.is_available():
                                                torch.cuda.empty_cache()
                                                torch.cuda.ipc_collect()
                                        except Exception:
                                            pass

                                    await asyncio.get_running_loop().run_in_executor(self._llm_executor, _cleanup)
                                    logger.info("GPU 显存已完全释放")
                                except Exception as e:
                                    logger.error(f"清理 GPU 实例失败: {e}")

                            asyncio.create_task(delayed_unload())
                    else:
                        logger.error("加载 CPU 实例失败，原子切换中止")

                # 更新资源管理器状态
                if self.resource_manager:
                    await self.resource_manager.update_model_state(
                        "llm_engine",
                        ResourceState.OFFLOADED,
                        device="cpu",
                        vram_usage_mb=0
                    )
                logger.info("LLM 迁移到 CPU 完成")
            _metrics = get_runtime_metrics()
            if _metrics is not None:
                _metrics.update_offload(time.time() - _t0_offload)
            return

        # 2. 如果是 C++ 原生 Worker 后端
        if self._llm_backend == "cpp":
            logger.warning("C++ 原生 Worker (GPULLMWorker) 暂不支持动态显存迁移，请切换到 Python 后端以使用此功能")
            _metrics = get_runtime_metrics()
            if _metrics is not None:
                _metrics.update_offload(time.time() - _t0_offload)
            return

    async def restore_llm_to_gpu(self) -> bool:
        """
        将 LLM 恢复到 GPU (原子热替换)

        流程：
        1. 加载新的 GPU 实例
        2. 迁移 KV Cache
        3. 原子替换
        4. 后台释放旧的 CPU 实例
        """
        logger.info(f"LLM 引擎尝试回迁 GPU (Backend: {self._llm_backend})...")
        _t0_restore = time.time()

        def _finish_restore(result: bool) -> bool:
            try:
                _m = get_runtime_metrics()
                if _m is not None:
                    _m.update_restore(time.time() - _t0_restore)
            except Exception:
                pass
            return result

        if self._llm_backend == "python":
            async with self._python_llm_lock:
                # 检查是否已经在 GPU 模式
                if self._python_llm and not self._python_force_cpu:
                    logger.info("LLM 已在 GPU 模式，跳过回迁")
                    return _finish_restore(True)

                # 如果没有实例，直接加载到 GPU
                if not self._python_llm:
                    if self.resource_manager:
                        await self.resource_manager.update_model_state(
                            "llm_engine",
                            ResourceState.LOADING
                        )

                    try:
                        if self._gpu_config:
                            gpu_config = dict(self._gpu_config)
                            # 恢复之前的 GPU 配置
                            if self._prev_n_gpu_layers is not None:
                                gpu_config["n_gpu_layers"] = self._prev_n_gpu_layers
                            if self._prev_n_ctx is not None:
                                gpu_config["max_context_size"] = self._prev_n_ctx
                            if self._prev_n_batch is not None:
                                gpu_config["max_batch_size"] = self._prev_n_batch
                            gpu_config["force_cpu"] = False

                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(self._llm_executor, self._setup_python_llm, gpu_config)

                        success = self._python_llm is not None
                        if success:
                            self._python_force_cpu = False
                            logger.info("LLM 已成功加载到 GPU")
                            if self.resource_manager:
                                await self.resource_manager.update_model_state(
                                    "llm_engine",
                                    ResourceState.READY,
                                    device=self._python_llm_device or "gpu"
                                )
                        return _finish_restore(success)
                    except Exception as e:
                        logger.error(f"LLM 加载到 GPU 失败: {e}")
                        return _finish_restore(False)

                # 原子热替换：从 CPU 迁移到 GPU
                logger.info("原子模式：准备 GPU 实例...")
                old_inst = self._python_llm

                # 1. 准备 GPU 配置
                gpu_config = dict(self._gpu_config) if isinstance(self._gpu_config, dict) else {}
                if self._prev_n_gpu_layers is not None:
                    gpu_config["n_gpu_layers"] = self._prev_n_gpu_layers
                if self._prev_n_ctx is not None:
                    gpu_config["max_context_size"] = self._prev_n_ctx
                if self._prev_n_batch is not None:
                    gpu_config["max_batch_size"] = self._prev_n_batch
                gpu_config["force_cpu"] = False

                # 2. 加载新实例 (GPU)
                new_inst = await asyncio.get_running_loop().run_in_executor(
                    self._llm_executor,
                    lambda: self._setup_python_llm(gpu_config, return_instance=True)
                )

                if new_inst:
                    # 3. 迁移 KV Cache
                    if old_inst:
                        try:
                            logger.info("正在迁移 KV Cache...")
                            state = await asyncio.get_running_loop().run_in_executor(
                                self._llm_executor, old_inst.save_state
                            )
                            await asyncio.get_running_loop().run_in_executor(
                                self._llm_executor, new_inst.load_state, state
                            )
                            logger.info("KV Cache 迁移完成")
                        except Exception as e:
                            logger.warning(f"KV Cache 迁移失败（非致命）: {e}")

                    # 4. 原子替换
                    self._python_llm = new_inst
                    self._python_force_cpu = False
                    self._python_llm_device = getattr(new_inst, "_xy_effective_device", None) or "gpu"
                    logger.info("LLM 已原子切换到 GPU 实例，服务保持")

                    # 5. 后台释放旧 CPU 实例
                    if old_inst:
                        async def delayed_unload_cpu(inst=old_inst):
                            try:
                                logger.info("正在清理旧的 CPU 实例...")
                                def _cleanup():
                                    try:
                                        if hasattr(inst, "close"):
                                            inst.close()
                                    except Exception:
                                        pass
                                    gc.collect()

                                await asyncio.get_running_loop().run_in_executor(self._llm_executor, _cleanup)
                                logger.info("CPU 实例已清理")
                            except Exception as e:
                                logger.error(f"清理 CPU 实例失败: {e}")

                        asyncio.create_task(delayed_unload_cpu())

                    if self.resource_manager:
                        await self.resource_manager.update_model_state(
                            "llm_engine",
                            ResourceState.READY,
                            device=self._python_llm_device or "gpu"
                        )
                    return _finish_restore(True)
                else:
                    logger.error("加载 GPU 实例失败，原子切换中止")
                    return _finish_restore(False)

        return _finish_restore(False)

    async def unload_llm(self):
        """卸载 Python 侧 LLM 以释放显存 (增强型：强制 GC + 显存回收)"""
        if self._llm_backend != "python":
            return

        async with self._python_llm_lock:
            if self._python_llm:
                logger.info("正在从专用线程安全卸载 mvp_core LLM...")

                def _do_unload():
                    nonlocal self
                    if self._python_llm:
                        # 触发 llama-cpp 内部的 close
                        try:
                            if hasattr(self._python_llm, "close"):
                                self._python_llm.close()
                        except Exception:
                            pass
                        del self._python_llm
                        self._python_llm = None
                        self._python_llm_device = None
                    gc.collect()

                await asyncio.get_running_loop().run_in_executor(self._llm_executor, _do_unload)

                # 显存强制回收
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except ImportError:
                    pass

                if self.resource_manager:
                    await self.resource_manager.update_model_state(
                        "llm_engine",
                        ResourceState.UNLOADED,
                        vram_usage_mb=0
                    )
                logger.info("mvp_core LLM 已成功卸载并清理显存")
