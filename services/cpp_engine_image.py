"""图像后端 Mixin：C++ 图像 Worker 装配与图像生成提交。

从 cpp_scheduler_engine.py 的 CPPSchedulerEngine 拆出。
- _setup_cpp_img_worker: C++ GPUImageWorker 装配
- submit_image_task: Forge → C++ 图像任务提交（含 LLM 显存避让与 burst 指标）
状态（self._gpu_config / _cpp_img_worker_ready / _breaker 等）仍由 CPPSchedulerEngine 管理。
"""
import asyncio
import logging
import os
import time
import uuid
from typing import Optional

try:
    from mvp_core.services.cpp_engine_utils import (
        GPUImageWorker,
        GPUImgWorker,
        ImgEngineType,
        ImgGenerationParams,
        ImgTask,
        get_runtime_metrics,
        scheduler_py,
    )
except ImportError:
    from services.cpp_engine_utils import (  # type: ignore
        GPUImageWorker,
        GPUImgWorker,
        ImgEngineType,
        ImgGenerationParams,
        ImgTask,
        get_runtime_metrics,
        scheduler_py,
    )

logger = logging.getLogger(__name__)


class ImageBackendMixin:
    """C++ 图像后端装配与图像生成提交"""

    def _setup_cpp_img_worker(self, config: dict):
        if not self.scheduler or not scheduler_py:
            return
        if (GPUImageWorker is None and GPUImgWorker is None) or ImgEngineType is None:
            return
        if self._cpp_img_worker_ready:
            return

        engine_type_raw = str(config.get("image_engine_type") or "SD15_TURBO").strip().upper()
        engine_type = getattr(ImgEngineType, engine_type_raw, None)
        if engine_type is None:
            engine_type = ImgEngineType.SD15_TURBO

        gpu_device_id = int(config.get("gpu_device_id", 0) or 0)

        worker_cls = GPUImageWorker or GPUImgWorker
        try:
            worker = worker_cls(gpu_device_id, engine_type)
        except TypeError:
            worker = worker_cls("gpu-img-worker-0", engine_type, gpu_device_id)
        worker.initialize()
        self.scheduler.addWorker(worker)
        self._cpp_img_worker_ready = True

    async def submit_image_task(self, prompt: str, skip_auto_offload: bool = False, **kwargs) -> str:
        if not (self.enabled and self.scheduler):
            raise RuntimeError("C++ Scheduler not available")
        if self._breaker_is_open("image"):
            raise RuntimeError("C++ scheduler not available")
        if not scheduler_py:
            raise RuntimeError("scheduler_py not available")
        if ImgGenerationParams is None or ImgTask is None:
            raise RuntimeError("Image bindings not available")

        out_dir = ""
        if isinstance(self._gpu_config, dict):
            out_dir = str(self._gpu_config.get("image_output_dir") or "").strip()
        if not out_dir:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mvp_root = os.path.dirname(os.path.dirname(current_dir))
            out_dir = os.path.join(mvp_root, "output", "images")
        os.makedirs(out_dir, exist_ok=True)

        task_id = f"img_{uuid.uuid4().hex}"

        on_start = kwargs.get("_xy_on_start")
        on_finish = kwargs.get("_xy_on_finish")

        backend = str(os.getenv("XIAOYOU_MVP_IMAGE_BACKEND", "forge_then_cpp") or "forge_then_cpp").strip().lower()

        async def _maybe_generate_via_forge() -> Optional[str]:
            if backend not in ("forge", "forge_then_cpp"):
                return None
            try:
                try:
                    from mvp_core.data.adapters.forge_ui_adapter import ForgeUIAdapter
                except Exception:
                    from data.adapters.forge_ui_adapter import ForgeUIAdapter  # type: ignore

                base_url = str(os.getenv("XIAOYOU_FORGE_BASE_URL", "http://127.0.0.1:7860") or "http://127.0.0.1:7860").strip()
                adapter = ForgeUIAdapter(base_url=base_url)
                try:
                    if not await adapter.is_available():
                        return None
                    width = int(kwargs.get("width", 512) or 512)
                    height = int(kwargs.get("height", 512) or 512)
                    steps = int(kwargs.get("steps", 20) or 20)
                    seed = int(kwargs.get("seed", -1) or -1)
                    cfg_scale = float(kwargs.get("guidance_scale", kwargs.get("cfg_scale", 7.0)) or 7.0)
                    negative_prompt = str(kwargs.get("negative_prompt") or "")
                    sampler_name = str(kwargs.get("sampler_name", "DPM++ 2M Karras") or "DPM++ 2M Karras")

                    result = await adapter.generate_image(
                        prompt=str(prompt or ""),
                        negative_prompt=negative_prompt,
                        steps=steps,
                        width=width,
                        height=height,
                        seed=seed,
                        sampler_name=sampler_name,
                        cfg_scale=cfg_scale,
                    )

                    if not isinstance(result, dict):
                        return None
                    if str(result.get("status") or "").lower() != "success":
                        return None
                    images = result.get("images")
                    if not isinstance(images, list) or not images:
                        return None

                    out_path = os.path.join(out_dir, f"{task_id}.png")
                    await adapter.save_image(images[0], out_path)
                    return out_path
                finally:
                    try:
                        await adapter.close()
                    except Exception:
                        pass
            except Exception:
                return None

        try:
            async with self._cpp_img_lock:
                if callable(on_start):
                    try:
                        on_start(time.perf_counter())
                    except Exception:
                        pass

                if self._python_llm and not skip_auto_offload:
                    logger.info("图像生成任务启动，尝试回收 LLM 显存以确保生成空间")
                    await self.offload_llm_to_cpu()

                _burst_t0 = time.time()
                _burst_vram_before = None
                try:
                    import torch as _torch_img
                    if _torch_img.cuda.is_available():
                        _vram_free, _vram_total = _torch_img.cuda.mem_get_info()
                        _burst_vram_before = int((_vram_total - _vram_free) / 1024 / 1024)
                        if _vram_free < 1024 * 1024 * 1024 and not skip_auto_offload:
                            logger.warning("显存紧张 (剩余 %.1f MB)，执行 LLM 显存避让", _vram_free / 1024 / 1024)
                            await self.offload_llm_to_cpu()
                except ImportError:
                    pass

                forge_path = await _maybe_generate_via_forge()
                if forge_path:
                    self._breaker_on_success("image")
                    if callable(on_finish):
                        try:
                            on_finish(time.perf_counter())
                        except Exception:
                            pass
                    return forge_path

                if not self._cpp_img_worker_ready:
                    if not isinstance(self._gpu_config, dict):
                        self._gpu_config = {}
                    await asyncio.to_thread(self._setup_cpp_img_worker, self._gpu_config)
                    if not self._cpp_img_worker_ready:
                        raise RuntimeError("C++ image worker not ready")
        except Exception:
            self._breaker_on_failure("image")
            raise

        params = ImgGenerationParams()
        params.prompt = str(prompt or "")
        negative_prompt = kwargs.get("negative_prompt")
        if negative_prompt is not None:
            params.negative_prompt = str(negative_prompt)

        params.width = int(kwargs.get("width", 512) or 512)
        params.height = int(kwargs.get("height", 512) or 512)
        params.guidance_scale = float(kwargs.get("guidance_scale", 7.5) or 7.5)
        params.steps = int(kwargs.get("steps", 20) or 20)
        params.seed = int(kwargs.get("seed", -1) or -1)

        if ImgEngineType is not None:
            model_raw = kwargs.get("model")
            if model_raw is not None:
                model_name = str(model_raw).strip().upper()
                model_enum = getattr(ImgEngineType, model_name, None)
                if model_enum is not None:
                    params.model = model_enum

        try:
            task = ImgTask(task_id, params)
        except TypeError:
            task = ImgTask(params)
        self.scheduler.submitTask(task)

        deadline = time.time() + float(kwargs.get("total_timeout", 120.0) or 120.0)
        while True:
            if time.time() > deadline:
                self._breaker_on_failure("image")
                raise TimeoutError("Image task timeout")
            status = None
            try:
                status = task.getStatus()
            except Exception:
                status = None

            if status == scheduler_py.TaskStatus.COMPLETED:
                break
            if status == scheduler_py.TaskStatus.FAILED:
                self._breaker_on_failure("image")
                raise RuntimeError("ImgTask failed")
            await asyncio.sleep(0.05)

        output_name = ""
        try:
            output_name = str(task.getOutputPath() or "")
        except Exception:
            output_name = ""

        if not output_name:
            output_name = f"{task_id}.ppm"

        out_path = output_name if os.path.isabs(output_name) else os.path.join(out_dir, output_name)

        if not os.path.isfile(out_path):
            if not out_path.lower().endswith(".ppm"):
                base, _ = os.path.splitext(out_path)
                out_path = base + ".ppm"
            width = int(kwargs.get("width", 64) or 64)
            height = int(kwargs.get("height", 64) or 64)
            width = max(1, min(512, width))
            height = max(1, min(512, height))
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            buf = bytearray()
            w_den = max(1, width - 1)
            h_den = max(1, height - 1)
            for y in range(height):
                for x in range(width):
                    r = int(x * 255 / w_den)
                    g = int(y * 255 / h_den)
                    b = 128
                    buf.extend((r, g, b))

            data = header + bytes(buf)

            def _write_file():
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(data)

            await asyncio.to_thread(_write_file)
        self._breaker_on_success("image")

        # Record burst timing for admission controller metrics
        try:
            _burst_dt = time.time() - _burst_t0
            _burst_vram_after = None
            try:
                import torch as _torch_img2
                if _torch_img2.cuda.is_available():
                    _vf, _vt = _torch_img2.cuda.mem_get_info()
                    _burst_vram_after = int((_vt - _vf) / 1024 / 1024)
            except ImportError:
                pass
            _burst_vram = None
            if _burst_vram_before is not None and _burst_vram_after is not None:
                _burst_vram = max(0, _burst_vram_after - _burst_vram_before)
            _m = get_runtime_metrics()
            if _m is not None:
                _m.update_burst(_burst_dt, vram_mb=_burst_vram)
        except Exception:
            pass

        if callable(on_finish):
            try:
                on_finish(time.perf_counter())
            except Exception:
                pass
        return out_path
