import logging
import asyncio
from typing import AsyncGenerator
import gc
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from domain.interfaces.base_interfaces import LLMInterface
try:
    from mvp_core.config import get_settings
except Exception:
    from config import get_settings

logger = logging.getLogger("GGUFLLMAdapter")

class GGUFLLMAdapter(LLMInterface):
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self._lock = asyncio.Lock()
        self._cpp_start_lock = asyncio.Lock()
    
    def _get_cpp_engine(self):
        try:
            from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
            return cpp_scheduler_engine
        except Exception:
            return None

    def _should_use_cpp_scheduler(self) -> bool:
        try:
            settings = get_settings()
            scheduler_settings = getattr(settings, "scheduler", None)
            if not scheduler_settings:
                return False
            return bool(getattr(scheduler_settings, "use_cpp", False)) and bool(
                getattr(scheduler_settings, "use_cpp_for_llm", False)
            )
        except Exception:
            return False

    async def _ensure_cpp_started(self):
        cpp_engine = self._get_cpp_engine()
        if not cpp_engine or not getattr(cpp_engine, "enabled", False):
            return None
        if getattr(cpp_engine, "scheduler", None):
            return cpp_engine

        async with self._cpp_start_lock:
            if getattr(cpp_engine, "scheduler", None):
                return cpp_engine

            settings = get_settings()
            scheduler_settings = getattr(settings, "scheduler", None)
            model_settings = getattr(settings, "model", None)

            if not scheduler_settings or not model_settings:
                return cpp_engine

            backend = getattr(scheduler_settings, "llm_backend", "python")
            if backend not in ("cpp", "python"):
                backend = "python"

            gpu_config = {
                "backend": backend,
                "model_path": self.model_path,
                "model_type": "llama",
                "gpu_device_id": 0,
                "n_gpu_layers": int(getattr(model_settings, "n_gpu_layers", self.n_gpu_layers) or 0),
                "max_context_size": int(getattr(model_settings, "n_ctx", self.n_ctx) or 4096),
                "max_batch_size": 512,
            }

            worker_count = int(getattr(scheduler_settings, "worker_count", 4) or 4)
            preload_llm = bool(getattr(scheduler_settings, "preload_llm", False))

            await asyncio.to_thread(
                cpp_engine.start,
                worker_count=worker_count,
                gpu_config=gpu_config,
                preload_llm=preload_llm,
            )
            return cpp_engine

    async def _ensure_loaded(self):
        if self.model is None:
            await self._load_model()

    async def _load_model(self):
        if Llama is None:
            raise ImportError("llama_cpp is not installed. Please install llama-cpp-python.")
        logger.info(f"Loading GGUF Model from {self.model_path}")
        try:
            self.model = await asyncio.to_thread(
                Llama,
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info("GGUF Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise

    async def unload(self) -> None:
        async with self._lock:
            if self.model is None:
                return
            self.model = None

        await asyncio.to_thread(gc.collect)
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        if self._should_use_cpp_scheduler():
            try:
                cpp_engine = await self._ensure_cpp_started()
                if cpp_engine and getattr(cpp_engine, "enabled", False) and getattr(cpp_engine, "scheduler", None):
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    chunks = []
                    async for token in cpp_engine.submit_llm_task(
                        full_prompt,
                        max_tokens=kwargs.get("max_tokens", 512),
                        temperature=kwargs.get("temperature", 0.7),
                        first_token_timeout=kwargs.get("first_token_timeout", 10.0),
                        total_timeout=kwargs.get("total_timeout", 180.0),
                        stream=False,
                    ):
                        chunks.append(token)
                    return "".join(chunks)
            except Exception as e:
                logger.warning(f"C++ scheduler LLM generate failed, fallback to python: {e}")

        async with self._lock:
            await self._ensure_loaded()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await asyncio.to_thread(
                self.model.create_chat_completion,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                stream=False,
            )

            return response["choices"][0]["message"]["content"]

    async def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncGenerator[str, None]:
        if self._should_use_cpp_scheduler():
            try:
                cpp_engine = await self._ensure_cpp_started()
                if cpp_engine and getattr(cpp_engine, "enabled", False) and getattr(cpp_engine, "scheduler", None):
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    async for token in cpp_engine.submit_llm_task(
                        full_prompt,
                        max_tokens=kwargs.get("max_tokens", 512),
                        temperature=kwargs.get("temperature", 0.7),
                        first_token_timeout=kwargs.get("first_token_timeout", 10.0),
                        total_timeout=kwargs.get("total_timeout", 180.0),
                    ):
                        if token:
                            yield token
                    return
            except Exception as e:
                logger.warning(f"C++ scheduler LLM stream failed, fallback to python: {e}")

        async with self._lock:
            await self._ensure_loaded()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = await asyncio.to_thread(
                self.model.create_chat_completion,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                stream=True,
            )

            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]
                    await asyncio.sleep(0)
