import logging
import os
import sys
import types

import uvicorn
from fastapi import FastAPI, WebSocket


def _ensure_mvp_core_alias() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    if "mvp_core" in sys.modules:
        return

    mvp_core = types.ModuleType("mvp_core")
    mvp_core.__path__ = [base_dir]
    mvp_core.__file__ = os.path.join(base_dir, "__init__.py")
    sys.modules["mvp_core"] = mvp_core

    for subdir in ["services", "data", "domain", "utils", "presentation", "experiments"]:
        subpath = os.path.join(base_dir, subdir)
        if os.path.isdir(subpath):
            submod = types.ModuleType(f"mvp_core.{subdir}")
            submod.__path__ = [subpath]
            submod.__file__ = os.path.join(subpath, "__init__.py")
            sys.modules[f"mvp_core.{subdir}"] = submod


_ensure_mvp_core_alias()
from shared.di import container
from domain.interfaces.base_interfaces import LLMInterface, MemoryInterface, ImageGenInterface
from data.adapters.local_llm_adapter import LocalLLMAdapter
from data.adapters.mock_llm_adapter import MockLLMAdapter
from data.adapters.gguf_llm_adapter import GGUFLLMAdapter
from data.adapters.synthetic_image_adapter import SyntheticImageAdapter
from data.repositories.memory_repository import InMemoryMemoryRepository
from domain.services.chat_service import ChatService
from presentation.websocket.handler import WebSocketHandler
from services.task_scheduler import GlobalTaskScheduler, get_global_scheduler, initialize_scheduler, shutdown_scheduler
try:
    from mvp_core.config import get_settings
except Exception:
    from config import get_settings

# Setup Logger
logging.basicConfig(level=logging.INFO)

# Bootstrap
def bootstrap():
    # Register Dependencies
    # Check if model exists
    settings = get_settings()

    scheduler = get_global_scheduler()
    container.register(GlobalTaskScheduler, scheduler)

    model_path = settings.model.text_path
    if os.path.exists(model_path):
        if str(model_path).lower().endswith(".gguf"):
            try:
                logging.info(f"Using GGUFLLMAdapter with model at {model_path}")
                container.register(
                    LLMInterface,
                    GGUFLLMAdapter(
                        model_path=model_path,
                        n_ctx=getattr(settings.model, "n_ctx", 4096),
                        n_gpu_layers=getattr(settings.model, "n_gpu_layers", -1),
                    ),
                )
            except Exception as e:
                logging.warning(f"Failed to init GGUFLLMAdapter: {e}, using MockLLMAdapter")
                container.register(LLMInterface, MockLLMAdapter())
        else:
            logging.info(f"Using LocalLLMAdapter with model at {model_path}")
            container.register(LLMInterface, LocalLLMAdapter())
    else:
        logging.warning(f"Model not found at {model_path}, using MockLLMAdapter")
        container.register(LLMInterface, MockLLMAdapter())
        
    container.register(MemoryInterface, InMemoryMemoryRepository())

    use_cpp_for_image = False
    try:
        scheduler_settings = getattr(settings, "scheduler", None)
        use_cpp_for_image = bool(getattr(scheduler_settings, "use_cpp", False)) and bool(
            getattr(scheduler_settings, "use_cpp_for_image", False)
        )
    except Exception:
        use_cpp_for_image = False

    if use_cpp_for_image:
        container.register(ImageGenInterface, SyntheticImageAdapter(use_cpp_scheduler=True))
    else:
        try:
            from data.adapters.sd_adapter import SDAdapter

            sd_conf = {
                "sd_model_path": getattr(settings.model, "sd_path", ""),
                "device": getattr(settings.model, "device", "cpu"),
                "generation": {},
            }
            container.register(ImageGenInterface, SDAdapter(sd_conf))
        except Exception as e:
            logging.warning(f"Failed to init SDAdapter: {e}, using SyntheticImageAdapter")
            container.register(ImageGenInterface, SyntheticImageAdapter())
    return None

app = FastAPI()
bootstrap()
chat_service = ChatService()
ws_handler = WebSocketHandler(chat_service)

@app.on_event("startup")
async def _startup():
    await initialize_scheduler()

@app.on_event("shutdown")
async def _shutdown():
    await shutdown_scheduler()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_handler.handle_connection(websocket)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "mvp-core-1.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
