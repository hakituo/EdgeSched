import sys
import os

# Add project root to sys.path to allow importing from config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config.integrated_config import get_settings as get_integrated_settings
except ImportError:
    # Fallback if import fails (e.g. running standalone)
    print("Warning: Could not import integrated_config, using mock")
    def get_integrated_settings():
        def _get_int_env(name: str, default: int) -> int:
            raw = str(os.environ.get(name, "") or "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except Exception:
                return default

        def _get_bool_env(name: str, default: bool = False) -> bool:
            raw = str(os.environ.get(name, "") or "").strip().lower()
            if not raw:
                return bool(default)
            return raw in {"1", "true", "yes", "y", "on"}

        class MockModel:
            def __init__(self):
                self.text_path = str(
                    os.environ.get("XIAOYOU_BENCH_LLM_PATH")
                    or os.environ.get("XIAOYOU_MODEL_TEXT_PATH")
                    or ""
                )
                self.n_ctx = _get_int_env("XIAOYOU_BENCH_N_CTX", 4096)
                self.n_gpu_layers = _get_int_env("XIAOYOU_BENCH_N_GPU_LAYERS", -1)
                self.image_gen_path = str(os.environ.get("XIAOYOU_BENCH_SD_PATH") or "")
                self.vision_path = str(os.environ.get("XIAOYOU_BENCH_VL_PATH") or "")
                self.device = str(os.environ.get("XIAOYOU_MODEL_DEVICE") or "auto")
                self.llm_preload_on_startup = _get_bool_env("XIAOYOU_BENCH_PRELOAD_LLM", False)

        class MockScheduler:
            def __init__(self):
                self.use_cpp = _get_bool_env("XIAOYOU_USE_CPP_SCHEDULER", False)
                self.use_cpp_for_llm = _get_bool_env("XIAOYOU_USE_CPP_FOR_LLM", self.use_cpp)
                self.use_cpp_for_image = _get_bool_env("XIAOYOU_USE_CPP_FOR_IMAGE", False)
                self.worker_count = _get_int_env("XIAOYOU_CPP_WORKER_COUNT", 4)
                self.llm_backend = str(os.environ.get("XIAOYOU_CPP_LLM_BACKEND") or "python")
                self.image_engine_type = str(os.environ.get("XIAOYOU_CPP_IMAGE_ENGINE") or "SD15_TURBO")
                self.gpu_gate_max_waiting = _get_int_env("XIAOYOU_GPU_GATE_MAX_WAITING", 8)
                self.gpu_gate_max_concurrent = _get_int_env("XIAOYOU_GPU_GATE_MAX_CONCURRENT", 1)

        class MockIntegrated:
            def __init__(self):
                self.model = MockModel()
                self.scheduler = MockScheduler()

        return MockIntegrated()

class LegacyModelSettings:
    def __init__(self, integrated_model_settings):
        self._settings = integrated_model_settings

    @property
    def text_path(self):
        return self._settings.text_path

    @property
    def n_ctx(self):
        return self._settings.n_ctx

    @property
    def n_gpu_layers(self):
        return self._settings.n_gpu_layers

    @property
    def sd_path(self):
        return self._settings.image_gen_path

    @property
    def vl_path(self):
        return self._settings.vision_path

    @property
    def tts_api(self):
        # Assuming TTS API URL from voice settings or default
        # integrated_config doesn't seem to have a direct tts_api url field in ModelSettings, 
        # but it might be in voice settings. Legacy default was "http://127.0.0.1:9880/tts"
        return "http://127.0.0.1:9880/tts" 

    @property
    def device(self):
        return self._settings.device

    def __getattr__(self, name):
        """Delegate any unknown attributes to the underlying settings object"""
        return getattr(self._settings, name)

class LegacySchedulerSettings:
    def __init__(self, integrated_settings):
        self._scheduler = integrated_settings.scheduler
        self._model = integrated_settings.model

    @property
    def backend(self):
        return getattr(self._scheduler, "llm_backend", "python")

    @backend.setter
    def backend(self, value):
        setattr(self._scheduler, "llm_backend", value)

    @property
    def use_cpp(self):
        return getattr(self._scheduler, "use_cpp", False)

    @use_cpp.setter
    def use_cpp(self, value):
        setattr(self._scheduler, "use_cpp", bool(value))

    @property
    def use_cpp_for_llm(self):
        return getattr(self._scheduler, "use_cpp_for_llm", False)

    @use_cpp_for_llm.setter
    def use_cpp_for_llm(self, value):
        setattr(self._scheduler, "use_cpp_for_llm", bool(value))

    @property
    def use_cpp_for_image(self):
        return getattr(self._scheduler, "use_cpp_for_image", False)

    @use_cpp_for_image.setter
    def use_cpp_for_image(self, value):
        setattr(self._scheduler, "use_cpp_for_image", bool(value))

    @property
    def worker_count(self):
        return getattr(self._scheduler, "worker_count", 4)

    @worker_count.setter
    def worker_count(self, value):
        setattr(self._scheduler, "worker_count", int(value))

    @property
    def llm_backend(self):
        return getattr(self._scheduler, "llm_backend", "python")

    @llm_backend.setter
    def llm_backend(self, value):
        setattr(self._scheduler, "llm_backend", value)

    @property
    def image_engine_type(self):
        return getattr(self._scheduler, "image_engine_type", "SD15_TURBO")

    @image_engine_type.setter
    def image_engine_type(self, value):
        setattr(self._scheduler, "image_engine_type", value)

    @property
    def preload_llm(self):
        return self._model.llm_preload_on_startup

    @preload_llm.setter
    def preload_llm(self, value):
        setattr(self._model, "llm_preload_on_startup", bool(value))

    @property
    def image_queue_capacity(self):
        # integrated config has gpu_gate_max_waiting
        return getattr(self._scheduler, "gpu_gate_max_waiting", 8)

    @image_queue_capacity.setter
    def image_queue_capacity(self, value):
        setattr(self._scheduler, "gpu_gate_max_waiting", int(value))

    @property
    def image_max_concurrent_tasks(self):
        return getattr(self._scheduler, "gpu_gate_max_concurrent", 1)

    @image_max_concurrent_tasks.setter
    def image_max_concurrent_tasks(self, value):
        setattr(self._scheduler, "gpu_gate_max_concurrent", int(value))

class LegacyAppSettings:
    def __init__(self):
        self._integrated = get_integrated_settings()
        self.model = LegacyModelSettings(self._integrated.model)
        self.scheduler = LegacySchedulerSettings(self._integrated)

_SETTINGS: LegacyAppSettings | None = None

def get_settings():
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = LegacyAppSettings()
    return _SETTINGS
