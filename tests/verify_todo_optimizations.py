import asyncio
import os
import sys
import time


def _ensure_mvp_core_on_path() -> None:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    mvp_core_dir = os.path.dirname(tests_dir)
    legacy_dir = os.path.dirname(mvp_core_dir)
    repo_root = os.path.dirname(legacy_dir)

    for p in (repo_root, legacy_dir, mvp_core_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_mvp_core_on_path()


class _DummyLlm:
    def __init__(self):
        self.closed = False
        self.saved = False
        self.loaded_state = None

    def save_state(self):
        self.saved = True
        return {"ts": time.time(), "kv": "dummy"}

    def load_state(self, state):
        self.loaded_state = state

    def close(self):
        self.closed = True


async def _verify_urgent_save_state() -> None:
    from mvp_core.services.cpp_scheduler_engine import CPPSchedulerEngine

    engine = CPPSchedulerEngine()
    engine._llm_backend = "python"
    engine._gpu_config = {"model_path": "dummy.gguf"}
    engine._python_force_cpu = False
    engine._python_llm_device = "gpu"

    old = _DummyLlm()
    engine._python_llm = old

    def _setup_python_llm_stub(config, loop=None, return_instance=False):
        engine._python_llm = _DummyLlm()
        engine._python_llm_device = "cpu"
        return engine._python_llm if return_instance else None

    engine._setup_python_llm = _setup_python_llm_stub

    await engine.offload_llm_to_cpu(urgent=True)
    await asyncio.sleep(0.2)

    assert old.saved is True
    assert old.closed is True
    assert engine._python_force_cpu is True
    assert engine._python_llm is not None
    assert getattr(engine._python_llm, "loaded_state", None) is not None
    assert engine._saved_llm_state is None


async def _verify_backplane_recovery() -> None:
    from mvp_core.services.task_scheduler import GlobalTaskScheduler

    class _DummyBackplane:
        def __init__(self):
            self.updated = []

        def get_active_tasks(self):
            return [
                {"task_id": "task_running_1", "status": 2, "priority": 2, "timestamp": time.time()},
                {"task_id": "task_pending_1", "status": 1, "priority": 1, "timestamp": time.time()},
            ]

        def update_task_status(self, task_id: str, status: int):
            self.updated.append((task_id, status))
            return True

    class _DummyPythonWorker:
        def __init__(self):
            self.state_backplane = _DummyBackplane()
            self.submitted = []

        async def submit_task(self, task_info, func, args, kwargs):
            self.submitted.append((task_info.task_id, task_info.status.value))
            return asyncio.get_running_loop().create_future()

    scheduler = GlobalTaskScheduler()
    scheduler._python_worker = _DummyPythonWorker()

    await scheduler._recover_tasks_from_backplane()

    sb = scheduler._python_worker.state_backplane
    assert ("task_running_1", 4) in sb.updated
    assert any(tid == "task_pending_1" for tid, _ in scheduler._python_worker.submitted)

async def _verify_tasktype_helpers() -> None:
    from mvp_core.services.task_scheduler import GlobalTaskScheduler, TaskType

    class _DummyPythonWorker:
        def __init__(self):
            self.state_backplane = None
            self.submitted = []

        async def submit_task(self, task_info, func, args, kwargs):
            self.submitted.append((task_info.task_id, task_info.task_type))
            return asyncio.get_running_loop().create_future()

        async def start(self, worker_count: int = 3):
            return

        async def stop(self):
            return

        async def clean_completed_tasks(self, max_age: float) -> int:
            return 0

    scheduler = GlobalTaskScheduler()
    scheduler._python_worker = _DummyPythonWorker()
    scheduler._running = True

    await scheduler.schedule_cpu_task(lambda: 1)
    await scheduler.schedule_gpu_task(lambda: 1)

    assert any(t == TaskType.CPU_BOUND for _tid, t in scheduler._python_worker.submitted)
    assert any(t == TaskType.GPU_BOUND for _tid, t in scheduler._python_worker.submitted)


async def _main() -> None:
    await _verify_urgent_save_state()
    await _verify_backplane_recovery()
    await _verify_tasktype_helpers()


if __name__ == "__main__":
    asyncio.run(_main())
