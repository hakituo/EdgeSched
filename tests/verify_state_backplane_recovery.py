import os
import sys
import time
import multiprocessing as mp


def _legacy_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _mvp_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


sys.path.append(_legacy_dir())

from mvp_core.services.state_backplane import StateBackplane


def _writer(file_path: str, task_count: int, queue: mp.Queue) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)
    backplane = StateBackplane(file_path)
    for i in range(task_count):
        task_id = f"task_{i:04d}"
        backplane.register_task(task_id, 1, 2)
        backplane.update_task_status(task_id, 3)
    backplane.close()
    queue.put({"task_count": task_count})


def _reader(file_path: str, queue: mp.Queue) -> None:
    backplane = StateBackplane(file_path)
    tasks = backplane.get_active_tasks()
    backplane.close()
    queue.put({"recovered": len(tasks)})


def main() -> None:
    task_count = 8
    data_dir = os.path.join(_mvp_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "state_backplane_recovery_check.dat")

    writer_queue: mp.Queue = mp.Queue()
    reader_queue: mp.Queue = mp.Queue()

    writer = mp.Process(target=_writer, args=(file_path, task_count, writer_queue))
    reader = mp.Process(target=_reader, args=(file_path, reader_queue))

    writer.start()
    writer.join()
    time.sleep(0.1)
    reader.start()
    reader.join()

    writer_stats = writer_queue.get() if not writer_queue.empty() else {}
    reader_stats = reader_queue.get() if not reader_queue.empty() else {}

    expected = writer_stats.get("task_count", task_count)
    recovered = reader_stats.get("recovered", 0)

    ok = expected == recovered and recovered > 0
    payload = {
        "ok": ok,
        "expected": expected,
        "recovered": recovered,
        "file_path": file_path,
    }
    print(payload)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

