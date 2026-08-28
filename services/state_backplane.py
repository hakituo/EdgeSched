import mmap
import os
import struct
import time
import logging
from typing import Dict, List
import threading


logger = logging.getLogger(__name__)


HEADER_FORMAT = "4sI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

PRIMARY_MAGIC = b"XYBP"
LEGACY_MAGIC = b"XYNV"
VALID_MAGICS = {PRIMARY_MAGIC, LEGACY_MAGIC}

VERSION = 1

TASK_SLOT_FORMAT = "32sii d"
TASK_SLOT_SIZE = struct.calcsize(TASK_SLOT_FORMAT)
MAX_TASK_SLOTS = 32
TASKS_OFFSET = 1024

BACKPLANE_FILE_SIZE = 1024 * 1024


class StateBackplane:
    _instances: Dict[str, "StateBackplane"] = {}
    _lock = threading.Lock()

    def __new__(cls, file_path: str | None = None):
        key = "__default__"
        if file_path:
            try:
                key = os.path.abspath(str(file_path))
            except Exception:
                key = str(file_path)

        with cls._lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = super(StateBackplane, cls).__new__(cls)
                cls._instances[key] = inst
        return inst

    def __init__(self, file_path: str | None = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

            preferred = os.path.join(data_dir, "state_backplane.dat")
            legacy = os.path.join(data_dir, "nvram.dat")
            if os.path.exists(preferred):
                self.file_path = preferred
            elif os.path.exists(legacy):
                self.file_path = legacy
            else:
                self.file_path = preferred
        else:
            self.file_path = file_path

        self._file = None
        self._mmap = None
        self._init_backplane()
        self._initialized = True
        logger.info("State Backplane initialized at %s", self.file_path)

    def _init_backplane(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "wb") as f:
                f.write(b"\x00" * BACKPLANE_FILE_SIZE)
        else:
            current_size = os.path.getsize(self.file_path)
            if current_size < BACKPLANE_FILE_SIZE:
                with open(self.file_path, "ab") as f:
                    f.write(b"\x00" * (BACKPLANE_FILE_SIZE - current_size))

        self._file = open(self.file_path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), BACKPLANE_FILE_SIZE)
        self._check_or_init_header()

    def _check_or_init_header(self):
        magic, version = struct.unpack_from(HEADER_FORMAT, self._mmap, 0)
        if magic not in VALID_MAGICS:
            logger.warning("State Backplane magic mismatch or new file. Initializing...")
            struct.pack_into(HEADER_FORMAT, self._mmap, 0, PRIMARY_MAGIC, VERSION)
            self._mmap.flush()
            return

        if int(version) != int(VERSION):
            logger.warning("State Backplane version mismatch (%s != %s). Initializing...", version, VERSION)
            struct.pack_into(HEADER_FORMAT, self._mmap, 0, PRIMARY_MAGIC, VERSION)
            self._mmap.flush()

    def register_task(self, task_id: str, priority: int, status: int = 1):
        task_bytes = task_id.encode("utf-8")[:32]

        reusable_terminal_status = {3, 4, 5}
        empty_slot = None
        reusable_candidates: list[tuple[float, int]] = []

        for i in range(MAX_TASK_SLOTS):
            offset = TASKS_OFFSET + (i * TASK_SLOT_SIZE)
            _, slot_status, _, ts = struct.unpack_from(TASK_SLOT_FORMAT, self._mmap, offset)

            if slot_status == 0:
                empty_slot = i
                break

            if slot_status in reusable_terminal_status:
                reusable_candidates.append((float(ts), i))

        chosen = None
        if empty_slot is not None:
            chosen = empty_slot
        elif reusable_candidates:
            reusable_candidates.sort(key=lambda x: x[0])
            chosen = reusable_candidates[0][1]

        if chosen is None:
            logger.warning("State Backplane task slots full, overwriting slot 0")
            chosen = 0

        offset = TASKS_OFFSET + (chosen * TASK_SLOT_SIZE)
        struct.pack_into(TASK_SLOT_FORMAT, self._mmap, offset, task_bytes, status, priority, time.time())
        return chosen

    def update_task_status(self, task_id: str, status: int):
        task_bytes = task_id.encode("utf-8")[:32]
        for i in range(MAX_TASK_SLOTS):
            offset = TASKS_OFFSET + (i * TASK_SLOT_SIZE)
            tid_raw, _, _, _ = struct.unpack_from(TASK_SLOT_FORMAT, self._mmap, offset)
            current_tid = tid_raw.rstrip(b"\x00")

            if current_tid == task_bytes:
                struct.pack_into("i", self._mmap, offset + 32, status)
                struct.pack_into("d", self._mmap, offset + 40, time.time())
                return True
        return False

    def get_active_tasks(self) -> List[Dict]:
        tasks = []
        for i in range(MAX_TASK_SLOTS):
            offset = TASKS_OFFSET + (i * TASK_SLOT_SIZE)
            tid_raw, status, priority, ts = struct.unpack_from(TASK_SLOT_FORMAT, self._mmap, offset)
            if status != 0:
                tasks.append(
                    {
                        "task_id": tid_raw.rstrip(b"\x00").decode("utf-8"),
                        "status": status,
                        "priority": priority,
                        "timestamp": ts,
                    }
                )
        return tasks

    def close(self):
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
        self._initialized = False
        key = "__default__"
        if self.file_path:
            try:
                key = os.path.abspath(str(self.file_path))
            except Exception:
                key = str(self.file_path)
        with self._lock:
            if key in self._instances:
                del self._instances[key]


def get_state_backplane() -> StateBackplane:
    return StateBackplane()
