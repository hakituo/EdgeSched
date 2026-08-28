import mmap
import os
import struct
import time
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)

HEADER_FORMAT = "4sI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b"XYNV"
VERSION = 1

BIO_FORMAT = "dddd"
BIO_OFFSET = 64
BIO_SIZE = struct.calcsize(BIO_FORMAT)

TASK_SLOT_FORMAT = "32sii d"
TASK_SLOT_SIZE = struct.calcsize(TASK_SLOT_FORMAT)
MAX_TASK_SLOTS = 32
TASKS_OFFSET = 1024

NVRAM_FILE_SIZE = 1024 * 1024

class NVRAMManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(NVRAMManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, file_path: str = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.file_path = os.path.join(data_dir, "nvram.dat")
        else:
            self.file_path = file_path

        self._file = None
        self._mmap = None
        self._init_nvram()
        self._initialized = True
        logger.info(f"NVRAM Manager initialized at {self.file_path}")

    def _init_nvram(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "wb") as f:
                f.write(b"\x00" * NVRAM_FILE_SIZE)
        else:
            current_size = os.path.getsize(self.file_path)
            if current_size < NVRAM_FILE_SIZE:
                with open(self.file_path, "ab") as f:
                    f.write(b"\x00" * (NVRAM_FILE_SIZE - current_size))

        self._file = open(self.file_path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), NVRAM_FILE_SIZE)
        self._check_or_init_header()

    def _check_or_init_header(self):
        magic, version = struct.unpack_from(HEADER_FORMAT, self._mmap, 0)
        if magic != MAGIC:
            logger.warning("NVRAM magic mismatch or new file. Initializing...")
            struct.pack_into(HEADER_FORMAT, self._mmap, 0, MAGIC, VERSION)
            self._mmap.flush()

    def update_bio_state(self, dopamine: float, energy: float, mood: float):
        try:
            struct.pack_into(BIO_FORMAT, self._mmap, BIO_OFFSET, dopamine, energy, mood, time.time())
        except Exception as e:
            logger.error(f"Failed to update bio state: {e}")

    def get_bio_state(self) -> Dict[str, float]:
        try:
            dopamine, energy, mood, ts = struct.unpack_from(BIO_FORMAT, self._mmap, BIO_OFFSET)
            return {
                "dopamine": dopamine,
                "energy": energy,
                "mood": mood,
                "timestamp": ts
            }
        except Exception:
            return {"dopamine": 0.0, "energy": 0.0, "mood": 0.0, "timestamp": 0.0}

    def register_task(self, task_id: str, priority: int, status: int = 1):
        task_bytes = task_id.encode('utf-8')[:32]
        
        for i in range(MAX_TASK_SLOTS):
            offset = TASKS_OFFSET + (i * TASK_SLOT_SIZE)
            _, slot_status, _, _ = struct.unpack_from(TASK_SLOT_FORMAT, self._mmap, offset)
            
            if slot_status == 0:
                struct.pack_into(TASK_SLOT_FORMAT, self._mmap, offset, task_bytes, status, priority, time.time())
                return i
        
        logger.warning("NVRAM Task slots full, overwriting slot 0")
        struct.pack_into(TASK_SLOT_FORMAT, self._mmap, TASKS_OFFSET, task_bytes, status, priority, time.time())
        return 0

    def update_task_status(self, task_id: str, status: int):
        task_bytes = task_id.encode('utf-8')[:32]
        for i in range(MAX_TASK_SLOTS):
            offset = TASKS_OFFSET + (i * TASK_SLOT_SIZE)
            tid_raw, _, _, _ = struct.unpack_from(TASK_SLOT_FORMAT, self._mmap, offset)
            current_tid = tid_raw.rstrip(b'\x00')
            
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
                tasks.append({
                    "task_id": tid_raw.rstrip(b'\x00').decode('utf-8'),
                    "status": status,
                    "priority": priority,
                    "timestamp": ts
                })
        return tasks

    def close(self):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

def get_nvram_manager() -> NVRAMManager:
    return NVRAMManager()
