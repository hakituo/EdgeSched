import mmap
import os
import struct
import time
import logging
from typing import Dict
import threading


logger = logging.getLogger(__name__)


BIO_FORMAT = "dddd"
BIO_SIZE = struct.calcsize(BIO_FORMAT)
BIO_FILE_SIZE = 4096


class BioStateStore:
    _instances: Dict[str, "BioStateStore"] = {}
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
                inst = super(BioStateStore, cls).__new__(cls)
                cls._instances[key] = inst
        return inst

    def __init__(self, file_path: str | None = None):
        if hasattr(self, "_initialized") and self._initialized:
            return

        if file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.file_path = os.path.join(data_dir, "bio_state.dat")
        else:
            self.file_path = str(file_path)

        self._file = None
        self._mmap = None
        self._init_store()
        self._initialized = True
        logger.info("BioStateStore initialized at %s", self.file_path)

    def _init_store(self) -> None:
        if not os.path.exists(self.file_path):
            with open(self.file_path, "wb") as f:
                f.write(b"\x00" * BIO_FILE_SIZE)
        else:
            current_size = os.path.getsize(self.file_path)
            if current_size < BIO_FILE_SIZE:
                with open(self.file_path, "ab") as f:
                    f.write(b"\x00" * (BIO_FILE_SIZE - current_size))

        self._file = open(self.file_path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), BIO_FILE_SIZE)

    def update_bio_state(self, dopamine: float, energy: float, mood: float) -> None:
        try:
            struct.pack_into(BIO_FORMAT, self._mmap, 0, dopamine, energy, mood, time.time())
            self._mmap.flush()
        except Exception as e:
            logger.error("Failed to update bio state: %s", e)

    def get_bio_state(self) -> Dict[str, float]:
        try:
            dopamine, energy, mood, ts = struct.unpack_from(BIO_FORMAT, self._mmap, 0)
            return {
                "dopamine": dopamine,
                "energy": energy,
                "mood": mood,
                "timestamp": ts,
            }
        except Exception:
            return {"dopamine": 0.0, "energy": 0.0, "mood": 0.0, "timestamp": 0.0}

    def close(self) -> None:
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


def get_bio_state_store() -> BioStateStore:
    return BioStateStore()
