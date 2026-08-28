import mmap
import os
import struct
import time
import ctypes

class VirtualNVRAM:
    """
    基于内存映射文件 (mmap) 的虚拟 NVRAM 模拟器。
    用于在 mvp_core 实验中模拟字节寻址、持久化内存。
    """
    
    # 模拟 BiologicalSystem 的结构体布局 (POD)
    # Neurotransmitter: 5 floats (20 bytes)
    # energy: 1 float (4 bytes)
    # sleep_debt: 1 float (4 bytes)
    # phase: 1 int (4 bytes)
    # total: 32 bytes
    BIO_STRUCT_FMT = "fffffff i" 
    BIO_STRUCT_SIZE = struct.calcsize(BIO_STRUCT_FMT)

    def __init__(self, file_path="nvram_sim.bin", size=1024*1024): # 默认 1MB
        self.file_path = file_path
        self.size = size
        self._setup_mmap()

    def _setup_mmap(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "wb") as f:
                f.write(b'\x00' * self.size)
        
        self.file = open(self.file_path, "r+b")
        self.mm = mmap.mmap(self.file.fileno(), self.size)
        print(f"[NVRAM] 已挂载虚拟 NVRAM: {self.file_path} ({self.size} bytes)")

    def write_bio_state(self, dopamine, serotonin, norepinephrine, oxytocin, cortisol, energy, sleep_debt, phase):
        """写入生物系统状态到 NVRAM 预定偏移量"""
        data = struct.pack(self.BIO_STRUCT_FMT, 
                           dopamine, serotonin, norepinephrine, oxytocin, cortisol, 
                           energy, sleep_debt, phase)
        self.mm[0:self.BIO_STRUCT_SIZE] = data
        self.mm.flush()

    def read_bio_state(self):
        """从 NVRAM 读取生物系统状态"""
        data = self.mm[0:self.BIO_STRUCT_SIZE]
        return struct.unpack(self.BIO_STRUCT_FMT, data)

    def close(self):
        self.mm.close()
        self.file.close()

if __name__ == "__main__":
    # 简单的持久化测试
    nvram = VirtualNVRAM("mvp_nvram_test.bin")
    
    # 模拟一次写入
    print("写入初始状态: 多巴胺=0.8, 能量=0.9")
    nvram.write_bio_state(0.8, 0.5, 0.5, 0.5, 0.3, 0.9, 0.1, 1)
    
    # 读取验证
    state = nvram.read_bio_state()
    print(f"当前读取状态: {state}")
    
    nvram.close()
    print("NVRAM 已关闭。下次运行将保持该状态。")
