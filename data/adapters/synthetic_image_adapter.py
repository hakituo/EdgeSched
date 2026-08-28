import logging
import asyncio
import os
import time
import random
from domain.interfaces.base_interfaces import ImageGenInterface

logger = logging.getLogger("SyntheticImageAdapter")

class SyntheticImageAdapter(ImageGenInterface):
    """
    Synthetic Image Adapter (Fallback)
    Generates placeholder images (.ppm) when the main engine is unavailable.
    """
    def __init__(self, latency_mean: float = 2.0, latency_std: float = 0.5):
        self.latency_mean = latency_mean
        self.latency_std = latency_std

    def _get_output_dir(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mvp_root = os.path.dirname(os.path.dirname(current_dir))
        out_dir = os.path.join(mvp_root, "output", "images")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _write_ppm(self, path: str, width: int = 64, height: int = 64) -> None:
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(header)
            f.write(buf)

    async def generate_image(self, prompt: str, **kwargs) -> str:
        """
        Simulate image generation with CPU load.
        """
        logger.info(f"Generating synthetic image for prompt: {prompt[:20]}...")
        
        # Simulate latency
        delay = max(0.1, random.gauss(self.latency_mean, self.latency_std))
        
        # Simulate CPU load (optional: simple spin lock or math)
        # We use sleep to simulate I/O bound wait (GPU wait), 
        # and a bit of math for CPU overhead if needed.
        
        start = time.time()
        # Simulate 10% CPU work, 90% GPU wait
        cpu_time = delay * 0.1
        gpu_time = delay * 0.9
        
        # CPU Burn
        end_cpu = start + cpu_time
        while time.time() < end_cpu:
            _ = [x**2 for x in range(1000)]
            
        # GPU Wait
        await asyncio.sleep(gpu_time)
        
        logger.info(f"Synthetic image generation complete in {delay:.2f}s")
        out_dir = self._get_output_dir()
        filename = f"synthetic_{int(time.time() * 1000)}.ppm"
        path = os.path.join(out_dir, filename)
        await asyncio.to_thread(self._write_ppm, path)
        return path
