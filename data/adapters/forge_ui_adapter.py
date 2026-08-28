"""
Forge UI API 适配器

通过 HTTP API 调用 Forge UI 进行图像生成
"""

import asyncio
import aiohttp
import os
import time
import logging
from typing import Dict, Any, Optional
from domain.interfaces.base_interfaces import ImageGenInterface

logger = logging.getLogger("ForgeUIAdapter")


class ForgeUIAdapter(ImageGenInterface):
    """Forge UI API 适配器"""
    
    def __init__(
        self, 
        base_url: str = "http://127.0.0.1:7860",
        default_steps: int = 4,
        default_width: int = 512,
        default_height: int = 512,
        timeout: float = 300.0
    ):
        self.base_url = base_url.rstrip("/")
        self.default_steps = default_steps
        self.default_width = default_width
        self.default_height = default_height
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def is_available(self) -> bool:
        """检查 Forge UI 是否可用"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/sdapi/v1/sd-models") as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Forge UI not available: {e}")
            return False
    
    async def get_models(self) -> list:
        """获取可用模型列表"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/sdapi/v1/sd-models") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.warning(f"Failed to get models: {e}")
        return []
    
    async def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "",
        steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: int = -1,
        sampler_name: str = "DPM++ 2M Karras",
        cfg_scale: float = 7.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成图像
        
        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            steps: 步数
            width: 宽度
            height: 高度
            seed: 随机种子，-1 表示随机
            sampler_name: 采样器名称
            cfg_scale: CFG 缩放
            
        Returns:
            dict: 包含 status 和 images 的字典
        """
        steps = steps or self.default_steps
        width = width or self.default_width
        height = height or self.default_height
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "ugly, blurry, low quality, distorted",
            "steps": steps,
            "width": width,
            "height": height,
            "seed": seed,
            "sampler_name": sampler_name,
            "cfg_scale": cfg_scale if cfg_scale != 7.0 else 1.0,  # Turbo 用 cfg=1
        }
        
        # 添加额外参数
        payload.update(kwargs)
        
        logger.info(f"Generating image via Forge UI: {prompt[:30]}...")
        start_time = time.time()
        
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Forge UI error: {resp.status} - {error_text}")
                
                result = await resp.json()
                
            elapsed = time.time() - start_time
            logger.info(f"Image generated in {elapsed:.2f}s")
            
            # 解码 base64 图像
            import base64
            from PIL import Image
            import io
            
            images = []
            for img_data in result.get("images", []):
                if isinstance(img_data, dict):
                    img_data = img_data.get("data") or img_data.get("image") or img_data.get("base64")
                if not isinstance(img_data, str) or not img_data:
                    continue
                s = img_data.strip()
                if s.startswith("data:") and "base64," in s:
                    s = s.split("base64,", 1)[1].strip()
                img_bytes = base64.b64decode(s)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)

            seed_out = seed
            info = result.get("info")
            if isinstance(info, dict):
                seed_out = info.get("seed", seed)
            elif isinstance(info, str) and info.strip():
                try:
                    import json
                    parsed = json.loads(info)
                    if isinstance(parsed, dict):
                        seed_out = parsed.get("seed", seed)
                except Exception:
                    seed_out = seed
            
            return {
                "status": "success",
                "images": images,
                "seed": seed_out,
                "elapsed_s": elapsed
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Forge UI timeout after {self.timeout}s")
            return {
                "status": "error",
                "error": "timeout",
                "images": []
            }
        except Exception as e:
            logger.error(f"Forge UI generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "images": []
            }
    
    async def save_image(self, image, output_path: str) -> str:
        """保存图像到文件"""
        image.save(output_path)
        return output_path
