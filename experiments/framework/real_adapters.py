"""实验框架使用的真实生产适配器导出。

这里不实现模拟逻辑；Mock 必须从 ``mock_tasks`` 或
``data.adapters.mock_tts_adapter`` 显式导入。
"""

from data.adapters.qwen3_tts_adapter import Qwen3TTSAdapter

__all__ = ["Qwen3TTSAdapter"]
